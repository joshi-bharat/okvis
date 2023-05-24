#include <gtest/gtest.h>

#include <Eigen/Core>

#include "okvis/Time.hpp"
#include "okvis/ceres/MagneticError.hpp"
#include "okvis/ceres/PoseError.hpp"
#include "okvis/ceres/PoseParameterBlock.hpp"
#include "okvis/kinematics/Transformation.hpp"

TEST(okvisTestSuite, MagneticErrorYaw) {
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error);

  using namespace std;
  Eigen::Vector3d nM(22653.29982, -1956.83010, 44202.47862);
  double scale = 255.0 / 50000.0;
  double yaw = 2.0;
  Eigen::Vector3d bias(10, -10, 50);

  Eigen::Vector3d scaled(scale * nM);
  Eigen::Matrix<double, 6, 1> delta;
  delta << 0, 0, 0, 0, 0, yaw;
  okvis::kinematics::Transformation T_WS;
  T_WS.oplus(delta);
  Eigen::Vector3d measured = T_WS.inverse() * scaled + bias;

  double s(scale * nM.norm());
  Eigen::Vector3d nM_normalized(nM.normalized());

  cout << "nM: " << nM.transpose() << endl;
  cout << "scaled: " << scaled.transpose() << endl;
  cout << "measured: " << measured.transpose() << endl;
  cout << "s: " << s << endl;
  cout << "norm: " << nM_normalized.transpose() << endl;
  double variance = 1.0;

  cout << "[  OK  ] Set up Magnetometer Yaw Problem" << endl;

  okvis::ceres::MagErrorYaw* mag_yaw_cost_func =
      new okvis::ceres::MagErrorYaw(measured, s, nM_normalized, bias, variance);

  double angle[1];
  double* parameters[1];

  angle[0] = yaw;
  parameters[0] = angle;
  Eigen::Matrix<double, 3, 1> residuals;

  double* jacobians[1];
  Eigen::Matrix<double, 3, 1> j;
  jacobians[0] = j.data();

  mag_yaw_cost_func->Evaluate(parameters, residuals.data(), jacobians);
  OKVIS_ASSERT_TRUE(Exception, residuals.norm() < 1e-10, "residuals should be zero with same params");

  cout << "[  OK  ] Residual Check" << endl;
  // cout << "Residuals: \n" << residuals << endl;
  // cout << "Jacobian: \n" << j << endl;

  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  // Change angle to 0.0 to test the problem
  // Assume noise
  angle[0] = 0.0;
  variance = 5 * 5;
  mag_yaw_cost_func->setInformation(Eigen::Matrix3d::Identity() * 1.0 / variance);
  ceres::Problem* problem = new ceres::Problem();
  problem->AddParameterBlock(angle, 1);
  problem->AddResidualBlock(mag_yaw_cost_func, NULL, angle);

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);
  cout << summary.BriefReport() << endl;

  OKVIS_ASSERT_TRUE(Exception, (angle[0] - yaw) < 1e-5, "Yaw  not close enough to original value");
}

TEST(okvisTestSuite, MagneticErrorYawLocal) {
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error);

  using namespace std;
  Eigen::Vector3d nM(22653.29982, -1956.83010, 44202.47862);
  double scale = 255.0 / 50000.0;
  double yaw = 1.0;
  Eigen::Vector3d bias(10, -10, 50);

  Eigen::Vector3d scaled(scale * nM);
  Eigen::Matrix<double, 6, 1> delta;
  delta << 0, 0, 0, 0, 0, yaw;
  okvis::kinematics::Transformation T_WS;
  T_WS.oplus(delta);
  Eigen::Vector3d measured = T_WS.inverse() * scaled + bias;

  double s(scale * nM.norm());
  Eigen::Vector3d nM_normalized(nM.normalized());

  cout << "nM: " << nM.transpose() << endl;
  cout << "scaled: " << scaled.transpose() << endl;
  cout << "measured: " << measured.transpose() << endl;
  cout << "s: " << s << endl;
  cout << "norm: " << nM_normalized.transpose() << endl;
  double variance = 1.0;

  cout << "[  OK  ] Set up Magnetometer Yaw Problem" << endl;

  okvis::ceres::MagErrorYawLocal* mag_yaw_cost_func =
      new okvis::ceres::MagErrorYawLocal(measured, s, nM_normalized, bias, variance);

  okvis::Time t_0;
  okvis::ceres::PoseParameterBlock poseParameterBlock(T_WS, 0, t_0);

  double* parameters[1];
  Eigen::Matrix<double, 3, 1> residuals;
  parameters[0] = poseParameterBlock.parameters();

  Eigen::Matrix<double, 3, 7, Eigen::RowMajor> j;
  double* jacobians[1];
  jacobians[0] = j.data();

  mag_yaw_cost_func->Evaluate(parameters, residuals.data(), jacobians);
  OKVIS_ASSERT_TRUE(Exception, residuals.norm() < 1e-10, "residuals should be zero with same params");

  cout << "[  OK  ] Residual Check" << endl;
  // cout << "Residuals: \n" << residuals << endl;
  // cout << "Jacobian: \n" << j << endl;

  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  // Set the estimate to zero Yaw to test the algorithm
  okvis::kinematics::Transformation T_init;
  poseParameterBlock.setEstimate(T_init);

  mag_yaw_cost_func->setInformation(Eigen::Matrix3d::Identity() * 1.0 / variance);
  ceres::Problem* problem = new ceres::Problem();

  ceres::LocalParameterization* poseLocalParameterizationYaw = new okvis::ceres::PoseLocalParameterizationYaw;

  problem->AddParameterBlock(poseParameterBlock.parameters(), poseParameterBlock.dimension());
  problem->SetParameterization(poseParameterBlock.parameters(), poseLocalParameterizationYaw);
  problem->AddResidualBlock(mag_yaw_cost_func, NULL, poseParameterBlock.parameters());

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);
  cout << summary.BriefReport() << endl;

  OKVIS_ASSERT_TRUE(Exception, summary.final_cost < 1e-2, "cost not reducible");
  OKVIS_ASSERT_TRUE(Exception,
                    2 * (T_WS.q() * poseParameterBlock.estimate().q().inverse()).vec().norm() < 1e-2,
                    "quaternions not close enough");
  OKVIS_ASSERT_TRUE(
      Exception, (T_WS.r() - poseParameterBlock.estimate().r()).norm() < 0.04, "translation not close enough");

  cout << poseParameterBlock.estimate().T() << endl;
}

TEST(okvisTestSuite, MagneticErrorPose) {
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error);

  using namespace std;
  using namespace std;
  Eigen::Vector3d nM(22653.29982, -1956.83010, 44202.47862);
  double scale = 255.0 / 50000.0;
  double yaw = 0.5;
  Eigen::Vector3d bias(10, -10, 50);

  Eigen::Vector3d scaled(scale * nM);
  Eigen::Matrix<double, 6, 1> delta;
  delta << 0, 0, 0, 0.1, 0.1, yaw;
  okvis::kinematics::Transformation T_WS;
  T_WS.oplus(delta);
  Eigen::Vector3d measured = T_WS.inverse() * scaled + bias;

  double s(scale * nM.norm());
  Eigen::Vector3d nM_normalized(nM.normalized());

  cout << "nM: " << nM.transpose() << endl;
  cout << "scaled: " << scaled.transpose() << endl;
  cout << "measured: " << measured.transpose() << endl;
  cout << "s: " << s << endl;
  cout << "norm: " << nM_normalized.transpose() << endl;

  double variance = 1.0;
  okvis::ceres::MagPoseError* mag_cost_func =
      new okvis::ceres::MagPoseError(measured, s, nM_normalized, bias, variance);

  cout << "[  OK  ] Set up Magnetometer Pose Error" << endl;

  okvis::Time t_0;
  okvis::ceres::PoseParameterBlock poseParameterBlock(T_WS, 0, t_0);

  double* parameters[1];
  double* jacobians[1];

  Eigen::Matrix<double, 3, 1> residuals;
  Eigen::Matrix<double, 3, 7, Eigen::RowMajor> j;

  parameters[0] = poseParameterBlock.parameters();
  jacobians[0] = j.data();

  mag_cost_func->Evaluate(parameters, residuals.data(), jacobians);
  OKVIS_ASSERT_TRUE(Exception, residuals.norm() < 1e-10, "residuals should be zero with same params");

  cout << "[  OK  ] Residual Check Pass" << endl;
  // cout << "Residuals: \n" << residuals << endl;
  // cout << "Jacobian: \n" << j << endl;

  okvis::kinematics::Transformation T_dist;
  T_dist.setRandom(0, 0.1);
  T_dist = T_WS * T_dist;
  poseParameterBlock.setEstimate(T_dist);

  mag_cost_func->setInformation(Eigen::Matrix3d::Identity() * 1.0 / variance);

  ceres::Problem problem;
  ceres::LocalParameterization* poseLocalParameterization3d = new okvis::ceres::PoseLocalParameterization3d;

  problem.AddParameterBlock(poseParameterBlock.parameters(), okvis::ceres::PoseParameterBlock::Dimension);
  problem.SetParameterization(poseParameterBlock.parameters(), poseLocalParameterization3d);
  problem.AddResidualBlock(mag_cost_func, NULL, poseParameterBlock.parameters());

  // Run the solver!
  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ::ceres::Solver::Summary summary;
  ::ceres::Solve(options, &problem, &summary);

  cout << summary.BriefReport() << endl;

  // okvis::kinematics::Transformation estimate = poseParameterBlock.estimate();
  // Eigen::Vector3d m_est = estimate.inverse() * scaled + bias;
  // Eigen::Vector3d m_dist = T_dist.inverse() * scaled + bias;

  // cout << "error: " << (m_est - measured).norm() << endl;
  // cout << "error: " << (m_dist - measured).norm() << endl;

  // cout << (T_WS.q() * estimate.q().inverse()).vec().norm() << endl;
  // cout << (T_WS.q() * T_dist.q().inverse()).vec().norm() << endl;

  // make sure it converged
  OKVIS_ASSERT_TRUE(Exception, summary.final_cost < 1e-2, "cost not reducible");
  OKVIS_ASSERT_TRUE(Exception,
                    2 * (T_WS.q() * poseParameterBlock.estimate().q().inverse()).vec().norm() < 1e-2,
                    "quaternions not close enough");
}
