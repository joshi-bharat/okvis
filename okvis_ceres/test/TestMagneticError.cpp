#include <gtest/gtest.h>

#include <Eigen/Core>

#include "okvis/Time.hpp"
#include "okvis/ceres/MagneticError.hpp"
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

  mag_yaw_cost_func->Evaluate(parameters, residuals.data(), NULL);
  OKVIS_ASSERT_TRUE(Exception, residuals.norm() < 1e-10, "residuals should be zero with same params");

  cout << "[  OK  ] Residual Check" << endl;

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

TEST(okvisTestSuite, MagneticErrorPose) {
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error);

  using namespace std;
  // *****************************************************************************
  // Magnetic field in the nav frame (NED), with units of nT.
  Eigen::Vector3d nM(22653.29982, -1956.83010, 44202.47862);

  // Assumed scale factor (scales a unit vector to magnetometer units of nT).
  double scale = 255.0 / 50000.0;

  // Ground truth Pose2/Pose3 in the nav frame.
  okvis::kinematics::Transformation T_WS;
  Eigen::Matrix<double, 6, 1> delta;
  delta << -3, 12, 5, 0, 0, -0.1;
  T_WS.oplus(delta);

  Eigen::Vector3d bias(10, -10, 50);
  Eigen::Vector3d dir(nM.normalized());
  Eigen::Vector3d scaled(scale * dir);

  // Compute the measured field in the sensor frame.
  Eigen::Vector3d measured = T_WS.inverse() * scaled + bias;
  cout << "nM: " << nM.transpose() << endl;
  cout << "scaled: " << scaled.transpose() << endl;
  cout << "measured: " << measured.transpose() << endl;
  cout << "s: " << scale << endl;
  cout << "direction: " << dir.transpose() << endl;
  double variance = 1.0;

  okvis::ceres::MagPoseError* mag_cost_func = new okvis::ceres::MagPoseError(measured, scale, dir, bias, variance);

  cout << "[  OK  ] Set up Magnetometer Pose Error" << endl;

  okvis::Time t_0;
  okvis::ceres::PoseParameterBlock poseParameterBlock(T_WS, 0, t_0);

  double* parameters[1];
  parameters[0] = poseParameterBlock.parameters();

  Eigen::Matrix<double, 3, 1> residuals;

  mag_cost_func->Evaluate(parameters, residuals.data(), NULL);
  OKVIS_ASSERT_TRUE(Exception, residuals.norm() < 1e-10, "residuals should be zero with same params");

  cout << "[  OK  ] Residual Check Pass" << endl;

  okvis::kinematics::Transformation T_disturb;
  T_disturb.setRandom(1, 0.02);

  cout << "T_WS Earlier" << T_WS.T() << endl;
  T_WS = T_WS * T_disturb;
  cout << "T_WS Current" << T_WS.T() << endl;
  variance = 2 * 2;

  poseParameterBlock.setEstimate(T_WS);

  mag_cost_func->setInformation(Eigen::Matrix3d::Identity() * 1.0 / variance);

  ceres::Problem problem;
  ceres::LocalParameterization* poseLocalParameterization3d = new okvis::ceres::PoseLocalParameterization3d;

  problem.AddParameterBlock(poseParameterBlock.parameters(), okvis::ceres::PoseParameterBlock::Dimension);
  problem.SetParameterization(poseParameterBlock.parameters(), poseLocalParameterization3d);
  problem.AddResidualBlock(mag_cost_func, NULL, poseParameterBlock.parameters());

  // Run the solver!
  std::cout << "Run the solver... " << std::endl;
  ::ceres::Solver::Options options;
  // options.check_gradients=true;
  // options.numeric_derivative_relative_step_size = 1e-6;
  // options.gradient_check_relative_precision=1e-2;
  options.minimizer_progress_to_stdout = false;
  ::FLAGS_stderrthreshold = google::WARNING;  // enable console warnings (Jacobian verification)
  ::ceres::Solver::Summary summary;
  ::ceres::Solve(options, &problem, &summary);

  cout << summary.BriefReport() << endl;
  // make sure it converged
  OKVIS_ASSERT_TRUE(Exception, summary.final_cost < 1e-2, "cost not reducible");
  OKVIS_ASSERT_TRUE(Exception,
                    2 * (T_WS.q() * poseParameterBlock.estimate().q().inverse()).vec().norm() < 1e-2,
                    "quaternions not close enough");
}
