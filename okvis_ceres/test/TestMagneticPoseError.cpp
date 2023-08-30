
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "okvis/FrameTypedefs.hpp"
#include "okvis/Time.hpp"
#include "okvis/assert_macros.hpp"
#include "okvis/ceres/HomogeneousPointLocalParameterization.hpp"
#include "okvis/ceres/HomogeneousPointParameterBlock.hpp"
#include "okvis/ceres/ImuError.hpp"
#include "okvis/ceres/MagneticPoseError.hpp"
#include "okvis/ceres/PoseError.hpp"
#include "okvis/ceres/PoseLocalParameterization.hpp"
#include "okvis/ceres/PoseParameterBlock.hpp"
#include "okvis/ceres/SpeedAndBiasError.hpp"
#include "okvis/ceres/SpeedAndBiasParameterBlock.hpp"
#include "okvis/kinematics/Transformation.hpp"

const double jacobianTolerance = 1.0e-3;

TEST(okvisTestSuite, MagneticPoseError) {
  // initialize random number generator
  // srand((unsigned int)time(0));  // disabled: make unit tests deterministic...

  // Build the problem.
  ::ceres::Problem problem;

  // check errors
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error);

  // set the imu parameters
  okvis::ImuParameters imuParameters;
  imuParameters.a0.setZero();
  imuParameters.g = 9.81;
  imuParameters.a_max = 1000.0;
  imuParameters.g_max = 1000.0;
  imuParameters.rate = 200;  // 1 kHz
  imuParameters.sigma_g_c = 6.0e-3;
  imuParameters.sigma_a_c = 2.0e-3;
  imuParameters.sigma_gw_c = 3.0e-5;
  imuParameters.sigma_aw_c = 2.0e-5;
  imuParameters.tau = 3600.0;

  // std::srand(std::time(nullptr));  // use current time as seed for random generator

  // generate random motion
  const double w_omega_S_x = Eigen::internal::random(0.1, 10.0);  // circular frequency
  const double w_omega_S_y = Eigen::internal::random(0.1, 10.0);  // circular frequency
  const double w_omega_S_z = Eigen::internal::random(0.1, 10.0);  // circular frequency
  const double p_omega_S_x = Eigen::internal::random(0.0, M_PI);  // phase
  const double p_omega_S_y = Eigen::internal::random(0.0, M_PI);  // phase
  const double p_omega_S_z = Eigen::internal::random(0.0, M_PI);  // phase
  const double m_omega_S_x = Eigen::internal::random(0.1, 1.0);   // magnitude
  const double m_omega_S_y = Eigen::internal::random(0.1, 1.0);   // magnitude
  const double m_omega_S_z = Eigen::internal::random(0.1, 1.0);   // magnitude
  const double w_a_W_x = Eigen::internal::random(0.1, 10.0);
  const double w_a_W_y = Eigen::internal::random(0.1, 10.0);
  const double w_a_W_z = Eigen::internal::random(0.1, 10.0);
  const double p_a_W_x = Eigen::internal::random(0.1, M_PI);
  const double p_a_W_y = Eigen::internal::random(0.1, M_PI);
  const double p_a_W_z = Eigen::internal::random(0.1, M_PI);
  const double m_a_W_x = Eigen::internal::random(0.1, 10.0);
  const double m_a_W_y = Eigen::internal::random(0.1, 10.0);
  const double m_a_W_z = Eigen::internal::random(0.1, 10.0);

  // generate randomized measurements - duration 10 seconds
  const double duration = 1.0;
  okvis::ImuMeasurementDeque imuMeasurements;
  okvis::kinematics::Transformation T_WS;
  // T_WS.setRandom();

  // time increment
  const double dt = 1.0 / double(imuParameters.rate);  // time discretization

  // states
  Eigen::Quaterniond q = T_WS.q();
  Eigen::Vector3d r = T_WS.r();
  okvis::SpeedAndBias speedAndBias;
  speedAndBias.setZero();
  Eigen::Vector3d v = speedAndBias.head<3>();

  // start
  okvis::kinematics::Transformation T_WS_0;
  okvis::SpeedAndBias speedAndBias_0;
  okvis::Time t_0;

  // end
  okvis::kinematics::Transformation T_WS_1;
  okvis::SpeedAndBias speedAndBias_1;
  okvis::Time t_1;

  Eigen::Vector3d magnetic_field(-3.0604, 22.7441, 42.5456);
  Eigen::Vector3d mag = Eigen::Vector3d::Zero();

  okvis::MagnetometerParameters magnetometer_parameters;
  magnetometer_parameters.rate = 20;
  magnetometer_parameters.sigma_m_c = 1.0;

  okvis::MagnetometerMeasurementDeque magnetic_measurements;
  okvis::MagnetometerMeasurement start_magnetic_field, end_magnetic_field;

  for (size_t i = 0; i < size_t(duration * imuParameters.rate); ++i) {
    double time = double(i) / imuParameters.rate;
    if (i == 10) {  // set this as starting pose
      T_WS_0 = T_WS;
      speedAndBias_0 = speedAndBias;
      t_0 = okvis::Time(time);
      start_magnetic_field = okvis::MagnetometerMeasurement(t_0, okvis::MagnetometerReading(mag));
    }
    if (i == size_t(duration * imuParameters.rate) - 10) {  // set this as starting pose
      T_WS_1 = T_WS;
      speedAndBias_1 = speedAndBias;
      t_1 = okvis::Time(time);
      end_magnetic_field.measurement.flux_density_ += magnetometer_parameters.sigma_m_c * Eigen::Vector3d::Random();
      end_magnetic_field = okvis::MagnetometerMeasurement(t_0, okvis::MagnetometerReading(mag));
    }

    Eigen::Vector3d omega_S(m_omega_S_x * sin(w_omega_S_x * time + p_omega_S_x),
                            m_omega_S_y * sin(w_omega_S_y * time + p_omega_S_y),
                            m_omega_S_z * sin(w_omega_S_z * time + p_omega_S_z));
    Eigen::Vector3d a_W(m_a_W_x * sin(w_a_W_x * time + p_a_W_x),
                        m_a_W_y * sin(w_a_W_y * time + p_a_W_y),
                        m_a_W_z * sin(w_a_W_z * time + p_a_W_z));

    // omega_S.setZero();
    // a_W.setZero();

    Eigen::Quaterniond dq;

    // propagate orientation
    const double theta_half = omega_S.norm() * dt * 0.5;
    const double sinc_theta_half = okvis::kinematics::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * 0.5 * dt * omega_S;
    dq.w() = cos_theta_half;
    q = q * dq;

    // propagate speed
    v += dt * a_W;

    // propagate position
    r += dt * v;

    // T_WS
    T_WS = okvis::kinematics::Transformation(r, q);

    // speedAndBias - v only, obviously, since this is the Ground Truth
    speedAndBias.head<3>() = v;

    // generate measurements
    Eigen::Vector3d gyr = omega_S + imuParameters.sigma_g_c / sqrt(dt) * Eigen::Vector3d::Random();
    Eigen::Vector3d acc = T_WS.inverse().C() * (a_W + Eigen::Vector3d(0, 0, imuParameters.g)) +
                          imuParameters.sigma_a_c / sqrt(dt) * Eigen::Vector3d::Random();
    imuMeasurements.push_back(okvis::ImuMeasurement(okvis::Time(time), okvis::ImuSensorReadings(gyr, acc)));

    mag = T_WS.inverse().C() * magnetic_field;
    if (i > 0 && i % magnetometer_parameters.rate == 0) {
      magnetic_measurements.push_back(
          okvis::MagnetometerMeasurement(okvis::Time(time), okvis::MagnetometerReading(mag)));
    }
  }

  ::ceres::CostFunction* cost_func_mag = new okvis::ceres::MagneticPoseError(
      {start_magnetic_field.measurement.flux_density_, end_magnetic_field.measurement.flux_density_},
      magnetometer_parameters.sigma_m_c * magnetometer_parameters.sigma_m_c);

  // create the pose parameter blocks
  okvis::kinematics::Transformation T_disturb;
  T_disturb.setRandom(1, 0.05);
  okvis::kinematics::Transformation T_WS_1_disturbed = T_WS_1 * T_disturb;
  okvis::kinematics::Transformation T_WS_0_disturbed = T_WS_0;
  okvis::ceres::PoseParameterBlock poseParameterBlock_0(T_WS_0_disturbed, 0, t_0);
  okvis::ceres::PoseParameterBlock poseParameterBlock_1(T_WS_1_disturbed, 2, t_1);
  problem.AddParameterBlock(poseParameterBlock_0.parameters(), okvis::ceres::PoseParameterBlock::Dimension);
  problem.AddParameterBlock(poseParameterBlock_1.parameters(), okvis::ceres::PoseParameterBlock::Dimension);

  // create the speed and bias
  okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_0(speedAndBias_0, 1, t_0);
  okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_1(speedAndBias_1, 3, t_1);
  problem.AddParameterBlock(speedAndBiasParameterBlock_0.parameters(),
                            okvis::ceres::SpeedAndBiasParameterBlock::Dimension);
  problem.AddParameterBlock(speedAndBiasParameterBlock_1.parameters(),
                            okvis::ceres::SpeedAndBiasParameterBlock::Dimension);

  // let's use our own local quaternion perturbation
  std::cout << "setting local parameterization for pose... " << std::flush;
  ::ceres::LocalParameterization* poseLocalParameterization = new okvis::ceres::PoseLocalParameterization;
  problem.SetParameterization(poseParameterBlock_0.parameters(), poseLocalParameterization);
  problem.SetParameterization(poseParameterBlock_1.parameters(), poseLocalParameterization);
  std::cout << " [ OK ] " << std::endl;

  // create the Imu error term
  okvis::ceres::ImuError* cost_function_imu = new okvis::ceres::ImuError(imuMeasurements, imuParameters, t_0, t_1);
  problem.AddResidualBlock(cost_function_imu,
                           NULL,
                           poseParameterBlock_0.parameters(),
                           speedAndBiasParameterBlock_0.parameters(),
                           poseParameterBlock_1.parameters(),
                           speedAndBiasParameterBlock_1.parameters());

  // let's also add some priors to check this alongside
  ::ceres::CostFunction* cost_function_pose = new okvis::ceres::PoseError(T_WS_0, 1e-12, 1e-4);  // pose prior...
  problem.AddResidualBlock(cost_function_pose, NULL, poseParameterBlock_0.parameters());
  ::ceres::CostFunction* cost_function_speedAndBias =
      new okvis::ceres::SpeedAndBiasError(speedAndBias_0, 1e-12, 1e-12, 1e-12);  // speed and biases prior...
  problem.AddResidualBlock(cost_function_speedAndBias, NULL, speedAndBiasParameterBlock_0.parameters());

  double* parameters[2];
  parameters[0] = poseParameterBlock_0.parameters();
  parameters[1] = poseParameterBlock_1.parameters();

  // jacobian blocks

  double* jacobains[2];
  Eigen::Matrix<double, 3, 7, Eigen::RowMajor> J0;
  Eigen::Matrix<double, 3, 7, Eigen::RowMajor> J1;
  jacobains[0] = J0.data();
  jacobains[1] = J1.data();

  double* jacobianMinimals[2];
  Eigen::Matrix<double, 3, 6, Eigen::RowMajor> J0min;
  Eigen::Matrix<double, 3, 6, Eigen::RowMajor> J1min;
  jacobianMinimals[0] = J0min.data();
  jacobianMinimals[1] = J1min.data();

  Eigen::Vector3d residuals;
  static_cast<okvis::ceres::MagneticPoseError*>(cost_func_mag)
      ->EvaluateWithMinimalJacobians(parameters, residuals.data(), jacobains, jacobianMinimals);

  std::cout << "JO: " << J0 << std::endl;
  //   and now num - diff :
  double dx = 1e-6;

  Eigen::Matrix<double, 3, 6> J0_numDiff;
  J0_numDiff.setZero();
  for (size_t i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> dp_0;
    Eigen::Matrix<double, 3, 1> residuals_p;
    Eigen::Matrix<double, 3, 1> residuals_m;
    dp_0.setZero();
    dp_0[i] = dx;
    poseLocalParameterization->Plus(parameters[0], dp_0.data(), parameters[0]);
    // std::cout<<poseParameterBlock_0.estimate().T()<<std::endl;
    cost_func_mag->Evaluate(parameters, residuals_p.data(), NULL);
    // std::cout<<residuals_p.transpose()<<std::endl;
    poseParameterBlock_0.setEstimate(T_WS_0_disturbed);  // reset
    dp_0[i] = -dx;
    // std::cout<<residuals.transpose()<<std::endl;
    poseLocalParameterization->Plus(parameters[0], dp_0.data(), parameters[0]);
    // std::cout<<poseParameterBlock_0.estimate().T()<<std::endl;
    cost_func_mag->Evaluate(parameters, residuals_m.data(), NULL);
    // std::cout<<residuals_m.transpose()<<std::endl;
    poseParameterBlock_0.setEstimate(T_WS_0_disturbed);  // reset
    J0_numDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
  }

  OKVIS_ASSERT_TRUE(Exception,
                    (J0min - J0_numDiff).norm() < jacobianTolerance,
                    "minimal Jacobian 0 = \n"
                        << J0min << std::endl
                        << "numDiff minimal Jacobian 0 = \n"
                        << J0_numDiff);
  // std::cout << "minimal Jacobian 0 = \n" << J0min << std::endl;
  // std::cout << "numDiff minimal Jacobian 0 = \n" << J0_numDiff << std::endl;
  Eigen::Matrix<double, 7, 6, Eigen::RowMajor> Jplus;
  poseLocalParameterization->ComputeJacobian(parameters[0], Jplus.data());

  Eigen::Matrix<double, 3, 6> J1_numDiff;
  J1_numDiff.setZero();
  for (size_t i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> dp_1;
    Eigen::Matrix<double, 3, 1> residuals_p;
    Eigen::Matrix<double, 3, 1> residuals_m;
    dp_1.setZero();
    dp_1[i] = dx;
    poseLocalParameterization->Plus(parameters[1], dp_1.data(), parameters[1]);
    cost_func_mag->Evaluate(parameters, residuals_p.data(), NULL);
    poseParameterBlock_1.setEstimate(T_WS_1_disturbed);  // reset
    dp_1[i] = -dx;
    poseLocalParameterization->Plus(parameters[1], dp_1.data(), parameters[1]);
    cost_func_mag->Evaluate(parameters, residuals_m.data(), NULL);
    poseParameterBlock_1.setEstimate(T_WS_1_disturbed);  // reset
    J1_numDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
  }

  OKVIS_ASSERT_TRUE(Exception,
                    (J1min - J1_numDiff).norm() < jacobianTolerance,
                    "minimal Jacobian 2 = \n"
                        << J1min << std::endl
                        << "numDiff minimal Jacobian 2 = \n"
                        << J1_numDiff);
  poseLocalParameterization->ComputeJacobian(parameters[1], Jplus.data());

  // Run the solver!
  std::cout << "run the solver... " << std::endl;
  ::ceres::Solver::Options options;
  // options.check_gradients=true;
  // options.numeric_derivative_relative_step_size = 1e-6;
  // options.gradient_check_relative_precision=1e-2;
  options.minimizer_progress_to_stdout = false;
  ::FLAGS_stderrthreshold = google::WARNING;  // enable console warnings (Jacobian verification)
  ::ceres::Solver::Summary summary;
  ::ceres::Solve(options, &problem, &summary);

  // print some infos about the optimization
  std::cout << summary.BriefReport() << "\n";
  std::cout << "initial T_WS_1 : " << T_WS_1_disturbed.T() << "\n"
            << "optimized T_WS_1 : " << poseParameterBlock_1.estimate().T() << "\n"
            << "correct T_WS_1 : " << T_WS_1.T() << "\n";

  double quaternion_imu_error = 2 * (T_WS_1.q() * poseParameterBlock_1.estimate().q().inverse()).vec().norm();
  if (quaternion_imu_error > 1.0e-2) {
    std::cout << "IMU Preintergration could not solve for Quaternion" << std::endl;
    std::cout << "Final Quaternion Error using IMU preintegration: " << quaternion_imu_error << std::endl;

  }  // make sure it converged

  OKVIS_ASSERT_TRUE(Exception, summary.final_cost < 1e-2, "cost not reducible");

  ceres::Problem* imu_mag_problem = new ceres::Problem();
  poseParameterBlock_0.setEstimate(T_WS_0);
  poseParameterBlock_1.setEstimate(T_WS_1_disturbed);
  speedAndBiasParameterBlock_0.setEstimate(speedAndBias_0);
  speedAndBiasParameterBlock_1.setEstimate(speedAndBias_1);

  imu_mag_problem->AddParameterBlock(poseParameterBlock_0.parameters(), okvis::ceres::PoseParameterBlock::Dimension);
  imu_mag_problem->AddParameterBlock(poseParameterBlock_1.parameters(), okvis::ceres::PoseParameterBlock::Dimension);
  imu_mag_problem->SetParameterization(poseParameterBlock_0.parameters(), poseLocalParameterization);
  imu_mag_problem->SetParameterization(poseParameterBlock_1.parameters(), poseLocalParameterization);

  imu_mag_problem->AddParameterBlock(speedAndBiasParameterBlock_0.parameters(),
                                     okvis::ceres::SpeedAndBiasParameterBlock::Dimension);
  imu_mag_problem->AddParameterBlock(speedAndBiasParameterBlock_1.parameters(),
                                     okvis::ceres::SpeedAndBiasParameterBlock::Dimension);

  imu_mag_problem->AddResidualBlock(
      cost_func_mag, NULL, poseParameterBlock_0.parameters(), poseParameterBlock_1.parameters());
  imu_mag_problem->AddResidualBlock(cost_function_imu,
                                    NULL,
                                    poseParameterBlock_0.parameters(),
                                    speedAndBiasParameterBlock_0.parameters(),
                                    poseParameterBlock_1.parameters(),
                                    speedAndBiasParameterBlock_1.parameters());
  imu_mag_problem->AddResidualBlock(cost_function_pose, NULL, poseParameterBlock_0.parameters());
  imu_mag_problem->AddResidualBlock(cost_function_speedAndBias, NULL, speedAndBiasParameterBlock_0.parameters());

  ::ceres::Solve(options, imu_mag_problem, &summary);

  std::cout << ".... Now using IMU Preintegration with Magnetometer .... " << std::endl;
  // print some infos about the optimization
  std::cout << summary.BriefReport() << "\n";

  // OKVIS_ASSERT_TRUE(Exception, summary.final_cost < 1e-2, "cost not reducible");

  std::cout << "initial T_WS_1 : " << T_WS_1_disturbed.T() << "\n"
            << "optimized T_WS_1 : " << poseParameterBlock_1.estimate().T() << "\n"
            << "correct T_WS_1 : " << T_WS_1.T() << "\n";

  double quaternion_imu_mag_error = 2 * (T_WS_1.q() * poseParameterBlock_1.estimate().q().inverse()).vec().norm();
  std::cout << "Quaternion Error using IMU preintegration: " << quaternion_imu_error << std::endl;
  std::cout << "Quaternion IMU-Mag Error: " << quaternion_imu_mag_error << std::endl;

  OKVIS_ASSERT_TRUE(Exception,
                    2 * (T_WS_1.q() * poseParameterBlock_1.estimate().q().inverse()).vec().norm() < 1e-2,
                    "quaternions not close enough");

  OKVIS_ASSERT_TRUE(
      Exception, quaternion_imu_mag_error < quaternion_imu_error, "IMU&Mag Error is larger than IMU Error");
}
