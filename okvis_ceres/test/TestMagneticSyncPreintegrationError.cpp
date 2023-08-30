
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
#include "okvis/ceres/MagneticSyncPreintegrationError.hpp"
#include "okvis/ceres/PoseError.hpp"
#include "okvis/ceres/PoseLocalParameterization.hpp"
#include "okvis/ceres/PoseParameterBlock.hpp"
#include "okvis/ceres/SpeedAndBiasError.hpp"
#include "okvis/ceres/SpeedAndBiasParameterBlock.hpp"
#include "okvis/kinematics/Transformation.hpp"

const double jacobianTolerance = 1.0e-3;

TEST(okvisTestSuite, MagneticSyncPreintegrationError) {
  // initialize random number generator
  // srand((unsigned int) time(0)); // disabled: make unit tests deterministic...

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
  imuParameters.sigma_g_c = 6.0e-4;
  imuParameters.sigma_a_c = 2.0e-3;
  imuParameters.sigma_gw_c = 3.0e-6;
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
  Eigen::Vector3d mag;

  okvis::MagnetometerParameters magnetometer_parameters;
  magnetometer_parameters.rate = 200;
  magnetometer_parameters.sigma_m_c = 0.01;

  okvis::MagnetometerMeasurement mag_measurement_end, mag_measurement_start;
  okvis::MagnetometerMeasurementDeque magnetic_measurements;

  for (size_t i = 0; i < size_t(duration * imuParameters.rate); ++i) {
    double time = double(i) / imuParameters.rate;
    if (i == 10) {  // set this as starting pose
      T_WS_0 = T_WS;
      speedAndBias_0 = speedAndBias;
      t_0 = okvis::Time(time);
      mag_measurement_start = okvis::MagnetometerMeasurement(okvis::Time(time), okvis::MagnetometerReading(mag));
    }
    if (i == size_t(duration * imuParameters.rate) - 10) {  // set this as starting pose
      T_WS_1 = T_WS;
      speedAndBias_1 = speedAndBias;
      t_1 = okvis::Time(time);
      mag_measurement_end = okvis::MagnetometerMeasurement(okvis::Time(time), okvis::MagnetometerReading(mag));
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
    if (i > 10 && (i < size_t(duration * imuParameters.rate) - 10)) {
      magnetic_measurements.push_back(
          okvis::MagnetometerMeasurement(okvis::Time(time), okvis::MagnetometerReading(mag)));
    }
  }

  okvis::Quaternions propagated_quaternions;
  //   okvis::SpeedAndBiases estimated_speed_and_biases;
  okvis::ceres::MagneticSyncPreintegrationError::Covariances propagaged_covariances;
  okvis::ceres::MagneticSyncPreintegrationError::Jacobians propagated_jacobians;

  int imu_measurements_used = okvis::ceres::MagneticSyncPreintegrationError::propagation(imuMeasurements,
                                                                                         imuParameters,
                                                                                         magnetic_measurements,
                                                                                         T_WS_0,
                                                                                         speedAndBias_0,
                                                                                         propagated_quaternions,
                                                                                         t_0,
                                                                                         &propagaged_covariances,
                                                                                         &propagated_jacobians);

  Eigen::Quaterniond quat = propagated_quaternions.back();
  okvis::kinematics::Transformation T_WS_0_copy(T_WS_0.r(), T_WS_0.q());
  okvis::SpeedAndBias speed_bias_copy = speedAndBias_0;

  okvis::ceres::ImuError::covariance_t imu_covariance;
  okvis::ceres::ImuError::jacobian_t imu_jacobian;
  okvis::ceres::ImuError::propagation(imuMeasurements,
                                      imuParameters,
                                      T_WS_0_copy,
                                      speed_bias_copy,
                                      t_0,
                                      magnetic_measurements.back().timeStamp,
                                      &imu_covariance,
                                      &imu_jacobian);

  Eigen::Quaterniond imu_propagated_quaternion = T_WS_0_copy.q();

  // OKVIS_ASSERT_TRUE(Exception,
  //                   imu_propagated_quaternion.isApprox(quat, 1e-4),
  //                   "IMU propagated and Magnetic Propagated Quaternion do not match");

  // create the pose parameter blocks

  okvis::kinematics::Transformation T_disturb;
  T_disturb.setRandom(0, 0.1);
  okvis::kinematics::Transformation T_WS_1_disturbed = T_WS_1 * T_disturb;
  okvis::ceres::PoseParameterBlock poseParameterBlock_0(T_WS_0, 0, t_0);            // ground truth
  okvis::ceres::PoseParameterBlock poseParameterBlock_1(T_WS_1_disturbed, 2, t_1);  // disturbed...
  problem.AddParameterBlock(poseParameterBlock_0.parameters(), okvis::ceres::PoseParameterBlock::Dimension);
  problem.AddParameterBlock(poseParameterBlock_1.parameters(), okvis::ceres::PoseParameterBlock::Dimension);

  // create the speed and bias
  okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_0(speedAndBias_0, 1, t_0);

  okvis::ceres::MagneticSyncPreintegrationError* magnetic_error = new okvis::ceres::MagneticSyncPreintegrationError(
      magnetic_measurements, magnetometer_parameters, imuMeasurements, imuParameters, t_0, mag_measurement_end);

  double* parameters[3];
  parameters[0] = poseParameterBlock_0.parameters();
  parameters[1] = speedAndBiasParameterBlock_0.parameters();
  parameters[2] = poseParameterBlock_1.parameters();

  int n_residuals_terms = magnetic_error->num_residuals();
  Eigen::Matrix<double, Eigen::Dynamic, 1> residuals = Eigen::MatrixXd::Zero(n_residuals_terms, 1);

  Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> J0_min = Eigen::MatrixXd::Zero(n_residuals_terms, 6);
  Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor> J1_min = Eigen::MatrixXd::Zero(n_residuals_terms, 9);
  Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> J2_min = Eigen::MatrixXd::Zero(n_residuals_terms, 6);

  Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor> J0 = Eigen::MatrixXd::Zero(n_residuals_terms, 7);
  Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor> J1 = Eigen::MatrixXd::Zero(n_residuals_terms, 9);
  Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor> J2 = Eigen::MatrixXd::Zero(n_residuals_terms, 7);

  double* jacobians[3];
  jacobians[0] = J0.data();
  jacobians[1] = J1.data();
  jacobians[2] = J2.data();

  double* jacobians_minimal[3];
  jacobians_minimal[0] = J0_min.data();
  jacobians_minimal[1] = J1_min.data();
  jacobians_minimal[2] = J2_min.data();

  // To fix the bias linealization point
  magnetic_error->EvaluateWithMinimalJacobians(parameters, residuals.data(), jacobians, jacobians_minimal);
  magnetic_error->EvaluateWithMinimalJacobians(parameters, residuals.data(), jacobians, jacobians_minimal);

  Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> J0_numDiff = Eigen::MatrixXd::Zero(n_residuals_terms, 6);
  ::ceres::LocalParameterization* poseLocalParameterization = new okvis::ceres::PoseLocalParameterization;

  // and now num-diff:
  double dx = 1e-6;

  for (size_t i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> dp_0;
    Eigen::Matrix<double, Eigen::Dynamic, 1> residuals_p = Eigen::MatrixXd::Zero(n_residuals_terms, 1);
    Eigen::Matrix<double, Eigen::Dynamic, 1> residuals_m = Eigen::MatrixXd::Zero(n_residuals_terms, 1);
    dp_0.setZero();
    dp_0[i] = dx;
    poseLocalParameterization->Plus(parameters[0], dp_0.data(), parameters[0]);
    magnetic_error->Evaluate(parameters, residuals_p.data(), NULL);
    poseParameterBlock_0.setEstimate(T_WS_0);  // reset
    dp_0[i] = -dx;
    poseLocalParameterization->Plus(parameters[0], dp_0.data(), parameters[0]);
    magnetic_error->Evaluate(parameters, residuals_m.data(), NULL);
    poseParameterBlock_0.setEstimate(T_WS_0);  // reset
    J0_numDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
  }

  OKVIS_ASSERT_TRUE(Exception,
                    (J0_min - J0_numDiff).norm() < jacobianTolerance,
                    "minimal Jacobian 2 = \n"
                        << J0_min << std::endl
                        << "numDiff minimal Jacobian 2 = \n"
                        << J0_numDiff);

  Eigen::Matrix<double, Eigen::Dynamic, 9> J1_numDiff = Eigen::MatrixXd::Zero(n_residuals_terms, 9);
  for (size_t i = 0; i < 9; ++i) {
    Eigen::Matrix<double, 9, 1> ds_0;
    Eigen::Matrix<double, Eigen::Dynamic, 1> residuals_p = Eigen::MatrixXd::Zero(n_residuals_terms, 1);
    Eigen::Matrix<double, Eigen::Dynamic, 1> residuals_m = Eigen::MatrixXd::Zero(n_residuals_terms, 1);
    ds_0.setZero();
    ds_0[i] = dx;
    Eigen::Matrix<double, 9, 1> plussed = speedAndBias_0 + ds_0;
    speedAndBiasParameterBlock_0.setEstimate(plussed);
    magnetic_error->Evaluate(parameters, residuals_p.data(), NULL);
    ds_0[i] = -dx;
    plussed = speedAndBias_0 + ds_0;
    speedAndBiasParameterBlock_0.setEstimate(plussed);
    magnetic_error->Evaluate(parameters, residuals_m.data(), NULL);
    speedAndBiasParameterBlock_0.setEstimate(speedAndBias_0);  // reset
    J1_numDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
  }

  OKVIS_ASSERT_TRUE(Exception,
                    (J1_min - J1_numDiff).norm() < jacobianTolerance,
                    "minimal Jacobian 2 = \n"
                        << J1_min << std::endl
                        << "numDiff minimal Jacobian 2 = \n"
                        << J1_numDiff);

  Eigen::Matrix<double, Eigen::Dynamic, 6> J2_numDiff = Eigen::MatrixXd::Zero(n_residuals_terms, 6);

  for (size_t i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> dp_1;
    Eigen::Matrix<double, Eigen::Dynamic, 1> residuals_p = Eigen::MatrixXd::Zero(n_residuals_terms, 1);
    Eigen::Matrix<double, Eigen::Dynamic, 1> residuals_m = Eigen::MatrixXd::Zero(n_residuals_terms, 1);
    dp_1.setZero();
    dp_1[i] = dx;
    poseLocalParameterization->Plus(parameters[2], dp_1.data(), parameters[2]);
    magnetic_error->Evaluate(parameters, residuals_p.data(), NULL);
    poseParameterBlock_1.setEstimate(T_WS_1_disturbed);  // reset
    dp_1[i] = -dx;
    poseLocalParameterization->Plus(parameters[2], dp_1.data(), parameters[2]);
    magnetic_error->Evaluate(parameters, residuals_m.data(), NULL);
    poseParameterBlock_1.setEstimate(T_WS_1_disturbed);  // reset
    J2_numDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
  }

  OKVIS_ASSERT_TRUE(Exception,
                    (J2_min - J2_numDiff).norm() < jacobianTolerance,
                    "minimal Jacobian 2 = \n"
                        << J2_min << std::endl
                        << "numDiff minimal Jacobian 2 = \n"
                        << J2_numDiff);

  // Run the solver!
  std::cout << "run the solver... " << std::endl;
  ::ceres::Solver::Options options;
  // options.check_gradients=true;
  // options.numeric_derivative_relative_step_size = 1e-6;
  // options.gradient_check_relative_precision=1e-2;
  options.minimizer_progress_to_stdout = false;
  ::FLAGS_stderrthreshold = google::WARNING;  // enable console warnings (Jacobian verification)
  ::ceres::Solver::Summary summary;
  ceres::Problem* mag_preintegration_problem = new ceres::Problem();
  poseParameterBlock_0.setEstimate(T_WS_0);
  poseParameterBlock_1.setEstimate(T_WS_1_disturbed);
  speedAndBiasParameterBlock_0.setEstimate(speedAndBias_0);

  ::ceres::CostFunction* cost_function_pose = new okvis::ceres::PoseError(T_WS_0, 1e-12, 1e-4);  // pose prior...
  problem.AddResidualBlock(cost_function_pose, NULL, poseParameterBlock_0.parameters());
  ::ceres::CostFunction* cost_function_speedAndBias =
      new okvis::ceres::SpeedAndBiasError(speedAndBias_0, 1e-12, 1e-12, 1e-12);  // speed and biases prior...
  problem.AddResidualBlock(cost_function_speedAndBias, NULL, speedAndBiasParameterBlock_0.parameters());

  okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_1(speedAndBias_1, 3, t_1);

  okvis::ceres::ImuError* cost_function_imu = new okvis::ceres::ImuError(imuMeasurements, imuParameters, t_0, t_1);

  mag_preintegration_problem->AddParameterBlock(poseParameterBlock_0.parameters(),
                                                okvis::ceres::PoseParameterBlock::Dimension);
  mag_preintegration_problem->AddParameterBlock(poseParameterBlock_1.parameters(),
                                                okvis::ceres::PoseParameterBlock::Dimension);
  mag_preintegration_problem->SetParameterization(poseParameterBlock_0.parameters(), poseLocalParameterization);
  mag_preintegration_problem->SetParameterization(poseParameterBlock_1.parameters(), poseLocalParameterization);

  mag_preintegration_problem->AddParameterBlock(speedAndBiasParameterBlock_0.parameters(),
                                                okvis::ceres::SpeedAndBiasParameterBlock::Dimension);

  // mag_preintegration_problem->AddResidualBlock(magnetic_error,
  //                                              NULL,
  //                                              poseParameterBlock_0.parameters(),
  //                                              speedAndBiasParameterBlock_0.parameters(),
  //                                              poseParameterBlock_1.parameters());

  mag_preintegration_problem->AddResidualBlock(cost_function_pose, NULL, poseParameterBlock_0.parameters());
  mag_preintegration_problem->AddResidualBlock(
      cost_function_speedAndBias, NULL, speedAndBiasParameterBlock_0.parameters());
  mag_preintegration_problem->AddResidualBlock(cost_function_imu,
                                               NULL,
                                               poseParameterBlock_0.parameters(),
                                               speedAndBiasParameterBlock_0.parameters(),
                                               poseParameterBlock_1.parameters(),
                                               speedAndBiasParameterBlock_1.parameters());

  ::ceres::Solve(options, mag_preintegration_problem, &summary);

  std::cout << ".... Now using IMU Preintegration with Magnetometer .... " << std::endl;
  // print some infos about the optimization
  std::cout << summary.BriefReport() << "\n";

  std::cout << "initial T_WS_1 : " << T_WS_1_disturbed.q().coeffs() << "\n"
            << "optimized T_WS_1 : " << poseParameterBlock_1.estimate().q().coeffs() << "\n"
            << "correct T_WS_1 : " << T_WS_1.q().coeffs() << "\n";

  double quaternion_error = 2 * (T_WS_1.q() * poseParameterBlock_1.estimate().q().inverse()).vec().norm();
  std::cout << "Quaternion error: " << quaternion_error << "\n";
}