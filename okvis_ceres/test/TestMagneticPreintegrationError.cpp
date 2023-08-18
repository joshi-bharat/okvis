
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
#include "okvis/ceres/MagneticPreintegrationError.hpp"
#include "okvis/ceres/PoseError.hpp"
#include "okvis/ceres/PoseLocalParameterization.hpp"
#include "okvis/ceres/PoseParameterBlock.hpp"
#include "okvis/ceres/SpeedAndBiasError.hpp"
#include "okvis/ceres/SpeedAndBiasParameterBlock.hpp"
#include "okvis/kinematics/Transformation.hpp"

const double jacobianTolerance = 1.0e-3;

TEST(okvisTestSuite, MagneticPreintegrationError) {
  // initialize random number generator
  // srand((unsigned int) time(0)); // disabled: make unit tests deterministic...

  // Build the problem.
  ::ceres::Problem problem;

  srand(time(NULL));

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
  magnetometer_parameters.rate = 20;
  magnetometer_parameters.stdev = 3.0;

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
  okvis::Transformations propagated_transfromations;
  okvis::SpeedAndBiases estimated_speed_and_biases;
  okvis::ceres::MagneticPreintegrationError::Covariances propagaged_covariances;
  okvis::ceres::MagneticPreintegrationError::Jacobians propagated_jacobians;

  std::cout << "Initial: " << T_WS_0.q().coeffs().transpose() << std::endl;
  std::cout << "Magnetometer Measurements: " << magnetic_measurements.size() << std::endl;

  okvis::ceres::MagneticPreintegrationError::propagation(imuMeasurements,
                                                         imuParameters,
                                                         magnetic_measurements,
                                                         T_WS_0,
                                                         speedAndBias_0,
                                                         propagated_transfromations,
                                                         t_0,
                                                         t_1,
                                                         &propagaged_covariances,
                                                         &propagated_jacobians);

  OKVIS_ASSERT_EQ(Exception,
                  magnetic_measurements.size(),
                  propagated_transfromations.size(),
                  "Quaternion should be propagated for each measurement");

  for (size_t i = 0; i < magnetic_measurements.size(); ++i) {
    okvis::kinematics::Transformation transform = propagated_transfromations[i];
    okvis::kinematics::Transformation T_WS_0_copy(T_WS_0.r(), T_WS_0.q());
    okvis::SpeedAndBias speed_bias_copy = speedAndBias_0;

    okvis::ceres::ImuError::covariance_t imu_covariance;
    okvis::ceres::ImuError::jacobian_t imu_jacobian;
    okvis::ceres::ImuError::propagation(imuMeasurements,
                                        imuParameters,
                                        T_WS_0_copy,
                                        speed_bias_copy,
                                        t_0,
                                        magnetic_measurements[i].timeStamp,
                                        &imu_covariance,
                                        &imu_jacobian);

    OKVIS_ASSERT_TRUE(
        Exception, 2 * (T_WS_0_copy.q() * transform.q().inverse()).vec().norm() < 1e-3, "quaternions not close enough");
  }

  // create the pose parameter blocks
  okvis::kinematics::Transformation T_disturb;
  T_disturb.setRandom(1, 1.0);
  okvis::kinematics::Transformation T_WS_1_disturbed = T_WS_1 * T_disturb;
  okvis::ceres::PoseParameterBlock poseParameterBlock_0(T_WS_0, 0, t_0);            // ground truth
  okvis::ceres::PoseParameterBlock poseParameterBlock_1(T_WS_1_disturbed, 2, t_1);  // disturbed...
  // problem.AddParameterBlock(poseParameterBlock_0.parameters(), okvis::ceres::PoseParameterBlock::Dimension);
  // problem.AddParameterBlock(poseParameterBlock_1.parameters(), okvis::ceres::PoseParameterBlock::Dimension);

  // create the speed and bias
  okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_0(speedAndBias_0, 1, t_0);
  okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_1(speedAndBias_1, 3, t_1);

  okvis::ceres::MagneticPreintegrationError mag_error =
      okvis::ceres::MagneticPreintegrationError(magnetic_measurements,
                                                magnetometer_parameters,
                                                imuMeasurements,
                                                imuParameters,
                                                start_magnetic_field,
                                                end_magnetic_field);

  double* parameters[2];
  parameters[0] = poseParameterBlock_0.parameters();
  parameters[1] = speedAndBiasParameterBlock_0.parameters();
  // mag_error.Evaluate(parameters, nullptr, nullptr);

  Eigen::Matrix<double, 15, 6> J0_numDiff;
  ::ceres::LocalParameterization* poseLocalParameterization = new okvis::ceres::PoseLocalParameterization;

  // and now num-diff:
  double dx = 1e-6;

  for (size_t i = 0; i < 3; ++i) {
    Eigen::Matrix<double, 6, 1> dp_0;
    Eigen::Matrix<double, Eigen::Dynamic, 1> residuals_p;
    residuals_p.resize(3 * magnetic_measurements.size(), 1);
    Eigen::Matrix<double, Eigen::Dynamic, 1> residuals_m;
    residuals_m.resize(3 * magnetic_measurements.size(), 1);

    dp_0.setZero();
    dp_0[i + 3] = dx;
    poseLocalParameterization->Plus(parameters[0], dp_0.data(), parameters[0]);
    // std::cout<<poseParameterBlock_0.estimate().T()<<std::endl;
    mag_error.Evaluate(parameters, residuals_p.data(), NULL);

    // std::cout<<residuals_p.transpose()<<std::endl;
    poseParameterBlock_0.setEstimate(T_WS_0);  // reset
    dp_0[i + 3] = -dx;
    // std::cout<<residuals.transpose()<<std::endl;
    poseLocalParameterization->Plus(parameters[0], dp_0.data(), parameters[0]);
    // std::cout<<poseParameterBlock_0.estimate().T()<<std::endl;
    mag_error.Evaluate(parameters, residuals_m.data(), NULL);
    // std::cout<<residuals_m.transpose()<<std::endl;
    poseParameterBlock_0.setEstimate(T_WS_0);  // reset
    Eigen::Matrix<double, Eigen::Dynamic, 1> jacobian = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    std::cout << "Calculated Jacobian: " << jacobian.transpose() << std::endl;
  }
}
