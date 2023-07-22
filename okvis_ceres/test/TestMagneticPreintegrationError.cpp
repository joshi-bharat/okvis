
#include <gtest/gtest.h>

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

  Eigen::Vector3d magnetic_field(-3024, 22795, -34437);
  okvis::MagnetometerParameters magnetometer_parameters;
  magnetometer_parameters.rate = 20;
  magnetometer_parameters.stdev = 3.0;

  okvis::MagnetometerMeasurementDeque magnetic_measurements;

  for (size_t i = 0; i < size_t(duration * imuParameters.rate); ++i) {
    double time = double(i) / imuParameters.rate;
    if (i == 10) {  // set this as starting pose
      T_WS_0 = T_WS;
      speedAndBias_0 = speedAndBias;
      t_0 = okvis::Time(time);
    }
    if (i == size_t(duration * imuParameters.rate) - 10) {  // set this as starting pose
      T_WS_1 = T_WS;
      speedAndBias_1 = speedAndBias;
      t_1 = okvis::Time(time);
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

    if (i > 0 && i % magnetometer_parameters.rate == 0) {
      Eigen::Vector3d mag =
          T_WS.inverse().C() * magnetic_field + magnetometer_parameters.stdev / sqrt(dt) * Eigen::Vector3d::Random();

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

  int imu_used_mag = okvis::ceres::MagneticPreintegrationError::propagation(imuMeasurements,
                                                                            imuParameters,
                                                                            magnetic_measurements,
                                                                            T_WS_0,
                                                                            propagated_transfromations,
                                                                            speedAndBias_0,
                                                                            estimated_speed_and_biases,
                                                                            t_0,
                                                                            &propagaged_covariances,
                                                                            &propagated_jacobians);

  OKVIS_ASSERT_EQ(Exception,
                  magnetic_measurements.size(),
                  propagated_transfromations.size(),
                  "Quaternion should be propagated for each measurement");

  for (int i = 0; i < magnetic_measurements.size(); ++i) {
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
}
