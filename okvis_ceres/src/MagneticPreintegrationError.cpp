#include "okvis/ceres/MagneticPreintegrationError.hpp"

#include <glog/logging.h>

#include "okvis/assert_macros.hpp"
#include "okvis/ceres/ode/ode.hpp"

namespace okvis {
namespace ceres {

MagneticPreintegrationError::MagneticPreintegrationError(
    const okvis::MagnetometerMeasurementDeque& magnetometer_measurements,
    const okvis::MagnetometerParameters& magnetometer_params,
    const okvis::ImuMeasurementDeque& imu_measurements,
    const okvis::ImuParameters& imu_params,
    const okvis::Time& t_start,
    const okvis::Time& t_end) {
  // Functions inherited from ceres::CostFunction.
  // https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/cost_function.h
  // Set int num_residuals_.

  int number_of_residuals = 3 * magnetometer_measurements.size();
  set_num_residuals(number_of_residuals);

  mutable_parameter_block_sizes()->clear();
  mutable_parameter_block_sizes()->push_back(7);  // T_WS
  mutable_parameter_block_sizes()->push_back(9);  // speed and biases

  setImuMeasurements(imu_measurements);
  setMagnetometerMeasurements(magnetometer_measurements);
  setImuParameters(imu_params);
  setMagnetometerParameters(magnetometer_params);
  setStartTime(t_start);
  setEndTime(t_end);

  OKVIS_ASSERT_TRUE_DBG(Exception,
                        imu_measurements.back().timeStamp < magnetometer_measurements.back().timeStamp,
                        "Oldest IMU measurement is newer than oldest magnetometer measurement!");
  OKVIS_ASSERT_TRUE_DBG(Exception,
                        imu_measurements.front().timeStamp > magnetometer_measurements.front().timeStamp,
                        "Newest IMU measurement is older than newest magnetometer measurement!");
}

int MagneticPreintegrationError::propagation(const okvis::ImuMeasurementDeque& imu_measurements,
                                             const okvis::ImuParameters& imu_parameters,
                                             const okvis::MagnetometerMeasurementDeque& magnetometer_measurements,
                                             okvis::kinematics::Transformation& T_WS0,
                                             Transformations& T_WS,
                                             SpeedAndBias& speed_and_biases0,
                                             SpeedsAndBiases& speed_and_biases,
                                             const okvis::Time& t_start,
                                             const bool compute_covariance,
                                             const bool compute_jacobian,
                                             Covariances& covariances,
                                             Jacobians& jacobians) {
  okvis::Time time = t_start;
  assert(imu_measurements.front().timeStamp <= time);

  // Oldest imu meas older than oldest gp meas.
  if (!(imu_measurements.front().timeStamp < magnetometer_measurements.front().timeStamp)) {
    return -1;  // nothing to do...
  }

  // Newest imu meas newer than newest gp meas.
  if (!(imu_measurements.back().timeStamp > magnetometer_measurements.back().timeStamp)) {
    return -1;  // nothing to do...
  }

  // initial condition
  Eigen::Vector3d r_0 = T_WS0.r();
  Eigen::Quaterniond q_WS_0 = T_WS0.q();
  Eigen::Matrix3d C_WS_0 = T_WS0.C();

  // increments
  Eigen::Quaterniond delta_q = Eigen::Quaterniond(1, 0, 0, 0);

  // sub-Jacobians
  Eigen::Matrix3d dalpha_db_g = Eigen::Matrix3d::Zero();

  // the Jacobian of the increment (w/o biases)
  Eigen::Matrix<double, 6, 6> P_delta = Eigen::Matrix<double, 6, 6>::Zero();

  // Vectors for preintegrated values at rts residual time.
  std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> delta_qs;
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> dalpha_db_gs;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> P_deltas_magnetometer;

  uint32_t num_magnetometer_measurements = magnetometer_measurements.size();
  uint32_t current_mag_index = 0;

  bool has_started = false;
  bool last_iteration = false;
  uint32_t imu_measumentes_used = 0;

  for (okvis::ImuMeasurementDeque::const_iterator it = imu_measurements.begin(); it != imu_measurements.end(); ++it) {
    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // Meaning the magnetometer is between the current and next IMU measurement.
    okvis::Time next_time = (it + 1)->timeStamp;
    if (next_time < t_start) continue;

    // time delta
    double dt = (next_time - time).toSec();

    // Sanity check
    if (dt <= 0.0) {
      continue;
    }

    if (!has_started) {
      has_started = true;
      const double r = dt / (next_time - imu_measurements.front().timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    // ensure integrity
    double sigma_g_c = imu_parameters.sigma_g_c;
    double sigma_a_c = imu_parameters.sigma_a_c;

    if (std::abs(omega_S_0[0]) > imu_parameters.g_max || std::abs(omega_S_0[1]) > imu_parameters.g_max ||
        std::abs(omega_S_0[2]) > imu_parameters.g_max) {
      sigma_g_c *= 100;
      LOG(WARNING) << "gyr saturation";
    }

    if (std::abs(acc_S_0[0]) > imu_parameters.a_max || std::abs(acc_S_0[1]) > imu_parameters.a_max ||
        std::abs(acc_S_0[2]) > imu_parameters.a_max) {
      sigma_a_c *= 100;
      LOG(WARNING) << "acc saturation";
    }

    if (magnetometer_measurements[current_mag_index].timeStamp <= next_time) {
      current_mag_index++;
      if (current_mag_index == num_magnetometer_measurements) {
        last_iteration = true;
      }

      double interval = (next_time - it->timeStamp).toSec();
      double dt_until_magnetic = (magnetometer_measurements[current_mag_index].timeStamp - time).toSec();
      const double r = dt_until_magnetic / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();

      // actual propagation
      // orientation:
      Eigen::Quaterniond dq;
      const Eigen::Vector3d omega_S_true = (0.5 * (omega_S_0 + omega_S_1) - speed_and_biases0.segment<3>(3));
      const double theta_half = omega_S_true.norm() * 0.5 * dt_until_magnetic;
      const double sinc_theta_half = ode::sinc(theta_half);
      const double cos_theta_half = cos(theta_half);
      dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt_until_magnetic;
      dq.w() = cos_theta_half;
      Eigen::Quaterniond delta_q_1 = delta_q * dq;
      // rotation matrix integral:
      const Eigen::Matrix3d C_1 = delta_q_1.toRotationMatrix();

      // Jacobian parts
      const Eigen::Matrix3d dalpha_db_g_at_mag = dalpha_db_g + dt_until_magnetic * C_1;

      // covariance propagation
      if (compute_covariance) {
        Eigen::Matrix<double, 6, 6> F_delta = Eigen::Matrix<double, 6, 6>::Identity();
        F_delta.block<3, 3>(3, 3) = -dt_until_magnetic * C_1;

        Eigen::Matrix<double, 6, 6> P_delta_mag = Eigen::Matrix<double, 6, 6>::Identity();
        P_delta_mag = F_delta * P_delta * F_delta.transpose();
        // add noise. Note that transformations with rotation matrices can be
        // ignored, since the noise is isotropic.
        // F_tot = F_delta*F_tot;
        const double sigma2_dalpha = dt_until_magnetic * sigma_g_c * sigma_g_c;
        P_delta_mag(0, 0) += sigma2_dalpha;
        P_delta_mag(1, 1) += sigma2_dalpha;
        P_delta_mag(2, 2) += sigma2_dalpha;
        const double sigma2_b_g = dt_until_magnetic * imu_parameters.sigma_gw_c * imu_parameters.sigma_gw_c;
        P_delta_mag(3, 3) += sigma2_b_g;
        P_delta_mag(3, 4) += sigma2_b_g;
        P_delta_mag(4, 5) += sigma2_b_g;
        // store quantity
        P_deltas_magnetometer.push_back(P_delta_mag);
      }

      // store quantities
      delta_qs.push_back(delta_q_1);
      dalpha_db_gs.push_back(dalpha_db_g_at_mag);
    }
    if (last_iteration) {
      break;
    } else {
      // actual propagation
      // orientation:
      Eigen::Quaterniond dq;
      const Eigen::Vector3d omega_S_true = (0.5 * (omega_S_0 + omega_S_1) - speed_and_biases0.segment<3>(3));
      const double theta_half = omega_S_true.norm() * 0.5 * dt;
      const double sinc_theta_half = ode::sinc(theta_half);
      const double cos_theta_half = cos(theta_half);
      dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
      dq.w() = cos_theta_half;
      Eigen::Quaterniond delta_q_1 = delta_q * dq;
      // rotation matrix integral:
      const Eigen::Matrix3d C_1 = delta_q_1.toRotationMatrix();
      if (compute_covariance) {
        // covariance propagation
        Eigen::Matrix<double, 6, 6> F_delta = Eigen::Matrix<double, 6, 6>::Identity();
        F_delta.block<3, 3>(0, 0) = -dt * C_1;
        P_delta = F_delta * P_delta * F_delta.transpose();
        // add noise. Note that transformations with rotation matrices can be
        // ignored, since the noise is isotropic.
        // F_tot = F_delta*F_tot;
        const double sigma2_dalpha = dt * sigma_g_c * sigma_g_c;
        P_delta(0, 0) += sigma2_dalpha;
        P_delta(1, 1) += sigma2_dalpha;
        P_delta(2, 2) += sigma2_dalpha;
        const double sigma2_b_g = dt * imu_parameters.sigma_gw_c * imu_parameters.sigma_gw_c;
        P_delta(3, 3) += sigma2_b_g;
        P_delta(4, 4) += sigma2_b_g;
        P_delta(5, 5) += sigma2_b_g;
      }
      // memory shift
      delta_q = delta_q_1;
      time = next_time;

      imu_measumentes_used++;
    }
  }

  // actual propagation output:
  // Only rotation
  for (uint32_t i = 0; i < num_magnetometer_measurements; i++) {
    T_WS.push_back(okvis::kinematics::Transformation(r_0, q_WS_0 * delta_qs[i]));

    speed_and_biases.push_back(speed_and_biases0);
  }

  // assign Jacobian, if requested (Only at ;ast rts measurement at the moment)
  if (compute_jacobian) {
    for (uint32_t i = 0; i < num_magnetometer_measurements; ++i) {
      Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity();
      F.block<3, 3>(0, 0) = -C_WS_0 * dalpha_db_gs[i];

      jacobians.push_back(F);
    }
  }

  // overall covariance, if requested (Only at ;ast rts measurement at the moment)
  if (compute_covariance) {
    for (size_t i = 0; i < num_magnetometer_measurements; ++i) {
      Eigen::Matrix<double, 6, 6> P;
      // transform from local increments to actual states
      Eigen::Matrix<double, 6, 6> T = Eigen::Matrix<double, 6, 6>::Identity();
      T.topLeftCorner<3, 3>() = C_WS_0;
      P = T * P_deltas_magnetometer[i] * T.transpose();

      covariances.push_back(P);
    }
  }

  return imu_measumentes_used;
}

}  // namespace ceres
}  // namespace okvis