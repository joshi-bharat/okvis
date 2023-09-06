#include "okvis/ceres/MagneticPoseError.hpp"

#include "okvis/ceres/PoseLocalParameterization.hpp"

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

MagneticPoseError::MagneticPoseError(const Eigen::Vector3d& start_magnetic_field,
                                     const Eigen::Vector3d& end_magnetic_field,
                                     const information_t& information) {
  setStartMagneticField(start_magnetic_field);
  setEndMagneticField(end_magnetic_field);
  setInformation(information);
}

MagneticPoseError::MagneticPoseError(const Eigen::Vector3d& start_magnetic_field,
                                     const Eigen::Vector3d& end_magnetic_field,
                                     double variance) {
  setStartMagneticField(start_magnetic_field);
  setEndMagneticField(end_magnetic_field);
  setInformation(Eigen::Matrix3d::Identity() * 1.0 / variance);
}

// Set the information.
void MagneticPoseError::setInformation(const information_t& information) {
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<information_t> lltOfInformation(information_);
  sqrt_information_ = lltOfInformation.matrixL().transpose();
}

bool MagneticPoseError::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

bool MagneticPoseError::EvaluateWithMinimalJacobians(double const* const* parameters,
                                                     double* residuals,
                                                     double** jacobians,
                                                     double** jacobiansMinimals) const {
  okvis::kinematics::Transformation T_WS0(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]));

  okvis::kinematics::Transformation T_WS1(
      Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]),
      Eigen::Quaterniond(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]));

  Eigen::Quaterniond dq = T_WS1.q().inverse() * T_WS0.q();

  Eigen::Vector3d w_mag1_estimated = dq * measurement0_;
  // get the error
  Eigen::Matrix<double, 3, 1> error = w_mag1_estimated - measurement1_;

  // if (error.norm() > 1.0) {
  // LOG(WARNING) << "Magnetic Error: " << error.transpose() << "\n";
  //   LOG(WARNING) << "Magnetic Measurement 0: " << measurement0_.transpose() << "\n";
  //   LOG(WARNING) << "Magnetic Measurement 1: " << measurement1_.transpose() << "\n";
  //   LOG(WARNING) << "Magnetic Estimated 1: " << w_mag1_estimated.transpose() << "\n";
  // }

  // weigh it
  Eigen::Map<Eigen::Matrix<double, 3, 1>> weighted_error(residuals);
  weighted_error = sqrt_information_ * error;

  if (jacobians != NULL) {
    if (jacobians[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J0(jacobians[0]);
      // de/dq
      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> J0_minimal =
          sqrt_information_ * T_WS1.q().inverse() * -okvis::kinematics::crossMx(T_WS0.q() * measurement0_);

      // pseudo inverse of the local parametrization Jacobian:
      Eigen::Matrix<double, 3, 7, Eigen::RowMajor> J0_lift;
      PoseLocalParameterization3d::liftJacobian(parameters[0], J0_lift.data());

      // hallucinate Jacobian w.r.t. state
      J0 = J0_minimal * J0_lift;

      if (jacobiansMinimals != NULL) {
        if (jacobiansMinimals[1] != NULL) {
          Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> J0_minimal_mapped(jacobiansMinimals[0]);
          J0_minimal_mapped.setZero();
          J0_minimal_mapped.block<3, 3>(0, 3) = J0_minimal;
        }
      }
    }

    if (jacobians[1] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J1(jacobians[1]);
      // de/dq
      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> J1_minimal =
          sqrt_information_ * T_WS1.q().inverse() * okvis::kinematics::crossMx(T_WS0.q() * measurement0_);

      // pseudo inverse of the local parametrization Jacobian:
      Eigen::Matrix<double, 3, 7, Eigen::RowMajor> J1_lift;
      PoseLocalParameterization3d::liftJacobian(parameters[1], J1_lift.data());

      // hallucinate Jacobian w.r.t. state
      J1 = J1_minimal * J1_lift;

      if (jacobiansMinimals != NULL) {
        if (jacobiansMinimals[1] != NULL) {
          Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> J1_minimal_mapped(jacobiansMinimals[1]);
          J1_minimal_mapped.setZero();
          J1_minimal_mapped.block<3, 3>(0, 3) = J1_minimal;
        }
      }
    }
  }
  return true;
}

}  // namespace ceres
}  // namespace okvis