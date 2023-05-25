#include <okvis/ceres/MagneticError.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

MagErrorYaw::MagErrorYaw(const Eigen::Vector3d& measurement,
                         double scale,
                         const Eigen::Vector3d& direction,
                         const Eigen::Vector3d& bias,
                         const information_t& information)
    : nM_(scale * direction), bias_(bias) {
  setMeasurement(measurement);
  setInformation(information);
}

MagErrorYaw::MagErrorYaw(const Eigen::Vector3d& measurement,
                         double scale,
                         const Eigen::Vector3d& direction,
                         const Eigen::Vector3d& bias,
                         double variance)
    : nM_(scale * direction), bias_(bias) {
  setMeasurement(measurement);
  setInformation(Eigen::Matrix3d::Identity() * 1.0 / variance);
}

// Set the information.
void MagErrorYaw::setInformation(const information_t& information) {
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<information_t> lltOfInformation(information_);
  sqrt_information_ = lltOfInformation.matrixL().transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
bool MagErrorYaw::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool MagErrorYaw::EvaluateWithMinimalJacobians(double const* const* parameters,
                                               double* residuals,
                                               double** jacobians,
                                               double** jacobiansMinimal) const {
  Eigen::Matrix<double, 6, 1> delta_;
  delta_.setZero();
  delta_[5] = parameters[0][0];

  // compute error
  okvis::kinematics::Transformation T_WS;
  T_WS.oplus(delta_);
  auto T_SW = T_WS.inverse();

  // get the error
  Eigen::Matrix<double, 3, 1> error;
  const Eigen::Vector3d mag_field = T_SW * nM_;
  error = mag_field + bias_ - measurement_;

  // weigh it
  Eigen::Map<Eigen::Matrix<double, 3, 1>> weighted_error(residuals);
  weighted_error = sqrt_information_ * error;

  // compute Jacobian...
  if (jacobians != NULL) {
    if (jacobians[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 3, 1>> J0(jacobians[0]);
      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> J0_minimal = okvis::kinematics::crossMx(mag_field);
      J0 = J0_minimal.col(2);
      J0 = (sqrt_information_ * J0);

      if (jacobiansMinimal != NULL) {
        if (jacobiansMinimal[0] != NULL) {
          Eigen::Map<Eigen::Matrix<double, 3, 1>> J0_minimal_mapped(jacobiansMinimal[0]);
          J0_minimal_mapped = J0_minimal.col(2);
        }
      }
    }
  }

  return true;
}

MagErrorYawLocal::MagErrorYawLocal(const Eigen::Vector3d& measurement,
                                   double scale,
                                   const Eigen::Vector3d& direction,
                                   const Eigen::Vector3d& bias,
                                   const information_t& information)
    : nM_(scale * direction), bias_(bias) {
  setMeasurement(measurement);
  setInformation(information);
}

MagErrorYawLocal::MagErrorYawLocal(const Eigen::Vector3d& measurement,
                                   double scale,
                                   const Eigen::Vector3d& direction,
                                   const Eigen::Vector3d& bias,
                                   double variance)
    : nM_(scale * direction), bias_(bias) {
  setMeasurement(measurement);
  setInformation(Eigen::MatrixXd::Identity(kNumResiduals, kNumResiduals) * 1.0 / variance);
}

// Set the information.
void MagErrorYawLocal::setInformation(const information_t& information) {
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<information_t> lltOfInformation(information_);
  sqrt_information_ = lltOfInformation.matrixL().transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
bool MagErrorYawLocal::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool MagErrorYawLocal::EvaluateWithMinimalJacobians(double const* const* parameters,
                                                    double* residuals,
                                                    double** jacobians,
                                                    double** jacobiansMinimal) const {
  okvis::kinematics::Transformation T_WS(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]));
  auto T_SW = T_WS.inverse();

  // get the error
  double error;
  Eigen::Vector3d mag_field = T_SW * nM_ + bias_;
  Eigen::Vector3d normalized_mag_field = mag_field.normalized();
  Eigen::Vector3d normalized_measurement = measurement_.normalized();
  error = normalized_measurement.dot(normalized_mag_field) - 1.0;

  // std::cout << "mag_field: " << mag_field.transpose() << std::endl;

  // weigh it
  residuals[0] = sqrt_information_(0, 0) * error;

  // compute Jacobian...
  if (jacobians != NULL && jacobians[0] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> J0(jacobians[0]);

    Eigen::Matrix<double, 1, 3, Eigen::RowMajor> J_dot_product = normalized_measurement.transpose();
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> J_norm = normalizationJacobian<3>(mag_field);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> J0_minimal = okvis::kinematics::crossMx(mag_field);

    Eigen::Matrix<double, 1, 7, Eigen::RowMajor> J_lift;
    PoseLocalParameterizationYaw::liftJacobian(parameters[0], J_lift.data());

    Eigen::Matrix<double, 1, 3> J_effective = J_dot_product * J_norm * J0_minimal;
    J0 = sqrt_information_(0, 0) * J_effective.col(2) * J_lift;

    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 1, 1>> J0_minimal_mapped(jacobiansMinimal[0]);
        J0_minimal_mapped = sqrt_information_(0, 0) * J_effective.col(2);
      }
    }
  }

  return true;
}

MagPoseError::MagPoseError(const Eigen::Vector3d& measurement,
                           double scale,
                           const Eigen::Vector3d& direction,
                           const Eigen::Vector3d& bias,
                           const information_t& information)
    : nM_(scale * direction), bias_(bias) {
  setMeasurement(measurement);
  setInformation(information);
}

MagPoseError::MagPoseError(const Eigen::Vector3d& measurement,
                           double scale,
                           const Eigen::Vector3d& direction,
                           const Eigen::Vector3d& bias,
                           double variance)
    : nM_(scale * direction), bias_(bias) {
  setMeasurement(measurement);
  setInformation(Eigen::Matrix3d::Identity() * 1.0 / variance);
}

// Set the information.
void MagPoseError::setInformation(const information_t& information) {
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<information_t> lltOfInformation(information_);
  sqrt_information_ = lltOfInformation.matrixL().transpose();
}

bool MagPoseError::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

bool MagPoseError::EvaluateWithMinimalJacobians(double const* const* parameters,
                                                double* residuals,
                                                double** jacobians,
                                                double** jacobiansMinimal) const {
  okvis::kinematics::Transformation T_WS(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]));

  auto T_SW = T_WS.inverse();

  // get the error
  Eigen::Matrix<double, 3, 1> error;
  const Eigen::Vector3d mag_field = T_SW * nM_;
  error = mag_field + bias_ - measurement_;

  // weigh it
  Eigen::Map<Eigen::Matrix<double, 3, 1>> weighted_error(residuals);
  weighted_error = sqrt_information_ * error;

  if (jacobians != NULL && jacobians[0] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J0(jacobians[0]);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> J0_minimal = okvis::kinematics::crossMx(mag_field);

    J0_minimal = (sqrt_information_ * J0_minimal).eval();

    // pseudo inverse of the local parametrization Jacobian:
    Eigen::Matrix<double, 3, 7, Eigen::RowMajor> J_lift;
    PoseLocalParameterization3d::liftJacobian(parameters[0], J_lift.data());

    // hallucinate Jacobian w.r.t. state
    J0 = J0_minimal * J_lift;

    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 3, 6>> J0_minimal_mapped(jacobiansMinimal[0]);
        J0_minimal_mapped.setZero();
        J0_minimal_mapped.block<3, 3>(3, 3) = J0_minimal;
      }
    }
  }
  return true;
}

}  // namespace ceres
}  // namespace okvis