/**
 * @file PoseError.hpp
 * @brief Header file for the PoseError class.
 * @author Bharat Joshi
 */

#ifndef INCLUDE_OKVIS_CERES_MAGNETICERROR_HPP_
#define INCLUDE_OKVIS_CERES_MAGNETICERROR_HPP_

#include <ceres/ceres.h>

#include <vector>

#include "okvis/assert_macros.hpp"
#include "okvis/ceres/ErrorInterface.hpp"
#include "okvis/kinematics/Transformation.hpp"

namespace okvis {
namespace ceres {

template <int dim>
[[nodiscard]] Eigen::Matrix<double, dim, dim> normalizationJacobian(const Eigen::Matrix<double, dim, 1>& vec) {
  Eigen::Matrix<double, dim, dim> J;
  J = (Eigen::MatrixXd::Identity(dim, dim) - vec * vec.transpose() / vec.squaredNorm()) / vec.norm();
  return J;
}

class MagErrorYaw : public ::ceres::SizedCostFunction<3, 1>, public ErrorInterface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  typedef ::ceres::SizedCostFunction<3, 1> base_t;

  static const int kNumResiduals = 3;
  typedef Eigen::Matrix3d information_t;
  typedef Eigen::Matrix3d covariance_t;

  /// \brief Default constructor.
  MagErrorYaw(){};

  MagErrorYaw(const Eigen::Vector3d& measurement,
              double scale,
              const Eigen::Vector3d& direction,
              const Eigen::Vector3d& bias,
              const information_t& information);

  MagErrorYaw(const Eigen::Vector3d& measurement,
              double scale,
              const Eigen::Vector3d& direction,
              const Eigen::Vector3d& bias,
              double variance);

  /// \brief Trivial destructor.
  virtual ~MagErrorYaw(){};  // trivial destructor

  void setMeasurement(const Eigen::Vector3d& measurement) { measurement_ = measurement; }

  void setInformation(const information_t& information);

  // getters
  /// \brief Get the measurement.
  /// \return The measurement vector.
  const Eigen::Vector3d& measurement() const { return measurement_; }

  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  const information_t& information() const { return information_; }

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  const information_t& covariance() const { return covariance_; }

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(double const* const* parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const { return kNumResiduals; }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const { return base_t::parameter_block_sizes().size(); }

  /// \brief Dimension of an individual parameter block.
  size_t parameterBlockDim(size_t parameterBlockId) const {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const { return "MagneticErrorYaw"; }

 protected:
  Eigen::Vector3d measurement_;  // The measured magnetometer values
  const Eigen::Vector3d nM_;     // Local Magnetic Field
  const Eigen::Vector3d bias_;   // Bias

  // weights
  information_t information_;
  information_t sqrt_information_;
  covariance_t covariance_;
};

class MagErrorYawLocal : public ::ceres::SizedCostFunction<1, 7>, public ErrorInterface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  typedef ::ceres::SizedCostFunction<1, 7> base_t;

  static const int kNumResiduals = 1;
  typedef Eigen::Matrix<double, 1, 1> information_t;
  typedef Eigen::Matrix<double, 1, 1> covariance_t;

  /// \brief Default constructor.
  MagErrorYawLocal(){};

  MagErrorYawLocal(const Eigen::Vector3d& measurement,
                   double scale,
                   const Eigen::Vector3d& direction,
                   const Eigen::Vector3d& bias,
                   const information_t& information);

  MagErrorYawLocal(const Eigen::Vector3d& measurement,
                   double scale,
                   const Eigen::Vector3d& direction,
                   const Eigen::Vector3d& bias,
                   double variance);

  /// \brief Trivial destructor.
  virtual ~MagErrorYawLocal(){};  // trivial destructor

  void setMeasurement(const Eigen::Vector3d& measurement) { measurement_ = measurement; }

  void setInformation(const information_t& information);

  // getters
  /// \brief Get the measurement.
  /// \return The measurement vector.
  const Eigen::Vector3d& measurement() const { return measurement_; }

  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  const information_t& information() const { return information_; }

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  const information_t& covariance() const { return covariance_; }

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(double const* const* parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const { return kNumResiduals; }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const { return base_t::parameter_block_sizes().size(); }

  /// \brief Dimension of an individual parameter block.
  size_t parameterBlockDim(size_t parameterBlockId) const {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const { return "MagneticErrorYawLocal"; }

 protected:
  Eigen::Vector3d measurement_;  // The measured magnetometer values
  const Eigen::Vector3d nM_;     // Local Magnetic Field
  const Eigen::Vector3d bias_;   // Bias

  // weights
  information_t information_;
  information_t sqrt_information_;
  covariance_t covariance_;
};

class MagPoseError : public ::ceres::SizedCostFunction<3, 7>, public ErrorInterface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  typedef ::ceres::SizedCostFunction<3, 7> base_t;

  static const int kNumResiduals = 3;
  typedef Eigen::Matrix3d information_t;
  typedef Eigen::Matrix3d covariance_t;

  /// \brief Default constructor.
  MagPoseError(){};

  MagPoseError(const Eigen::Vector3d& measurement,
               double scale,
               const Eigen::Vector3d& direction,
               const Eigen::Vector3d& bias,
               const information_t& information);

  MagPoseError(const Eigen::Vector3d& measurement,
               double scale,
               const Eigen::Vector3d& direction,
               const Eigen::Vector3d& bias,
               double variance);

  /// \brief Trivial destructor.
  virtual ~MagPoseError(){};  // trivial destructor

  void setMeasurement(const Eigen::Vector3d& measurement) { measurement_ = measurement; }

  void setInformation(const information_t& information);

  // getters
  /// \brief Get the measurement.
  /// \return The measurement vector.
  const Eigen::Vector3d& measurement() const { return measurement_; }

  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  const information_t& information() const { return information_; }

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  const information_t& covariance() const { return covariance_; }

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(double const* const* parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const { return kNumResiduals; }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const { return base_t::parameter_block_sizes().size(); }

  /// \brief Dimension of an individual parameter block.
  size_t parameterBlockDim(size_t parameterBlockId) const {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const { return "MagneticPoseError"; }

 protected:
  Eigen::Vector3d measurement_;  // The measured magnetometer values
  const Eigen::Vector3d nM_;     // Local Magnetic Field
  const Eigen::Vector3d bias_;   // Bias

  // weights
  information_t information_;
  information_t sqrt_information_;
  covariance_t covariance_;
};

}  // namespace ceres
}  // namespace okvis

#endif