
#ifndef INCLUDE_OKVIS_CERES_MAGNETICPOSEERROR_HPP_
#define INCLUDE_OKVIS_CERES_MAGNETICPOSEERROR_HPP_

#include <ceres/ceres.h>

#include <vector>

#include "okvis/assert_macros.hpp"
#include "okvis/ceres/ErrorInterface.hpp"
#include "okvis/kinematics/Transformation.hpp"

namespace okvis {
namespace ceres {

class MagneticPoseError : public ::ceres::SizedCostFunction<3, 7, 7>, public ErrorInterface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  typedef ::ceres::SizedCostFunction<3, 7, 7> base_t;

  static const int kNumResiduals = 3;
  typedef Eigen::Matrix3d information_t;
  typedef Eigen::Matrix3d covariance_t;

  /// \brief Default constructor.
  MagneticPoseError(){};

  MagneticPoseError(const Eigen::Vector3d& start_magnetic_field,
                    const Eigen::Vector3d& end_magnetic_field,
                    const information_t& information);

  MagneticPoseError(const Eigen::Vector3d& start_magnetic_field,
                    const Eigen::Vector3d& end_magnetic_field,
                    double variance);

  /// \brief Trivial destructor.
  virtual ~MagneticPoseError(){};  // trivial destructor

  void setStartMagneticField(const Eigen::Vector3d& magnetic_field) { measurement0_ = magnetic_field; }

  void setEndMagneticField(const Eigen::Vector3d& magnetic_field) { measurement1_ = magnetic_field; }

  void setInformation(const information_t& information);

  // getters
  /// \brief Get the measurement.
  /// \return The measurement vector.
  const std::pair<Eigen::Vector3d, Eigen::Vector3d> measurement() const { return {measurement0_, measurement1_}; }

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
  Eigen::Vector3d measurement0_;  // The magnetometer measurement at T_WS_0
  Eigen::Vector3d measurement1_;

  // weights
  information_t information_;
  information_t sqrt_information_;
  covariance_t covariance_;
};

}  // namespace ceres
}  // namespace okvis

#endif