
#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace glomap {
struct C2CBATADirectionError {
  C2CBATADirectionError(const Eigen::Vector3d& translation_obs,
                        const Eigen::Quaterniond& inv_rotation1,
                        const Eigen::Quaterniond& inv_rotation2)
      : translation_obs_(translation_obs),
        inv_rotation1_(inv_rotation1),
        inv_rotation2_(inv_rotation2) {}

  template <typename T>
  bool operator()(const T* position1,
                  const T* position2,
                  const T* translation1,
                  const T* translation2,
                  const T* scale,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        translation_obs_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
                    inv_rotation2_.cast<T>() *
                        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(translation2) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1) +
                    inv_rotation1_.cast<T>() *
                        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(translation1));
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs,
                                     const Eigen::Quaterniond& inv_rotation1,
                                     const Eigen::Quaterniond& inv_rotation2) {
    return (new ceres::
                AutoDiffCostFunction<C2CBATADirectionError, 3, 3, 3, 3, 3, 1>(
                    new C2CBATADirectionError(
                        translation_obs, inv_rotation1, inv_rotation2)));
  }

  const Eigen::Vector3d translation_obs_;
  const Eigen::Quaterniond inv_rotation1_, inv_rotation2_;
};

struct SC2CBATADirectionError {
  SC2CBATADirectionError(const Eigen::Vector3d& translation_obs,
                         const Eigen::Quaterniond& inv_rotation1,
                         const Eigen::Quaterniond& inv_rotation2)
      : translation_obs_(translation_obs),
        inv_rotation1_(inv_rotation1),
        inv_rotation2_(inv_rotation2) {}

  template <typename T>
  bool operator()(const T* position1,
                  const T* position2,
                  const T* translation,
                  const T* scale,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        translation_obs_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
                    inv_rotation2_.cast<T>() *
                        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(translation) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1) +
                    inv_rotation1_.cast<T>() *
                        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(translation));
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs,
                                     const Eigen::Quaterniond& inv_rotation1,
                                     const Eigen::Quaterniond& inv_rotation2) {
    return (
        new ceres::AutoDiffCostFunction<SC2CBATADirectionError, 3, 3, 3, 3, 1>(
            new SC2CBATADirectionError(
                translation_obs, inv_rotation1, inv_rotation2)));
  }

  const Eigen::Vector3d translation_obs_;
  const Eigen::Quaterniond inv_rotation1_, inv_rotation2_;
};

struct C2PBATADirectionError {
  C2PBATADirectionError(const Eigen::Vector3d& translation_obs,
                        const Eigen::Quaterniond& inv_rotation)
      : translation_obs_(translation_obs), inv_rotation_(inv_rotation) {}

  template <typename T>
  bool operator()(const T* position1,
                  const T* position2,
                  const T* translation,
                  const T* scale,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        translation_obs_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1) +
                    inv_rotation_.cast<T>() *
                        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(translation));
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs,
                                     const Eigen::Quaterniond& inv_rotation) {
    return (
        new ceres::AutoDiffCostFunction<C2PBATADirectionError, 3, 3, 3, 3, 1>(
            new C2PBATADirectionError(translation_obs, inv_rotation)));
  }

  const Eigen::Vector3d translation_obs_;
  const Eigen::Quaterniond inv_rotation_;
};

template <typename T>
bool AngleError(const Eigen::Matrix<T, 3, 1>& translation,
                T* residuals,
                const Eigen::Vector3d& translation_obs) {
  Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
  residuals_vec = translation_obs.cast<T>() - translation.normalized();
  return true;
}

// Internal camera-to-camera constraints
struct IC2CAngleError {
  IC2CAngleError(const Eigen::Vector3d& translation_obs,
                 const Eigen::Quaterniond& inv_rotation1,
                 const Eigen::Quaterniond& inv_rotation2)
      : translation_obs_(translation_obs),
        inv_rotation1_(inv_rotation1),
        inv_rotation2_(inv_rotation2) {};

  template <typename T>
  bool operator()(const T* translation1,
                  const T* translation2,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> translation =
        inv_rotation1_.cast<T>() *
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(translation1) -
        inv_rotation2_.cast<T>() *
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(translation2);
    return AngleError<T>(translation, residuals, translation_obs_);

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs,
                                     const Eigen::Quaterniond& inv_rotation1,
                                     const Eigen::Quaterniond& inv_rotation2) {
    return (new ceres::AutoDiffCostFunction<IC2CAngleError, 3, 3, 3>(
        new IC2CAngleError(translation_obs, inv_rotation1, inv_rotation2)));
  }

  const Eigen::Vector3d translation_obs_;
  const Eigen::Quaterniond inv_rotation1_, inv_rotation2_;
};

// camera-to-camera constraints for different cameras in different rigs
struct EMC2CAngleError {
  EMC2CAngleError(const Eigen::Vector3d& translation_obs,
                  const Eigen::Quaterniond& inv_rotation1,
                  const Eigen::Quaterniond& inv_rotation2)
      : translation_obs_(translation_obs),
        inv_rotation1_(inv_rotation1),
        inv_rotation2_(inv_rotation2) {};

  template <typename T>
  bool operator()(const T* position1,
                  const T* position2,
                  const T* translation1,
                  const T* translation2,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> translation =
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
        inv_rotation2_.cast<T>() *
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(translation2) -
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1) +
        inv_rotation1_.cast<T>() *
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(translation1);
    return AngleError<T>(translation, residuals, translation_obs_);

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs,
                                     const Eigen::Quaterniond& inv_rotation1,
                                     const Eigen::Quaterniond& inv_rotation2) {
    return (new ceres::AutoDiffCostFunction<EMC2CAngleError, 3, 3, 3, 3, 3>(
        new EMC2CAngleError(translation_obs, inv_rotation1, inv_rotation2)));
  }

  const Eigen::Vector3d translation_obs_;
  const Eigen::Quaterniond inv_rotation1_, inv_rotation2_;
};

// camera-to-camera constraints for the same camera in different rigs
struct ESC2CAngleError {
  ESC2CAngleError(const Eigen::Vector3d& translation_obs,
                  const Eigen::Quaterniond& inv_rotation1,
                  const Eigen::Quaterniond& inv_rotation2)
      : translation_obs_(translation_obs),
        inv_rotation1_(inv_rotation1),
        inv_rotation2_(inv_rotation2) {};

  template <typename T>
  bool operator()(const T* position1,
                  const T* position2,
                  const T* translation_rel,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> translation =
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
        inv_rotation2_.cast<T>() *
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(translation_rel) -
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1) +
        inv_rotation1_.cast<T>() *
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(translation_rel);
    return AngleError<T>(translation, residuals, translation_obs_);

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs,
                                     const Eigen::Quaterniond& inv_rotation1,
                                     const Eigen::Quaterniond& inv_rotation2) {
    return (new ceres::AutoDiffCostFunction<ESC2CAngleError, 3, 3, 3, 3>(
        new ESC2CAngleError(translation_obs, inv_rotation1, inv_rotation2)));
  }

  const Eigen::Vector3d translation_obs_;
  const Eigen::Quaterniond inv_rotation1_, inv_rotation2_;
};

struct TrajectoryRegularization {
  TrajectoryRegularization(const double& weight) : weight_(weight) {};

  template <typename T>
  bool operator()(const T* position1,
                  const T* position2,
                  const T* position3,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 4, 1>> residuals_vec(residuals);

    const Eigen::Matrix<T, 3, 1> trans1 =
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1);
    const Eigen::Matrix<T, 3, 1> trans2 =
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position3) -
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2);

    const T eps = T(1e-8);

    T norm1 = trans1.squaredNorm();
    T norm2 = trans2.squaredNorm();

    residuals_vec.template head<3>() =
        trans1 / ceres::sqrt(norm1 + eps) - trans2 / ceres::sqrt(norm2 + eps);

    residuals[3] =
        (norm1 + norm2) / (ceres::sqrt(norm1 * norm2) * T(2) + eps) - T(1);

    residuals_vec *= T(weight_);

    return true;
  }
  static ceres::CostFunction* Create(const double& weight) {
    return (
        new ceres::AutoDiffCostFunction<TrajectoryRegularization, 4, 3, 3, 3>(
            new TrajectoryRegularization(weight)));
  }

  const double weight_;
};

struct C2PMulti1DSfMError {
  C2PMulti1DSfMError(const Eigen::Vector3d& translation_obs,
                     const Eigen::Quaterniond& inv_rotation)
      : translation_obs_(translation_obs), inv_rotation_(inv_rotation) {};

  // The error is given by the position error described above.
  template <typename T>
  bool operator()(const T* position1,
                  const T* position2,
                  const T* translation_rel,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> translation =
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1) +
        inv_rotation_.cast<T>() *
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(translation_rel);
    return AngleError<T>(translation, residuals, translation_obs_);

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs,
                                     const Eigen::Quaterniond& inv_rotation) {
    return (new ceres::AutoDiffCostFunction<C2PMulti1DSfMError, 3, 3, 3, 3>(
        new C2PMulti1DSfMError(translation_obs, inv_rotation)));
  }

  const Eigen::Vector3d translation_obs_;
  const Eigen::Quaterniond inv_rotation_;
};
// ----------------------------------------
// BATAPairwiseDirectionError
// ----------------------------------------
// Computes the error between a translation direction and the direction formed
// from two positions such that t_ij - scale * (c_j - c_i) is minimized.
struct BATAPairwiseDirectionError {
  BATAPairwiseDirectionError(const Eigen::Vector3d& translation_obs)
      : translation_obs_(translation_obs) {}

  // The error is given by the position error described above.
  template <typename T>
  bool operator()(const T* position1,
                  const T* position2,
                  const T* scale,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        translation_obs_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1));
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs) {
    return (
        new ceres::AutoDiffCostFunction<BATAPairwiseDirectionError, 3, 3, 3, 1>(
            new BATAPairwiseDirectionError(translation_obs)));
  }

  // TODO: add covariance
  const Eigen::Vector3d translation_obs_;
};

// ----------------------------------------
// FetzerFocalLengthCost
// ----------------------------------------
// Below are assets for DMAP by Philipp Lindenberger
inline Eigen::Vector4d fetzer_d(const Eigen::Vector3d& ai,
                                const Eigen::Vector3d& bi,
                                const Eigen::Vector3d& aj,
                                const Eigen::Vector3d& bj,
                                const int u,
                                const int v) {
  Eigen::Vector4d d;
  d.setZero();
  d(0) = ai(u) * aj(v) - ai(v) * aj(u);
  d(1) = ai(u) * bj(v) - ai(v) * bj(u);
  d(2) = bi(u) * aj(v) - bi(v) * aj(u);
  d(3) = bi(u) * bj(v) - bi(v) * bj(u);
  return d;
}

inline std::array<Eigen::Vector4d, 3> fetzer_ds(
    const Eigen::Matrix3d& i1_G_i0) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      i1_G_i0, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d s = svd.singularValues();

  Eigen::Vector3d v_0 = svd.matrixV().col(0);
  Eigen::Vector3d v_1 = svd.matrixV().col(1);

  Eigen::Vector3d u_0 = svd.matrixU().col(0);
  Eigen::Vector3d u_1 = svd.matrixU().col(1);

  Eigen::Vector3d ai =
      Eigen::Vector3d(s(0) * s(0) * (v_0(0) * v_0(0) + v_0(1) * v_0(1)),
                      s(0) * s(1) * (v_0(0) * v_1(0) + v_0(1) * v_1(1)),
                      s(1) * s(1) * (v_1(0) * v_1(0) + v_1(1) * v_1(1)));

  Eigen::Vector3d aj = Eigen::Vector3d(u_1(0) * u_1(0) + u_1(1) * u_1(1),
                                       -(u_0(0) * u_1(0) + u_0(1) * u_1(1)),
                                       u_0(0) * u_0(0) + u_0(1) * u_0(1));

  Eigen::Vector3d bi = Eigen::Vector3d(s(0) * s(0) * v_0(2) * v_0(2),
                                       s(0) * s(1) * v_0(2) * v_1(2),
                                       s(1) * s(1) * v_1(2) * v_1(2));

  Eigen::Vector3d bj =
      Eigen::Vector3d(u_1(2) * u_1(2), -(u_0(2) * u_1(2)), u_0(2) * u_0(2));

  Eigen::Vector4d d_01 = fetzer_d(ai, bi, aj, bj, 1, 0);
  Eigen::Vector4d d_02 = fetzer_d(ai, bi, aj, bj, 0, 2);
  Eigen::Vector4d d_12 = fetzer_d(ai, bi, aj, bj, 2, 1);

  std::array<Eigen::Vector4d, 3> ds;
  ds[0] = d_01;
  ds[1] = d_02;
  ds[2] = d_12;

  return ds;
}

class FetzerFocalLengthCost {
 public:
  FetzerFocalLengthCost(const Eigen::Matrix3d& i1_F_i0,
                        const Eigen::Vector2d& principal_point0,
                        const Eigen::Vector2d& principal_point1) {
    Eigen::Matrix3d K0 = Eigen::Matrix3d::Identity(3, 3);
    K0(0, 2) = principal_point0(0);
    K0(1, 2) = principal_point0(1);

    Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity(3, 3);
    K1(0, 2) = principal_point1(0);
    K1(1, 2) = principal_point1(1);

    const Eigen::Matrix3d i1_G_i0 = K1.transpose() * i1_F_i0 * K0;

    const std::array<Eigen::Vector4d, 3> ds = fetzer_ds(i1_G_i0);

    d_01 = ds[0];
    d_02 = ds[1];
    d_12 = ds[2];
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d i1_F_i0,
                                     const Eigen::Vector2d& principal_point0,
                                     const Eigen::Vector2d& principal_point1) {
    return (new ceres::AutoDiffCostFunction<FetzerFocalLengthCost, 2, 1, 1>(
        new FetzerFocalLengthCost(
            i1_F_i0, principal_point0, principal_point1)));
  }

  template <typename T>
  bool operator()(const T* const fi_, const T* const fj_, T* residuals) const {
    const Eigen::Vector<T, 4> d_01_ = d_01.cast<T>();
    const Eigen::Vector<T, 4> d_12_ = d_12.cast<T>();

    const T fi = fi_[0];
    const T fj = fj_[0];

    T di = (fj * fj * d_01_(0) + d_01_(1));
    T dj = (fi * fi * d_12_(0) + d_12_(2));
    di = di == T(0) ? T(1e-6) : di;
    dj = dj == T(0) ? T(1e-6) : dj;

    const T K0_01 = -(fj * fj * d_01_(2) + d_01_(3)) / di;
    const T K1_12 = -(fi * fi * d_12_(1) + d_12_(3)) / dj;

    residuals[0] = (fi * fi - K0_01) / (fi * fi);
    residuals[1] = (fj * fj - K1_12) / (fj * fj);

    return true;
  }

 private:
  Eigen::Vector4d d_01;
  Eigen::Vector4d d_02;
  Eigen::Vector4d d_12;
};

// Calibration error for the image pairs sharing the camera
class FetzerFocalLengthSameCameraCost {
 public:
  FetzerFocalLengthSameCameraCost(const Eigen::Matrix3d& i1_F_i0,
                                  const Eigen::Vector2d& principal_point) {
    Eigen::Matrix3d K0 = Eigen::Matrix3d::Identity(3, 3);
    K0(0, 2) = principal_point(0);
    K0(1, 2) = principal_point(1);

    Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity(3, 3);
    K1(0, 2) = principal_point(0);
    K1(1, 2) = principal_point(1);

    const Eigen::Matrix3d i1_G_i0 = K1.transpose() * i1_F_i0 * K0;

    const std::array<Eigen::Vector4d, 3> ds = fetzer_ds(i1_G_i0);

    d_01 = ds[0];
    d_02 = ds[1];
    d_12 = ds[2];
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d i1_F_i0,
                                     const Eigen::Vector2d& principal_point) {
    return (
        new ceres::AutoDiffCostFunction<FetzerFocalLengthSameCameraCost, 2, 1>(
            new FetzerFocalLengthSameCameraCost(i1_F_i0, principal_point)));
  }

  template <typename T>
  bool operator()(const T* const fi_, T* residuals) const {
    const Eigen::Vector<T, 4> d_01_ = d_01.cast<T>();
    const Eigen::Vector<T, 4> d_12_ = d_12.cast<T>();

    const T fi = fi_[0];
    const T fj = fi_[0];

    T di = (fj * fj * d_01_(0) + d_01_(1));
    T dj = (fi * fi * d_12_(0) + d_12_(2));
    di = di == T(0) ? T(1e-6) : di;
    dj = dj == T(0) ? T(1e-6) : dj;

    const T K0_01 = -(fj * fj * d_01_(2) + d_01_(3)) / di;
    const T K1_12 = -(fi * fi * d_12_(1) + d_12_(3)) / dj;

    residuals[0] = (fi * fi - K0_01) / (fi * fi);
    residuals[1] = (fj * fj - K1_12) / (fj * fj);

    return true;
  }

 private:
  Eigen::Vector4d d_01;
  Eigen::Vector4d d_02;
  Eigen::Vector4d d_12;
};

// ----------------------------------------
// GravError
// ----------------------------------------
struct GravError {
  GravError(const Eigen::Vector3d& grav_obs) : grav_obs_(grav_obs) {}

  template <typename T>
  bool operator()(const T* const gvec, T* residuals) const {
    Eigen::Matrix<T, 3, 1> grav_est;
    grav_est << gvec[0], gvec[1], gvec[2];

    for (int i = 0; i < 3; i++) {
      residuals[i] = grav_est[i] - grav_obs_[i];
    }

    return true;
  }

  // Factory function
  static ceres::CostFunction* CreateCost(const Eigen::Vector3d& grav_obs) {
    return (new ceres::AutoDiffCostFunction<GravError, 3, 3>(
        new GravError(grav_obs)));
  }

 private:
  const Eigen::Vector3d& grav_obs_;
};

}  // namespace glomap