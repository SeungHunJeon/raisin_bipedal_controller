//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "raisin_bipedal_controller/helper/BasicEigenTypes.hpp"
#include <raisim/RaisimServer.hpp>
#include <algorithm>

#ifndef _RAISIM_GYM_RAIBO_CONTROLLER_HPP
#define _RAISIM_GYM_RAIBO_CONTROLLER_HPP

namespace raisim {

class RaiboController {
 public:
  inline bool create(raisim::ArticulatedSystem * robot, const double & pGain, const double & dGain,
                     const std::map<int, Eigen::VectorXd> & nominalJointConfigs,
                     const std::map<int, raisim::Mat<3,3>> & standRots) {
    raibo_ = robot;
    gc_.resize(raibo_->getGeneralizedCoordinateDim());
    gv_.resize(raibo_->getDOF());
    gc_init_.resize(raibo_->getGeneralizedCoordinateDim());
    gv_init_.resize(raibo_->getDOF());

    /// Observation
    nominalJointConfigs_ = nominalJointConfigs;
    standRots_ = standRots;
    jointTarget_.setZero(nJoints_);
    gc_init_ << 0, 0, 0.5225, 1, 0, 0, 0, nominalJointConfigs_[locomotion_type_];
    gv_init_.setZero();

    /// action
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    actionScaled_.setZero(actionDim_);
    previousAction_.setZero(actionDim_);

    actionMean_ << nominalJointConfigs_[locomotion_type_]; /// joint target
    actionStd_ << Eigen::VectorXd::Constant(nJoints_, 0.1); /// joint target

    obDouble_.setZero(obDim_);

    /// pd controller
    jointPgain_.setZero(gvDim_); jointPgain_.tail(nJoints_).setConstant(pGain);
    jointDgain_.setZero(gvDim_); jointDgain_.tail(nJoints_).setConstant(dGain);
    jointTorque_.setZero(nJoints_);
    raibo_->setPdGains(jointPgain_, jointDgain_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);

    clippedGenForce_.setZero(gvDim_);

    return true;
  };

  void reset() {
    clippedGenForce_.tail(nJoints_).setZero();
    command_.setZero();
    command_.tail(1) << 1.;
    raibo_->getState(gc_, gv_);
    jointTarget_ = gc_.tail(nJoints_);
    previousAction_ << gc_.tail(nJoints_);
  }

  void updateStateVariables() {
    raibo_->getState(gc_, gv_);

    raisim::Vec<4> quat;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, baseRot_);

    baseRot_.e() = baseRot_.e() * standRots_[locomotion_type_].e().transpose();
    bodyAngVel_ = baseRot_.e().transpose() * gv_.segment(3, 3);

    /// If the base rotation matrix tilted, the robot considers in transition mode.
    raisim::Vec<3> zAxis_{0,0,1};
    transitionAngle = std::acos(baseRot_.e().row(2).dot(zAxis_.e()));
    transitionRate = std::clamp(transitionAngle / M_PI_2, 0., 1.);
  }

  bool advance(const Eigen::Ref<EigenVec> &action) {
    /// action scaling
    jointTarget_ = action.cast<double>();
    jointTarget_ = jointTarget_.cwiseProduct(actionStd_);
    jointTarget_ += nominalJointConfigs_[locomotion_type_] * (1 - transitionRate) +
    nominalJointConfigs_[!locomotion_type_] * transitionRate;

    pTarget_.tail(nJoints_) = jointTarget_;
    raibo_->setPdTarget(pTarget_, vTarget_);

    previousAction_ = jointTarget_;

    jointTorque_ = raibo_->getGeneralizedForce().e().tail(nJoints_);

    return true;
  }

  void clipTorque() {
    jointPos_ = raibo_->getGeneralizedCoordinate().e().tail(nJoints_);
    jointVel_ = raibo_->getGeneralizedVelocity().e().tail(nJoints_);
    clippedGenForce_.tail(nJoints_) = jointPgain_.tail(1) * (jointTarget_ - jointPos_) - jointDgain_.tail(1) * jointVel_;

    for(int i = 0; i < nJoints_; i++) {
      /// torque limit clip
      if (std::abs(clippedGenForce_.tail(nJoints_)(i)) > torqueLimit_) {
        clippedTorque_ = torqueLimit_;
        clippedGenForce_.tail(nJoints_)(i) = std::copysign(clippedTorque_, clippedGenForce_.tail(nJoints_)(i));
      }
    }

    raibo_->setGeneralizedForce(clippedGenForce_);
  }

  void updateObservation() {
    updateStateVariables();

    /// body orientation
    obDouble_.head(3) = baseRot_.e().row(2).transpose();
    /// body ang vel
    obDouble_.segment(3, 3) = bodyAngVel_;
    /// joint pos
    obDouble_.segment(6, nJoints_) = gc_.tail(nJoints_);
    /// joint vel
    obDouble_.segment(18, nJoints_) = gv_.tail(nJoints_);
    /// previous action
    obDouble_.segment(30, nJoints_) = previousAction_;
    /// command
    obDouble_.tail(4) = command_;
  }

  Eigen::VectorXd getObservation() {
    return obDouble_;
  }

  void setCommand(const Eigen::Ref<EigenVec>& command) {
    command_ = command.cast<double>();
    locomotion_type_ = command_(3);
  }

  inline void setStandingMode(bool mode) { standingMode_ = mode; }

  void getInitState(Eigen::VectorXd &gc, Eigen::VectorXd &gv) {
    gc.resize(gcDim_);
    gv.resize(gvDim_);
    gc << gc_init_;
    gv << gv_init_;
  }

  Eigen::VectorXd getJointPGain() const { return jointPgain_; }
  Eigen::VectorXd getJointDGain() const { return jointDgain_; }
  Eigen::VectorXd getJointTorque() const { return jointTorque_;}
  Eigen::VectorXd getJointPTarget() const { return jointTarget_; }
  [[nodiscard]] static constexpr int getObDim() { return obDim_; }
  [[nodiscard]] static constexpr int getActionDim() { return actionDim_; }
  [[nodiscard]] double getSimDt() { return simDt_; }
  [[nodiscard]] double getConDt() { return conDt_; }
  void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) { gc = gc_.cast<float>(); gv = gv_.cast<float>(); }

  void setSimDt(double dt) { simDt_ = dt; };
  void setConDt(double dt) { conDt_ = dt; };

  // robot configuration variables
  raisim::ArticulatedSystem *raibo_;
  std::map<int, Eigen::VectorXd> nominalJointConfigs_;
  std::map<int, raisim::Mat<3,3>> standRots_;
  static constexpr int nJoints_ = 12;
  static constexpr int actionDim_ = 12;
  static constexpr size_t obDim_ = 46;
  double simDt_ = .0025;
  static constexpr int gcDim_ = 19;
  static constexpr int gvDim_ = 18;

  // robot state variables
  Eigen::VectorXd gc_, gv_, gc_init_, gv_init_;
  Eigen::Vector3d bodyAngVel_;
  raisim::Mat<3, 3> baseRot_;

  // robot observation variables
  Eigen::VectorXd obDouble_;

  // control variables
  double conDt_ = 0.01;
  bool standingMode_ = false;
  Eigen::VectorXd actionMean_, actionStd_, actionScaled_, previousAction_;
  Eigen::VectorXd pTarget_, vTarget_; // full robot gc dim
  Eigen::VectorXd jointTarget_, jointTorque_;
  Eigen::VectorXd jointPgain_, jointDgain_;
  Eigen::Vector4d command_;
  int locomotion_type_ = 1;

  double transitionAngle;
  double transitionRate;


  Eigen::VectorXd clippedGenForce_, frictionTorque_;
  Eigen::VectorXd jointPos_, jointVel_;
  double clippedTorque_, torqueLimit_ = 71.5;
};

}

#endif //_RAISIM_GYM_RAIBO_CONTROLLER_HPP
