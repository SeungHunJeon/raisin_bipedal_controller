//
// Created by donghoon on 8/23/22.
//
#pragma once

#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "raisim/World.hpp"
#include "helper/BasicEigenTypes.hpp"
#include "raisin_bipedal_controller/raiboController.hpp"
#include "raisin_parameter/parameter_container.hpp"
#include "raisin_controller/controller.hpp"
#include "raisin_interfaces/srv/vector3.hpp"
#include "raisin_data_logger/raisin_data_logger.hpp"
#include "helper/neuralNet.hpp"

/// TODO
#define OBSDIM 42
#define ACTDIM 12
#define COMDIM 4
#define ENCOUTDIM 128
#define ENCNUMLAYER 1

namespace raisin {

namespace controller {

class raiboLearningController : public Controller {

 public:
  raiboLearningController(
    raisim::World & world, raisim::RaisimServer & server,
    raisim::World & worldSim, raisim::RaisimServer & serverSim, GlobalResource & globalResource);
  bool create() final;
  bool init() final;
  Eigen::VectorXf obsScalingAndGetAction();
  bool advance() final;
  bool warmUp();
  bool reset() final;
  bool terminate() final;
  bool stop() final;

 private:
  void setCommand(
      const std::shared_ptr<raisin_interfaces::srv::Vector3::Request> request,
      std::shared_ptr<raisin_interfaces::srv::Vector3::Response> response,
      const int locomotion_type
  );
  void commandCallback(const raisin_interfaces::msg::Command::SharedPtr msg);
  void joySigCallback(const std_msgs::msg::Int16::SharedPtr msg);

  raisim::RaiboController raiboController_;
  Eigen::VectorXf obs_;
  Eigen::Vector3f command_;
  Eigen::Vector4f command_4_;
  int locomotion_type_ = 1; // default as quadruped
  Eigen::Matrix<float, ENCOUTDIM, 1> latent_;

  Eigen::VectorXf obsMean_;
  Eigen::VectorXf obsVariance_;
  raisim::nn::GRU<float, OBSDIM, ENCOUTDIM> encoder_;
  raisim::nn::Linear<float, ENCOUTDIM + OBSDIM + COMDIM, ACTDIM, raisim::nn::ActivationType::leaky_relu> actor_;

  double control_dt_;
  double communication_dt_;

  parameter::ParameterContainer & param_;

  std::chrono::time_point<std::chrono::high_resolution_clock> controlBegin_;
  std::chrono::time_point<std::chrono::high_resolution_clock> controlEnd_;
  std::chrono::time_point<std::chrono::high_resolution_clock> joySubscribeBegin_;
  std::chrono::time_point<std::chrono::high_resolution_clock> joySubscribeEnd_;
  double elapsedTime_ = 0.;
  double joySubscribeTime_;
};

}

}


