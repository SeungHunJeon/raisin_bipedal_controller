//
// Created by donghoon on 8/23/22.
// 
 
#include <filesystem>
#include "ament_index_cpp/get_package_prefix.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "raisin_bipedal_controller/raibo_bipedal_controller.hpp"

namespace raisin {

namespace controller {

using std::placeholders::_1;
using std::placeholders::_2;

raiboLearningController::raiboLearningController(
    raisim::World & world, raisim::RaisimServer & server,
    raisim::World & worldSim, raisim::RaisimServer & serverSim, GlobalResource & globalResource)
: Controller("raisin_bipedal_controller", world, server, worldSim, serverSim, globalResource),
      encoder_(ENCNUMLAYER),
      actor_({256, 128}),
      param_(parameter::ParameterContainer::getRoot()["raiboLearningController"])
      {
  param_.loadFromPackageParameterFile("raisin_bipedal_controller");

  rclcpp::QoS qos(rclcpp::KeepLast(1));
}

bool raiboLearningController::create() {
  control_dt_ = 0.01;
  communication_dt_ = 0.00025;
  double pGain, dGain;
  pGain = param_("p_gain");
  dGain = param_("d_gain");

  raisim::Mat<3,3> rot{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::map<int, Eigen::VectorXd> nominalJointConfigs;
  std::map<int, raisim::Mat<3,3>> standRots_;

  nominalJointConfigs.emplace(0, Eigen::VectorXd::Zero(12));
  nominalJointConfigs.emplace(1, Eigen::VectorXd::Zero(12));

  standRots_.emplace(0, rot);
  standRots_.emplace(1, rot);

  nominalJointConfigs[0] << 0, 1.580099, -1.195, 0, 1.580099, -1.195, 0, 1.84205, -0.557131, 0, 1.84205, -0.557131;
  nominalJointConfigs[1] << 0, 0.580099, -1.195, 0, 0.580099, -1.195, 0, 0.580099, -1.195, 0, 0.580099, -1.195;
  
  raisim::angleAxisToRotMat({0,1,0}, -M_PI_2, standRots_[0]);
  raisim::angleAxisToRotMat({0,1,0}, 0, standRots_[1]);

  raiboController_.create(robotHub_, pGain, dGain, nominalJointConfigs, standRots_);

  std::string network_path = std::string(param_("network_path"));

  /// load policy network parameters
  std::string model_itertaion = std::string(param_("model_number"));
  std::string encoder_file_name = std::string("GRU_") + model_itertaion + std::string(".txt");
  std::string actor_file_name = std::string("MLP_") + model_itertaion + std::string(".txt");
  std::string obs_mean_file_name = std::string("mean") + model_itertaion + std::string(".csv");
  std::string obs_var_file_name = std::string("var") + model_itertaion + std::string(".csv");

  std::filesystem::path pack_path(ament_index_cpp::get_package_prefix("raisin_bipedal_controller"));
  std::filesystem::path encoder_path = pack_path / network_path / encoder_file_name;
  std::filesystem::path actor_path = pack_path / network_path / actor_file_name;
  std::filesystem::path obs_mean_path = pack_path / network_path / obs_mean_file_name;
  std::filesystem::path obs_var_path = pack_path / network_path / obs_var_file_name;

  encoder_.readParamFromTxt(encoder_path.string());
  actor_.readParamFromTxt(actor_path.string());

  std::string in_line;
  std::ifstream obsMean_file(obs_mean_path.string());
  std::ifstream obsVariance_file(obs_var_path.string());
  obs_.setZero(raiboController_.getObDim());
  obsMean_.setZero(raiboController_.getObDim());
  obsVariance_.setZero(raiboController_.getObDim());

  /// load observation mean and variance
  if (obsMean_file.is_open()) {
    for (int i = 0; i < obsMean_.size(); ++i) {
      std::getline(obsMean_file, in_line, '\n');
      obsMean_(i) = std::stof(in_line);
    }
  }
  else {
    RSFATAL_GUI("Invalid model number.")
  }

  if (obsVariance_file.is_open()) {
    for (int i = 0; i < obsVariance_.size(); ++i) {
      std::getline(obsVariance_file, in_line, '\n');
      obsVariance_(i) = std::stof(in_line);
    }
  }
  else {
    RSFATAL_GUI("Invalid model number.")
  }

  obsMean_file.close();
  obsVariance_file.close();

  command_.setZero();
  command_4_ << command_, locomotion_type_;

  joySubscribeTime_ = 0.;

  logIdx_ = dataLogger_.initializeAnotherDataGroup(
      "bipedal",
      "observation", raiboController_.getObservation(),
      "command", command_4_,
      "joySubscribeTime", joySubscribeTime_,
      "targetPosition", raiboController_.getJointPTarget(),
      "actualTorque", raiboController_.getJointTorque()
  );

  return true;
}

bool raiboLearningController::init() {
  return true;
}

bool raiboLearningController::advance() {
  controlEnd_ = std::chrono::high_resolution_clock::now();
  elapsedTime_ = std::chrono::duration_cast<std::chrono::microseconds>(controlEnd_ - controlBegin_).count() / 1.e6;

  // if (fabs(fmod(elapsedTime_, 0.0025)) < 1e-6) {
  //   // RSINFO("clip torque")
  //   raiboController_.clipTorque();
  // }

  if (elapsedTime_ < control_dt_) {
    return true;
  }
  else {
  /// 100Hz controller
    controlBegin_ = std::chrono::high_resolution_clock::now();

    robotHub_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    robotHub_->setPdGains(raiboController_.getJointPGain(), raiboController_.getJointDGain());

    raiboController_.updateObservation();
    raiboController_.advance(obsScalingAndGetAction().head(12));

    dataLogger_.append(logIdx_,
        raiboController_.getObservation(), command_4_, joySubscribeTime_, raiboController_.getJointPTarget(), raiboController_.getJointTorque());
  }

  return true;
}

Eigen::VectorXf raiboLearningController::obsScalingAndGetAction() {
  /// normalize the obs
  obs_ = raiboController_.getObservation().cast<float>();

  for (int i = 0; i < obs_.size(); ++i) {
    obs_(i) = (obs_(i) - obsMean_(i)) / std::sqrt(obsVariance_(i) + 1e-8);
  }
  /// forward the obs to the encoder
  Eigen::Matrix<float, OBSDIM, 1> encoder_input = obs_.head(OBSDIM);
  latent_ = encoder_.forward(encoder_input);

  /// concat obs and e_out and forward to the actor
  Eigen::Matrix<float, ENCOUTDIM + OBSDIM + COMDIM, 1> actor_input;
  actor_input << latent_, obs_;

  Eigen::VectorXf action = actor_.forward(actor_input);
  return action;
}

bool raiboLearningController::reset() {
  raiboController_.reset();
  encoder_.initHidden();
  command_.setZero();
//  controlBegin_ = std::chrono::high_resolution_clock::now();
  controlEnd_ = std::chrono::high_resolution_clock::now();
  joySubscribeEnd_ = std::chrono::high_resolution_clock::now();
  return true;
}

bool raiboLearningController::terminate() { return true; }

bool raiboLearningController::stop() { return true; }

extern "C" Controller * create(
    raisim::World & world, raisim::RaisimServer & server,
    raisim::World & worldSim, raisim::RaisimServer & serverSim, GlobalResource & globalResource)
{
  return new raiboLearningController(world, server, worldSim, serverSim, globalResource);
}

extern "C" void destroy(Controller *p) {
  delete p;
}

void raiboLearningController::setCommand(const std::shared_ptr<raisin_interfaces::srv::Vector3::Request> request,
                                          std::shared_ptr<raisin_interfaces::srv::Vector3::Response> response,
                                          const int locomotion_type)
try {
  command_ << request->x, request->y, request->z;
  raiboController_.setCommand(command_);
  response->success = true;
} catch (const std::exception &e) {
  response->success = false;
  response->message = e.what();
}

void raiboLearningController::joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
try {
  joySubscribeBegin_ = std::chrono::high_resolution_clock::now();
  joySubscribeTime_ = std::chrono::duration_cast<std::chrono::microseconds>(
      joySubscribeBegin_ - joySubscribeEnd_).count();

  if(msg->buttons[0]) {
    locomotion_type_ = 0; // Bipedal
  }
  else if(msg->buttons[1]) {
    locomotion_type_ = 1; // Quadrupedal
  }

  command_ << msg->axes[0], msg->axes[1], msg->axes[2];
  command_4_ << command_, locomotion_type_;
  raiboController_.setCommand(command_4_);

  joySubscribeEnd_ = std::chrono::high_resolution_clock::now();
} catch (const std::exception &e) {
  std::cout << e.what();
}

}

}
