/*
 * yolo_obstacle_detector_node.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */
#include <pluginlib/class_list_macros.h>

#include <darknet_ros/YoloObjectDetector.hpp>
#include <nodelet/nodelet.h>
#include <ros/ros.h>

namespace darknet_ros {
class Nodelet_darknet_ros : public nodelet::Nodelet {

public:
  /** Nodelet onInit  function */
  virtual void onInit();
  ~Nodelet_darknet_ros() = default;
private:
  darknet_ros::YoloObjectDetector *yoloObjectDetector;
};
}

namespace darknet_ros {
void Nodelet_darknet_ros::onInit() {
  ros::NodeHandle nodeHandle("darknet_ros");
  //ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,ros::console::levels::Debug);
  yoloObjectDetector = new darknet_ros::YoloObjectDetector(nodeHandle);
}
}

PLUGINLIB_EXPORT_CLASS(darknet_ros::Nodelet_darknet_ros, nodelet::Nodelet)
