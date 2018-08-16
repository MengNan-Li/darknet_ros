/*
 * YoloObjectDetector.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

// yolo object detector
#include "darknet_ros/YoloObjectDetector.hpp"

// Check for xServer
#include <X11/Xlib.h>

/**
 * add_definitions(-DDARKNET_FILE_PATH="${DARKNET_PATH}")
 * 在CMakeList中定义的，可以通过#error来表示是否按照自己编译的流程来的
 */
#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

namespace darknet_ros {

char *cfg;
char *weights;
char *data;
char **detectionNames;

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh)
    : nodeHandle_(nh),
      imageTransport_(nodeHandle_),
      numClasses_(0),
      classLabels_(0),
      rosBoxes_(0),
      rosBoxCounter_(0)
{
  ROS_INFO("[YoloObjectDetector] Node started.");

  // Read parameters from config file.
  if (!readParameters()) {
    ROS_ERROR("readParameters() return error");
    ros::requestShutdown();
  }

  init();
}

YoloObjectDetector::~YoloObjectDetector()
{
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  yoloThread_.join();
}

bool YoloObjectDetector::readParameters()
{
  // Load common parameters.
  nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
  nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
  nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);
  /**
   *Xserver/X11，运行在能够显示界面的linux系统中，服务器版本则没有运行Xserver
   *服务器版本的如果需要在客户端显示界面的话，可以配置dispaly只向运行Xserver的客户机。 
   */
  // Check if Xserver is running on Linux.
  if (XOpenDisplay(NULL)) {
    // Do nothing!
    ROS_INFO("[YoloObjectDetector] Xserver is running.");
  } else {
    ROS_INFO("[YoloObjectDetector] Xserver is not running.");
    viewImage_ = false;
  }

  // Set vector sizes.
  nodeHandle_.param("yolo_model/detection_classes/names", classLabels_,
                    std::vector<std::string>(0));
  numClasses_ = classLabels_.size();
  //rosBoxes是numClasses×n的一个二维数组
  rosBoxes_ = std::vector<std::vector<RosBox_> >(numClasses_);
  rosBoxCounter_ = std::vector<int>(numClasses_);

  return true;
}

void YoloObjectDetector::init()
{
  ROS_INFO("[YoloObjectDetector] init().");

  // Initialize deep network of darknet.
  std::string weightsPath;
  std::string configPath;
  std::string dataPath;
  std::string configModel;
  std::string weightsModel;

  // Threshold of object detection.
  float thresh;
  nodeHandle_.param("yolo_model/threshold/value", thresh, (float) 0.3);

  // Path to weights file.
  nodeHandle_.param("yolo_model/weight_file/name", weightsModel,
                    std::string("yolov2-tiny.weights"));
  nodeHandle_.param("weights_path", weightsPath, std::string("/default"));
  weightsPath += "/" + weightsModel;
  weights = new char[weightsPath.length() + 1];
  strcpy(weights, weightsPath.c_str());

  //这个应该没用，detection，只需要模型cfg，weigets，class name
  // Path to config file.
  nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolov2-tiny.cfg"));
  nodeHandle_.param("config_path", configPath, std::string("/default"));
  configPath += "/" + configModel;
  cfg = new char[configPath.length() + 1];
  strcpy(cfg, configPath.c_str());

  // Path to data folder.
  dataPath = darknetFilePath_;
  dataPath += "/data";
  data = new char[dataPath.length() + 1];
  strcpy(data, dataPath.c_str());

  //C语言跟内存申请相关的函数主要有 alloca、calloc、malloc、free、realloc等
  // Get classes.
  detectionNames = (char**) realloc((void*) detectionNames, (numClasses_ + 1) * sizeof(char*));
  for (int i = 0; i < numClasses_; i++) {
    detectionNames[i] = new char[classLabels_[i].length() + 1];
    strcpy(detectionNames[i], classLabels_[i].c_str());
  }

  // Load network.
  setupNetwork(cfg, weights, data, thresh, detectionNames, numClasses_,
                0, 0, 1, 0.5, 0, 0, 0, 0);
  yoloThread_ = std::thread(&YoloObjectDetector::yolo, this);

  // Initialize publisher and subscriber.
  std::string cameraTopicName;
  int cameraQueueSize;
  std::string depthTopicName;
  int depthQueueSize;
  std::string depthScanTopicName;
  int depthScanQueueSize;
  std::string cameraInfoTopicName;
  int cameraInfoQueueSize;
  std::string objectDetectorTopicName;
  int objectDetectorQueueSize;
  bool objectDetectorLatch;
  std::string boundingBoxesTopicName;
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;
  std::string detectionImageTopicName;
  int detectionImageQueueSize;
  bool detectionImageLatch;
  std::string objectPoseTopicName;
  int objectPoseQueueSize;
  bool objectPoseLatch;

  nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName,
                    std::string("/camera/image_raw"));
  std::cout<<"subscribers/camera_reading/topic: "<< cameraTopicName << std::endl;
  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);

  nodeHandle_.param("subscribers/depth_reading/topic", depthTopicName,
                    std::string("/camera/image_raw"));
  std::cout<<"subscribers/depth_reading/topic: "<< depthTopicName << std::endl;
  nodeHandle_.param("subscribers/depth_reading/queue_size", depthQueueSize, 1);

  nodeHandle_.param("subscribers/depth_scan/topic", depthScanTopicName,
                    std::string("subscribers/depth_scan/topic"));
  std::cout<<"subscribers/depth_scan/topic: "<< depthScanTopicName << std::endl;
  nodeHandle_.param("subscribers/depth_scan/queue_size", depthScanQueueSize, 1);

  nodeHandle_.param("subscribers/camera_info/topic", cameraInfoTopicName,
                    std::string("/camera/image_raw"));
  std::cout<<"subscribers/camera_info/topic: "<< cameraInfoTopicName << std::endl;
  nodeHandle_.param("subscribers/camera_info/queue_size", cameraInfoQueueSize, 1);

  nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName,
                    std::string("found_object"));
  nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);
  nodeHandle_.param("publishers/bounding_boxes/topic", boundingBoxesTopicName,
                    std::string("bounding_boxes"));
  nodeHandle_.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
  nodeHandle_.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);
  nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName,
                    std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);
  nodeHandle_.param("publishers/object_pose/topic", objectPoseTopicName,
                    std::string("detection_image"));
  nodeHandle_.param("publishers/object_pose/queue_size", objectPoseQueueSize, 1);
  nodeHandle_.param("publishers/object_pose/latch", objectPoseLatch, true);



  //订阅图像
  imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, cameraQueueSize,
                                               &YoloObjectDetector::cameraCallback, this);

  //时间同步订阅边框和深度 --> 计算pose
  object_boxes_sub = new  message_filters::Subscriber<object_msgs::ObjectsInBoxes>(nodeHandle_, boundingBoxesTopicName, boundingBoxesQueueSize);
  depth_sub = new message_filters::Subscriber<sensor_msgs::Image>(nodeHandle_, depthTopicName, depthQueueSize); 
  depth_scan_sub = new message_filters::Subscriber<sensor_msgs::LaserScan>(nodeHandle_, depthScanTopicName, depthScanQueueSize);
  /*可以跑通，但是计算有问题  
  cameraInfo_sub = nodeHandle_.subscribe(cameraInfoTopicName,cameraInfoQueueSize, &YoloObjectDetector::depth_camera_info_callback, this);
  sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(200), *object_boxes_sub, *depth_sub);// 同步
  sync->registerCallback(boost::bind(&YoloObjectDetector::compute_pose_callback, this, _1, _2));  
*/

  /*订阅对象和sync必须是具备全局生命周期的，所以下面的局部变量就不合适了*/
  // typedef message_filters::sync_policies::ApproximateTime<object_msgs::ObjectsInBoxes,
  //       sensor_msgs::Image> MySyncPolicy;
  // message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(200), object_boxes_sub, depth_sub);// 同步
  // sync.registerCallback(boost::bind(&YoloObjectDetector::compute_pose_callback, this, _1, _2));   
  /*这种时间同步方式需要在队列中找时间一致的，比较严格*/
  // sync = new message_filters::TimeSynchronizer<object_msgs::ObjectsInBoxes, sensor_msgs::Image>
  //                                            (*object_boxes_sub, *depth_sub, 200);
  // sync->registerCallback(boost::bind(&YoloObjectDetector::compute_pose_callback, this, _1, _2));



/**bounding+depth scan得到scan的中心距离*/
  sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(200), *object_boxes_sub, *depth_scan_sub);// 同步
  sync->registerCallback(boost::bind(&YoloObjectDetector::object_scan_callback, this, _1, _2));  
  scanPublisher = nodeHandle_.advertise<sensor_msgs::LaserScan>("object_box_scan",10);
  dtl_.output_frame_id_ = "laser";
  dtl_.range_min_ = 0.1;
  dtl_.range_max_ = 8;

  //发布object的标号，和训练中的label标号顺序一样
  objectPublisher_ = nodeHandle_.advertise<std_msgs::Int8>(objectDetectorTopicName,
                                                           objectDetectorQueueSize,
                                                           objectDetectorLatch); 
  //发布被检测到的目标边框                                                       
  boundingBoxesPublisher_ = nodeHandle_.advertise<object_msgs::ObjectsInBoxes>(
      boundingBoxesTopicName, boundingBoxesQueueSize, boundingBoxesLatch);
  //发布带边框的图像
  detectionImagePublisher_ = nodeHandle_.advertise<sensor_msgs::Image>(detectionImageTopicName,
                                                                       detectionImageQueueSize,
                                                                       detectionImageLatch);
  objectPosePublisher_ = nodeHandle_.advertise<darknet_ros_msgs::ObjectPoses>(objectPoseTopicName,
                                                                       objectPoseQueueSize,
                                                                       objectPoseLatch);                                                                   

  // Action servers.
  std::string checkForObjectsActionName;
  nodeHandle_.param("actions/camera_reading/topic", checkForObjectsActionName,
                    std::string("check_for_objects"));
  checkForObjectsActionServer_.reset(
      new CheckForObjectsActionServer(nodeHandle_, checkForObjectsActionName, false));
  checkForObjectsActionServer_->registerGoalCallback(
      boost::bind(&YoloObjectDetector::checkForObjectsActionGoalCB, this));
  checkForObjectsActionServer_->registerPreemptCallback(
      boost::bind(&YoloObjectDetector::checkForObjectsActionPreemptCB, this));
  checkForObjectsActionServer_->start();
}

void YoloObjectDetector::depth_camera_info_callback(const sensor_msgs::CameraInfoConstPtr &cameraInfo){
  dtl_.cam_model_.fromCameraInfo(cameraInfo);
  cameraInfo_sub.shutdown();
}

void YoloObjectDetector::object_scan_callback(const object_msgs::ObjectsInBoxesConstPtr &objectInBoxes, 
                                               const sensor_msgs::LaserScanConstPtr &depth_scan)
{
  darknet_ros_msgs::ObjectPose op;
  darknet_ros_msgs::ObjectPoses ops;
  for(int i=0; i<objectInBoxes->objects_vector.size(); i++){
    int x_offset = objectInBoxes->objects_vector[i].roi.x_offset;
    // std::cout<<"x_offset:"<<x_offset<<std::endl;
    int width = objectInBoxes->objects_vector[i].roi.width;
    // std::cout<<"central point:"<<depth_scan->ranges[x_offset+width/10]<<std::endl;
    op.Class = objectInBoxes->objects_vector[i].object.object_name;
    op.probability = objectInBoxes->objects_vector[i].object.probability;
    
    //x=distance*sin(β))
    float y = depth_scan->ranges[x_offset+width/10] * sin(depth_scan->angle_min + (x_offset+width/10) * depth_scan->angle_increment);
    float x = depth_scan->ranges[x_offset+width/10] * cos(depth_scan->angle_min + (x_offset+width/10) * depth_scan->angle_increment);
    op.x = x;
    op.y = y;
    op.distance = depth_scan->ranges[x_offset+width/10];
    ops.object_poses.push_back(op);
  }
  objectPosePublisher_.publish(ops);

  /* 发布object scan方式，会有断隔
  sensor_msgs::LaserScanPtr depth_box_scan(new sensor_msgs::LaserScan());
  depth_box_scan->header = depth_scan->header;
  depth_box_scan->angle_min = depth_scan->angle_min;
  depth_box_scan->angle_max = depth_scan->angle_max;
  depth_box_scan->angle_increment = depth_scan->angle_increment;
  depth_box_scan->time_increment = depth_scan->time_increment;
  depth_box_scan->scan_time = depth_scan->scan_time;
  depth_box_scan->range_min = depth_scan->range_min;
  depth_box_scan->range_max = depth_scan->range_max;

  for(int i=0; i<objectInBoxes->objects_vector.size(); i++){
    int x_offset = objectInBoxes->objects_vector[i].roi.x_offset;
    int width = objectInBoxes->objects_vector[i].roi.width;
    // std::cout<<width<<std::endl;
    try
    {
      depth_box_scan->ranges.assign(depth_scan->ranges.size(), std::numeric_limits<float>::quiet_NaN());
      for(int j=x_offset; j<x_offset + width; j++){
         depth_box_scan->ranges[j] = depth_scan->ranges[j]; //这样会有断隔，因为边框附近并不是物体本身
      }
    }
    catch (std::runtime_error &e)
    {
        ROS_ERROR("[ depth_box_scan->ranges] exception: %s", e.what());
    }
    scanPublisher.publish(depth_box_scan);
  }
  */

}

void YoloObjectDetector::compute_pose_callback(const object_msgs::ObjectsInBoxesConstPtr &objectInBoxes, 
                                               const sensor_msgs::ImageConstPtr &depth)
{
  darknet_ros_msgs::ObjectPose op;
  darknet_ros_msgs::ObjectPose ops;
  std::cout<<"---"<<objectInBoxes->objects_vector.size()<<std::endl;

  cv::Mat depthMat1(depth->height, depth->width,  CV_32FC1, (void*)&depth->data[0]);
  cv::Mat depthMat;

  for(int i=0; i<objectInBoxes->objects_vector.size(); i++){
    // std::cout<<objectInBoxes->objects_vector[i].object.object_name<<std::endl;
    // std::cout<<objectInBoxes->objects_vector[i].object.probability<<std::endl;
    op.Class = objectInBoxes->objects_vector[i].object.object_name;
    op.probability = objectInBoxes->objects_vector[i].object.probability;
    int x = objectInBoxes->objects_vector[i].roi.x_offset + objectInBoxes->objects_vector[i].roi.width/2;
    int y = objectInBoxes->objects_vector[i].roi.y_offset + objectInBoxes->objects_vector[i].roi.height/2;

    std::cout<<"x_offset:"<<objectInBoxes->objects_vector[i].roi.x_offset<<"; y_offset:"<<objectInBoxes->objects_vector[i].roi.y_offset<<"; w:"<<
    objectInBoxes->objects_vector[i].roi.width<<"; h:"<<objectInBoxes->objects_vector[i].roi.height<<"; x:"<<x<<"; y:"<<y<<std::endl;


    std::cout<<depthMat1.rows<<"  "<<depthMat1.cols<<std::endl;
    
    depthMat = depthMat1(cv::Rect(objectInBoxes->objects_vector[i].roi.x_offset,
                                    objectInBoxes->objects_vector[i].roi.y_offset,
                                    objectInBoxes->objects_vector[i].roi.width,
                                    objectInBoxes->objects_vector[i].roi.height));
    std::cout<< "-------------"<<depthMat.at<int>(x,y)<<std::endl;
    std::cout<< "-------------"<<depth->encoding<<std::endl;
    double real_depth = 0;
    
   
    sensor_msgs::LaserScanPtr depth_box_scan(new sensor_msgs::LaserScan());

    int w = objectInBoxes->objects_vector[i].roi.width;
    int h = objectInBoxes->objects_vector[i].roi.height;
    int x_offset = objectInBoxes->objects_vector[i].roi.x_offset;
    int y_offset = objectInBoxes->objects_vector[i].roi.y_offset;

    if (depth->encoding == sensor_msgs::image_encodings::TYPE_16UC1)
    {
      dtl_.convert_point<uint16_t>(depth, dtl_.cam_model_, real_depth, x, y);
      //dtl_.convert_box<uint16_t>(depth, dtl_.cam_model_, depth_box_scan, x_offset, y_offset,w,h);
    }
    else if (depth->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
    {
      dtl_.convert_point<float>(depth, dtl_.cam_model_, real_depth, x, y);
      //dtl_.convert_box<float>(depth, dtl_.cam_model_, depth_box_scan, x_offset, y_offset,w,h);
    }
    else
    {
      std::stringstream ss;
      ss << "Depth image has unsupported encoding: " << depth->encoding;
      throw std::runtime_error(ss.str());
    }
    std::cout<<"real_depth:"<<real_depth<<std::endl;
    // sensor_msgs::LaserScanPtr scan_msg = dtl_.convert_msg(depth, cameraInfo);
    scanPublisher.publish(depth_box_scan);
  }

/*
  depthMat.convertTo(depthMat, CV_8UC1, 1.0 / 16);
  cv::Mat threshold_output;
  cv::threshold(depthMat, threshold_output, 20, 255, CV_THRESH_BINARY);
  //在Mat中找到障碍的轮廓
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  //找到障碍轮廓的凸形外廓
  std::vector<std::vector<cv::Point> >hull(contours.size());
  for (int i = 0; i < contours.size(); i++)
  {
      cv::convexHull(cv::Mat(contours[i]), hull[i], false);
  }
  //将面积小于100mm平方的凸形外廓当成噪点屏蔽
  std::vector<std::vector<cv::Point> > result;
  for (int i = 0; i< contours.size(); i++)
  {
      if (cv::contourArea(contours[i]) < 100)
      {
          continue;
      }
      result.push_back(hull[i]);
  }
  //绘制障碍轮廓
  cv::RNG rng(12345);
  cv::Mat drawing = cv::Mat::zeros(threshold_output.size(), CV_8UC3);
  for (int i = 0; i< contours.size(); i++)
  {
      cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
      cv::drawContours(drawing, contours, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
      cv::drawContours(drawing, hull, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
  }
  //显示
  cv::imshow("contours", drawing);
  cv::waitKey(30);
*/

}


void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr& msg)
{
  ROS_DEBUG("[YoloObjectDetector] USB image received.");

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    imageHeader_ = msg->header;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {//在unique_lock对象的声明周期内，它所管理的锁对象会一直保持上锁状态；而unique_lock的生命周期结束之后，它所管理的锁对象会被解锁
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::checkForObjectsActionGoalCB()
{
  ROS_DEBUG("[YoloObjectDetector] Start check for objects action.");

  boost::shared_ptr<const darknet_ros_msgs::CheckForObjectsGoal> imageActionPtr =
      checkForObjectsActionServer_->acceptNewGoal();
  sensor_msgs::Image imageAction = imageActionPtr->image;

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(imageAction, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexActionStatus_);
      actionId_ = imageActionPtr->id;
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::checkForObjectsActionPreemptCB()
{
  ROS_DEBUG("[YoloObjectDetector] Preempt check for objects action.");
  checkForObjectsActionServer_->setPreempted();
}

bool YoloObjectDetector::isCheckingForObjects() const
{
  return (ros::ok() && checkForObjectsActionServer_->isActive()
      && !checkForObjectsActionServer_->isPreemptRequested());
}

bool YoloObjectDetector::publishDetectionImage(const cv::Mat& detectionImage)
{
  if (detectionImagePublisher_.getNumSubscribers() < 1)
    return false;
  cv_bridge::CvImage cvImage;
  cvImage.header.stamp = ros::Time::now();
  cvImage.header.frame_id = "detection_image";
  cvImage.encoding = sensor_msgs::image_encodings::BGR8;
  cvImage.image = detectionImage;
  detectionImagePublisher_.publish(*cvImage.toImageMsg());
  ROS_DEBUG("Detection image has been published.");
  return true;
}

// double YoloObjectDetector::getWallTime()
// {
//   struct timeval time;
//   if (gettimeofday(&time, NULL)) {
//     return 0;
//   }
//   return (double) time.tv_sec + (double) time.tv_usec * .000001;
// }

int YoloObjectDetector::sizeNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      count += l.outputs;
    }
  }
  return count;
}

void YoloObjectDetector::rememberNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(predictions_[demoIndex_] + count, net->layers[i].output, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
}

detection *YoloObjectDetector::avgPredictions(network *net, int *nboxes)
{
  int i, j;
  int count = 0;
  fill_cpu(demoTotal_, 0, avg_, 1);
  for(j = 0; j < demoFrame_; ++j){
    axpy_cpu(demoTotal_, 1./demoFrame_, predictions_[j], 1, avg_, 1);
  }
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(l.output, avg_ + count, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
  detection *dets = get_network_boxes(net, buff_[0].w, buff_[0].h, demoThresh_, demoHier_, 0, 1, nboxes);
  return dets;
}

void *YoloObjectDetector::detectInThread()
{
  running_ = 1;
  float nms = .4;

  layer l = net_->layers[net_->n - 1];
  float *X = buffLetter_[(buffIndex_ + 2) % 3].data;
  float *prediction = network_predict(net_, X);

  rememberNetwork(net_);
  detection *dets = 0;
  int nboxes = 0;
  dets = avgPredictions(net_, &nboxes);

  if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

  if (enableConsoleOutput_) {
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps_);
    printf("Objects:\n\n");
  }
  image display = buff_[(buffIndex_+2) % 3];
  //该函数内会在终端不断输出检测信息
  draw_detections(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, demoClasses_);

  // extract the bounding boxes and send them to ROS
  int i, j;
  int count = 0;
  for (i = 0; i < nboxes; ++i) {
    float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
    float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
    float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
    float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

    if (xmin < 0)
      xmin = 0;
    if (ymin < 0)
      ymin = 0;
    if (xmax > 1)
      xmax = 1;
    if (ymax > 1)
      ymax = 1;

    // iterate through possible boxes and collect the bounding boxes
    for (j = 0; j < demoClasses_; ++j) {
      if (dets[i].prob[j]) {
        float x_center = (xmin + xmax) / 2;
        float y_center = (ymin + ymax) / 2;
        float BoundingBox_width = xmax - xmin;
        float BoundingBox_height = ymax - ymin;

        // define bounding box
        // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
        if (BoundingBox_width > 0.01 && BoundingBox_height > 0.01) {
          roiBoxes_[count].x = x_center;
          roiBoxes_[count].y = y_center;
          roiBoxes_[count].w = BoundingBox_width;
          roiBoxes_[count].h = BoundingBox_height;
          roiBoxes_[count].Class = j;
          roiBoxes_[count].prob = dets[i].prob[j];
          count++;
        }
      }
    }
  }

  // create array to store found bounding boxes
  // if no object detected, make sure that ROS knows that num = 0
  if (count == 0) {
    roiBoxes_[0].num = 0;
  } else {
    roiBoxes_[0].num = count;
  }

  free_detections(dets, nboxes);
  demoIndex_ = (demoIndex_ + 1) % demoFrame_;
  running_ = 0;
  return 0;
}

void *YoloObjectDetector::fetchInThread()
{
  IplImage* ROS_img = getIplImage();
  ipl_into_image(ROS_img, buff_[buffIndex_]);
  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    buffId_[buffIndex_] = actionId_;
  }
  rgbgr_image(buff_[buffIndex_]);
  letterbox_image_into(buff_[buffIndex_], net_->w, net_->h, buffLetter_[buffIndex_]);
  return 0;
}

void *YoloObjectDetector::displayInThread(void *ptr)
{
  show_image_cv(buff_[(buffIndex_ + 1)%3], "YOLO V3", ipl_);
  int c = cvWaitKey(waitKeyDelay_);
  if (c != -1) c = c%256;
  if (c == 27) {
      demoDone_ = 1;
      return 0;
  } else if (c == 82) {
      demoThresh_ += .02;
  } else if (c == 84) {
      demoThresh_ -= .02;
      if(demoThresh_ <= .02) demoThresh_ = .02;
  } else if (c == 83) {
      demoHier_ += .02;
  } else if (c == 81) {
      demoHier_ -= .02;
      if(demoHier_ <= .0) demoHier_ = .0;
  }
  return 0;
}

void *YoloObjectDetector::displayLoop(void *ptr)
{
  while (1) {
    displayInThread(0);
  }
}

void *YoloObjectDetector::detectLoop(void *ptr)
{
  while (1) {
    detectInThread();
  }
}

void YoloObjectDetector::setupNetwork(char *cfgfile, char *weightfile, char *datafile, float thresh,
                                      char **names, int classes,
                                      int delay, char *prefix, int avg_frames, float hier, int w, int h,
                                      int frames, int fullscreen)
{
  demoPrefix_ = prefix;
  demoDelay_ = delay;
  demoFrame_ = avg_frames;
  image **alphabet = load_alphabet_with_file(datafile);
  demoNames_ = names;
  demoAlphabet_ = alphabet;
  demoClasses_ = classes;
  demoThresh_ = thresh;
  demoHier_ = hier;
  fullScreen_ = fullscreen;
  printf("YOLO V3\n");
  net_ = load_network(cfgfile, weightfile, 0);
  set_batch_network(net_, 1);
}

void YoloObjectDetector::yolo()
{
  const auto wait_duration = std::chrono::milliseconds(2000);
  while (!getImageStatus()) {
    printf("Waiting for image.\n");
    if (!isNodeRunning()) {
      printf("darknet_ros node shutdown\n");
      return;
    }
    std::this_thread::sleep_for(wait_duration);
  }

  std::thread detect_thread;
  std::thread fetch_thread;

  srand(2222222);

  int i;
  demoTotal_ = sizeNetwork(net_);
  predictions_ = (float **) calloc(demoFrame_, sizeof(float*));
  for (i = 0; i < demoFrame_; ++i){
      predictions_[i] = (float *) calloc(demoTotal_, sizeof(float));
  }
  avg_ = (float *) calloc(demoTotal_, sizeof(float));

  layer l = net_->layers[net_->n - 1];
  roiBoxes_ = (darknet_ros::RosBox_ *) calloc(l.w * l.h * l.n, sizeof(darknet_ros::RosBox_));

  IplImage* ROS_img = getIplImage();
  buff_[0] = ipl_to_image(ROS_img);
  buff_[1] = copy_image(buff_[0]);
  buff_[2] = copy_image(buff_[0]);
  buffLetter_[0] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[1] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[2] = letterbox_image(buff_[0], net_->w, net_->h);
  ipl_ = cvCreateImage(cvSize(buff_[0].w, buff_[0].h), IPL_DEPTH_8U, buff_[0].c);

  int count = 0;

  if (!demoPrefix_ && viewImage_) {
    cvNamedWindow("YOLO V3", CV_WINDOW_NORMAL);
    if (fullScreen_) {
      cvSetWindowProperty("YOLO V3", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
      cvMoveWindow("YOLO V3", 0, 0);
      cvResizeWindow("YOLO V3", 640, 480);
    }
  }

  demoTime_ = what_time_is_it_now();

  while (!demoDone_) {
    buffIndex_ = (buffIndex_ + 1) % 3;
    fetch_thread = std::thread(&YoloObjectDetector::fetchInThread, this);
    detect_thread = std::thread(&YoloObjectDetector::detectInThread, this);
    if (!demoPrefix_) {
      fps_ = 1./(what_time_is_it_now() - demoTime_);
      demoTime_ = what_time_is_it_now();
      if (viewImage_) {
        displayInThread(0);
      }
      publishInThread();
    } else {
      char name[256];
      sprintf(name, "%s_%08d", demoPrefix_, count);
      save_image(buff_[(buffIndex_ + 1) % 3], name);
    }
    fetch_thread.join();
    detect_thread.join();
    ++count;
    if (!isNodeRunning()) {
      demoDone_ = true;
    }
  }

}

IplImage* YoloObjectDetector::getIplImage()
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
  IplImage* ROS_img = new IplImage(camImageCopy_);
  return ROS_img;
}

bool YoloObjectDetector::getImageStatus(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
  return imageStatus_;
}

bool YoloObjectDetector::isNodeRunning(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
  return isNodeRunning_;
}

void *YoloObjectDetector::publishInThread()
{
  // Publish image.
  cv::Mat cvImage = cv::cvarrToMat(ipl_);
  if (!publishDetectionImage(cv::Mat(cvImage))) {
    ROS_DEBUG("Detection image has not been broadcasted.");
  }

  // Publish bounding boxes and detection result.
  int num = roiBoxes_[0].num;
  if (num > 0 && num <= 100) {
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < numClasses_; j++) {
        if (roiBoxes_[i].Class == j) {
          rosBoxes_[j].push_back(roiBoxes_[i]);
          rosBoxCounter_[j]++;
        }
      }
    }

    std_msgs::Int8 msg;
    msg.data = num;
    objectPublisher_.publish(msg);

    for (int i = 0; i < numClasses_; i++) {
      if (rosBoxCounter_[i] > 0) {
        object_msgs::ObjectInBox boundingBox1;
        darknet_ros_msgs::BoundingBox boundingBox;

        for (int j = 0; j < rosBoxCounter_[i]; j++) {
          int xmin = (rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymin = (rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_;
          int xmax = (rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymax = (rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_;

          boundingBox1.object.object_name = classLabels_[i];
          boundingBox1.object.probability = rosBoxes_[i][j].prob;
          boundingBox1.roi.x_offset = xmin;
          boundingBox1.roi.y_offset = ymin;
          boundingBox1.roi.height = ymax-ymin;
          boundingBox1.roi.width = xmax-xmin;
          boundingBoxesResults_1.objects_vector.push_back(boundingBox1);


          boundingBox.Class = classLabels_[i];
          boundingBox.probability = rosBoxes_[i][j].prob;
          boundingBox.xmin = xmin;
          boundingBox.ymin = ymin;
          boundingBox.xmax = xmax;
          boundingBox.ymax = ymax;
          boundingBoxesResults_.bounding_boxes.push_back(boundingBox);
        }
      }
    }
    //imageHeader_.stamp = ros::Time();
    boundingBoxesResults_.header = imageHeader_;
    boundingBoxesResults_1.header = imageHeader_;
    boundingBoxesPublisher_.publish(boundingBoxesResults_1);
  } else {
    std_msgs::Int8 msg;
    msg.data = 0;
    objectPublisher_.publish(msg);
  }
  if (isCheckingForObjects()) {
    ROS_DEBUG("[YoloObjectDetector] check for objects in image.");
    darknet_ros_msgs::CheckForObjectsResult objectsActionResult;
    objectsActionResult.id = buffId_[0];
    objectsActionResult.bounding_boxes = boundingBoxesResults_;
    checkForObjectsActionServer_->setSucceeded(objectsActionResult, "Send bounding boxes.");
  }
  boundingBoxesResults_.bounding_boxes.clear();
  boundingBoxesResults_1.objects_vector.clear();

  for (int i = 0; i < numClasses_; i++) {
    rosBoxes_[i].clear();
    rosBoxCounter_[i] = 0;
  }

  return 0;
}


} /* namespace darknet_ros*/
