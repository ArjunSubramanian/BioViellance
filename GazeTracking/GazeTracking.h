#ifndef GAZE_TRACKING_H
#define GAZE_TRACKING_H
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include "CameraCalibration.h"
#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

void detectAndDisplay( cv::Mat frame );
bool processFrame(const cv::Mat& cameraFrame,cv::Mat& patternImage, CameraCalibration calibration	)
#endif