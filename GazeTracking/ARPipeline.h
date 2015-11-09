#ifndef ARPIPELINE_HPP
#define ARPIPELINE_HPP

////////////////////////////////////////////////////////////////////
// File includes:

#include "PatternDetector.h"
#include "CameraCalibration.h"
#include "GeometryTypes.h"
class ARPipeline
{
public:
  ARPipeline(const cv::Mat& patternImage, const CameraCalibration& calibration);

  bool processFrame(const cv::Mat& inputFrame);

  PatternDetector     m_patternDetector;
private:
	  
private:
  CameraCalibration   m_calibration;
  Pattern             m_pattern;
  PatternTrackingInfo m_patternInfo;
  //PatternDetector     m_patternDetector;
};

#endif