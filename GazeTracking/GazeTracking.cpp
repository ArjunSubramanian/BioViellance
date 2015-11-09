// GazeTracking.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
/*#include  "cv.h"
#include <objdetect.hpp>
#include <highgui.hpp>
#include <imgproc.hpp>
#include <core.hpp>*/
#include  "opencv\cv.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <Windows.h>
#include <math.h>
#include "constants.h"
#include "LocalizeIris.h"
#include "DetectEyeRegion.h"
//Changes for Read file
#include <fstream>
/** Constants **/
#include <time.h>
#include <ctime>

/** Function Headers */
void detectAndDisplay( cv::Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

int named=0;
float Leftwidth ;
float LeftdeltaWMax ;
std::vector<cv::Rect> faces, eyes;
float Leftdeltawidth;  
float Leftdeltax ;
float DeltaX;
//Angle on Y Left Eye
float LeftHeight ;
float LeftdeltaHmax ;
int LeftdeltaHeight;
float Leftdeltay ;
int distancey;
int sinvalue,cosvalue;
int rsinvalue,rcosvalue;
//Angle on X Right Eye
float Rightdeltax;
float Rightdeltay;
cv::Rect eye_bb;
cv::Mat eye_tpl;
float interocular_distance_Calib;
float interocular_distance_current=100.0;
float Scale;
double gradienthreshold;
int Face_Width ;
int Face__height;
//File Reading Changes start
std::vector <int> Eye_num;
std::vector <float> Xpo;
std::vector <float> Ypo;
std::vector <float> Zpo;
std::vector <int> ColorR;
std::vector <int> ColorG;
std::vector <int> ColorB;
std::vector <int> IsConnected;
std::vector <float> Rpo;
std::vector <cv::Point> RotatedPoints_Left;
std::vector <cv::Point> RotatedPoints_Right;
int NumofPoints=0;
//Logging File Changes Starts Here
std::ofstream outfile;
//Logging File Changes Ends Here
int thickness=1;
std::vector<float> Angle;
std::vector<float> ComputeAngle(cv::Point Iris,int EyeBallWidth,int EyeBallHeight,int EyeLocationonX,int DisyanceY);
//File Reading Changes End
/** Function Headers */
void computeRPO();
void detectAndDisplay( cv::Mat frame );
void ProcessInputPoints(cv::String path);
void computeRotatedPoints(int x_left ,int y_left,int x_right, int y_right);
cv::Mat drawViellance(cv::Point left,cv::Point right,cv::Mat frame);
/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations



using namespace std;


int main( int argc, const char** argv ) 
{		Angle.resize(2);
		CvCapture* capture;
		cv::Mat frame;
  		cv::CascadeClassifier eye_cascade;
		gradienthreshold= kGradientThreshold;
		//File Reading Changes start
		
		ProcessInputPoints("HEllo");
		computeRPO();
		//File Reading Changes End
		setkGradientThreshold ( kGradientThreshold);
		Face_Width = kEyePercentWidth;
		Face__height =kEyePercentHeight;
	  // Load the cascades
		if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -10; };

		cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
		//cv::moveWindow(main_window_name, 400, 100);
		cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
		//cv::moveWindow(face_window_name, 10, 100);
	 
   
		createCornerKernels();
		ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
				  43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

	   // Read the video stream


		outfile.open("Output.txt", std::ios_base::app);
		 capture = cvCaptureFromCAM( 0 );    
		if( capture ) 
		{
	 
					while( true )
					{
					  frame = cvQueryFrame( capture );
					  // mirror it
	   
					  cv::flip(frame, frame, 1);
					  cv::putText(frame, "Press 'H/N' to increase/decreas the height ", cv::Point(10,15), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0,200,0));
					  cv::putText(frame, "Press'W/S' to decrease the width ", cv::Point(10,25), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0,200,0));
					  cv::putText(frame, "Press 'G/B' to increase/decrease the gradient threshold ", cv::Point(10,35), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0,200,0));
					  cv::putText(frame, "Press 'T/R' to increase/decrease the gradient threshold ", cv::Point(10,45), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0,200,0));
					  frame.copyTo(debugImage);

					  // Apply the classifier to the frame
					  if( !frame.empty() )
					  {
						detectAndDisplay( frame );
					  }
					  else 
					  {
						printf(" --(!) No captured frame -- Break!");
						break;
					  }
	  
					  //imshow(main_window_name,debugImage);

					  int c = cv::waitKey(10);
					  if( (char)c == 'g' ) 
					  {
						  gradienthreshold +=5;
						  setkGradientThreshold(gradienthreshold+5);
					  }
					  if( (char)c == 'w' ) 
					  {
					   Face_Width+=2;
					  }
						if( (char)c == 'h' ) 
					  {
							Face__height+=2;
					  }
					 if( (char)c == 'b' ) 
					 {
						 gradienthreshold -=5;
						 setkGradientThreshold(gradienthreshold-5); 
					 }
					  if( (char)c == 's' ) 
					  {
					   Face_Width-=2;
					  }
					  if( (char)c == 'n' ) 
					  {
							Face__height-=2;
					  }
					   if( (char)c == 't' ) 
					  {
						  thickness+=1;
					  }
					    if( (char)c == 'r' ) 
					  {
							thickness-=1;
					  }
					  if( (char)c == 'q' ) 
                                            break;

			}
					 outfile.close();
	  }

	  releaseCornerKernels();
	 
	  return 0;
}



void findEyes(cv::Mat frame_gray, cv::Rect face,cv::Mat frame) 
{
        printf("Beginning of function: findEyes\n");
	cv:: Mat grey = frame_gray.clone();
        printf("Before submatting frame_gray.  Size of frame_gray = %d %d. Size+position of face = %d %d\n", frame_gray.cols, frame_gray.rows, face.width+face.x, face.height+face.y);
        if(face.x+face.width > frame_gray.cols) {
          printf("old width = %d\n", face.width);
          face.width = frame_gray.cols - face.x;
          printf("new width = %d\n", face.width);
          printf("x, y = %d %d\n", face.x, face.y);
          //assert(false);
        }
        if(face.y+face.height > frame_gray.rows)  {
          printf("old height = %d\n", face.height);
          face.height = frame_gray.rows - face.y;
          printf("new height = %d\n", face.height);
          printf("x, y = %d %d\n", face.x, face.y);
          //assert(false);
        }
	cv::Mat faceROI = frame_gray(face);
        printf("After submatting frame_gray\n");
	cv::Mat debugFace = faceROI;
 
 
	if (kSmoothFaceImage) 
		{
			double sigma = kSmoothFaceFactor * face.width;
			GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
		}
  //-- Find eye regions and draw them
	int eye_region_width = face.width * (Face_Width/100.0);
	int eye_region_height = face.width * (Face__height/100.0);
	int eye_region_top = face.height * (kEyePercentTop/100.0);
        printf("Before rect\n");
	cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                         eye_region_top,eye_region_width,eye_region_height);
	cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                          eye_region_top,eye_region_width,eye_region_height);
        printf("After rect, but before findEyeCenter\n");
  
  //-- Find Eye Centers
	cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
	cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
        printf("After findEyeCenter\n");

  // get corner regions
	cv::Rect leftRightCornerRegion(leftEyeRegion);
	leftRightCornerRegion.width -= leftPupil.x;
	leftRightCornerRegion.x += leftPupil.x;
	leftRightCornerRegion.height /= 2;
	leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
	cv::Rect leftLeftCornerRegion(leftEyeRegion);
	leftLeftCornerRegion.width = leftPupil.x;
	leftLeftCornerRegion.height /= 2;
	leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
	cv::Rect rightLeftCornerRegion(rightEyeRegion);
	rightLeftCornerRegion.width = rightPupil.x;
	rightLeftCornerRegion.height /= 2;
	rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
	cv::Rect rightRightCornerRegion(rightEyeRegion);
	rightRightCornerRegion.width -= rightPupil.x;
	rightRightCornerRegion.x += rightPupil.x;
	rightRightCornerRegion.height /= 2;
	rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
	cv::rectangle(debugFace,leftRightCornerRegion,200);
	cv::rectangle(debugFace,leftLeftCornerRegion,200);
	cv::rectangle(debugFace,rightRightCornerRegion,200);
	cv::rectangle(debugFace,rightLeftCornerRegion,200);
        printf("After rectangles\n");

	leftPupil.x += leftEyeRegion.x;
	leftPupil.y += leftEyeRegion.y;
	rightPupil.x += rightEyeRegion.x;
	rightPupil.y += rightEyeRegion.y;
	//Angle on X Left Eye
	
	//File Reading Changes start
	Leftwidth = (leftRightCornerRegion.width+leftLeftCornerRegion.width);
	LeftHeight =leftRightCornerRegion.height;
	int distancey = (leftPupil.y-leftRightCornerRegion.y)*1.5;
	ComputeAngle(leftPupil,Leftwidth,LeftHeight,leftLeftCornerRegion.width,distancey);
	/*Leftwidth-=(Leftwidth*0.25);
	LeftdeltaWMax = Leftwidth-Leftwidth/2;
	Leftdeltawidth=	.75*leftLeftCornerRegion.width - Leftwidth/2;  
	DeltaX = Leftdeltawidth/LeftdeltaWMax;
	Leftdeltax = (asinf(DeltaX) * 180.0/3.14159265);
    printf("After delta\n");
 
	//Angle on Y Left Eye
	LeftHeight =leftRightCornerRegion.height;
    LeftHeight+=LeftHeight*0.50;
	LeftdeltaHmax = LeftHeight-LeftHeight/2;	
	LeftdeltaHeight=distancey-LeftHeight/2;
	Leftdeltay = (asinf(LeftdeltaHeight/LeftdeltaHmax) * 180.0/3.14159265);*/
	//File Reading Changes end
	circle(debugFace, rightPupil,3, 2000);
	circle(debugFace, leftPupil, 3, 2000);
	//Angle on X Right EYe
	Leftdeltax=Angle[0];
	Leftdeltay=Angle[1];
	Rightdeltax =  Leftdeltax;
	Rightdeltay = Leftdeltay;
	//File Reading Changes end
	rightPupil.x += face.x;
	rightPupil.y += face.y;
  
	leftPupil.x += face.x;
	leftPupil.y += face.y;
	if(interocular_distance_Calib==0)
	{
		interocular_distance_Calib=rightPupil.x-leftPupil.x;
	}
	interocular_distance_current=rightPupil.x-leftPupil.x;
	Scale = interocular_distance_current/interocular_distance_Calib;
	/*int x,y,xr,yr =0;  
	//float temp_x,temp_y,temp_xr,temp_yr;
	cosvalue=75*(cos(Leftdeltay*(3.14159265/180))*sin(Leftdeltay*(3.14159265/180)));	
	sinvalue=75*sin(Leftdeltax*(3.14159265/180))*cos(Leftdeltax*(3.14159265/180));
	rcosvalue=75*cos(Leftdeltay*(3.14159265/180))*sin(Leftdeltay*(3.14159265/180));
	rsinvalue=75*sin(Leftdeltax*(3.14159265/180))*cos(Leftdeltax*(3.14159265/180));
	x=leftPupil.x+sinvalue;
	y=leftPupil.y+cosvalue ;
	xr=rightPupil.x+rsinvalue;
	yr=rightPupil.y+rcosvalue;*/
	//Logging File Changes Starts Here
	time_t timer;
	struct tm y2k = {0};
	long seconds;

	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time(&timer);  /* get current time; same as: timer = time(NULL)  */

	seconds = timer;
	outfile << leftPupil.x<<","<<leftPupil.y<<","<<rightPupil.x <<"," << rightPupil.y<<","<<Angle[0]* 180.0/3.14159265<<","<<Angle[1]* 180.0/3.14159265<<","<<seconds <<endl;
	//Logging File Changes Ends Here
	computeRotatedPoints(leftPupil.x,leftPupil.y,rightPupil.x,rightPupil.y);
	cv::Mat img;
	img = drawViellance(leftPupil,rightPupil,frame);

	/*cv::line(frame,cv::Point(leftPupil.x,leftPupil.y),cv::Point(x-15*Scale,y+15*Scale),(255,255,0));
	cv::line(frame,cv::Point(leftPupil.x,leftPupil.y),cv::Point(x-15*Scale,y-15*Scale),(255,255,0));
	cv::line(frame,cv::Point(x-15*Scale,y-15*Scale),cv::Point(x-15*Scale,y+1 5*Scale),(255,0,0));
	cv::line(frame,cv::Point(leftPupil.x,leftPupil.y),cv::Point(x+15*Scale,y+15*Scale),(255,255,0));
	cv::line(frame,cv::Point(leftPupil.x,leftPupil.y),cv::Point(x+15*Scale,y-15*Scale),(255,255,0));
	cv::line(frame,cv::Point(x+15*Scale,y-15*Scale),cv::Point(x+15*Scale,y+15*Scale),(255,0,0));
	cv::line(frame,cv::Point(x-15*Scale,y+15*Scale),cv::Point(x+15*Scale,y+15*Scale),(255,0,0));
	cv::line(frame,cv::Point(x-15*Scale,y-15*Scale),cv::Point(x+15*Scale,y-15*Scale),(255,0,0));
	cv::line(frame,cv::Point(rightPupil.x,rightPupil.y),cv::Point(xr+15*Scale,yr+15*Scale),(255,255,0));
	cv::line(frame,cv::Point(rightPupil.x,rightPupil.y),cv::Point(xr+15*Scale,yr-15*Scale),(255,255,0));
	cv::line(frame,cv::Point(xr+15*Scale,yr+15*Scale),cv::Point(xr+15*Scale,yr-15*Scale),(255,0,0));
	cv::line(frame,cv::Point(rightPupil.x,rightPupil.y),cv::Point(xr-15*Scale,yr+15*Scale),(255,255,0));
	cv::line(frame,cv::Point(rightPupil.x,rightPupil.y),cv::Point(xr-15*Scale,yr-15*Scale),(255,255,0));
	cv::line(frame,cv::Point(xr-15*Scale,yr+15*Scale),cv::Point(xr-15*Scale,yr-15*Scale),(255,0,0));
	cv::line(frame,cv::Point(xr+15*Scale,yr+15*Scale),cv::Point(xr-15*Scale,yr+15*Scale),(255,0,0));
	cv::line(frame,cv::Point(xr+15*Scale,yr-15*Scale),cv::Point(xr-15*Scale,yr-15*Scale),(255,0,0));*/

	



  //-- Find Eye Corners
  if (kEnableEyeCorner) 
	{
		cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
		leftRightCorner.x += leftRightCornerRegion.x;
		leftRightCorner.y += leftRightCornerRegion.y;
		cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
		leftLeftCorner.x += leftLeftCornerRegion.x;
		leftLeftCorner.y += leftLeftCornerRegion.y;
		cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
		rightLeftCorner.x += rightLeftCornerRegion.x;
		rightLeftCorner.y += rightLeftCornerRegion.y;
		cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
		rightRightCorner.x += rightRightCornerRegion.x;
		rightRightCorner.y += rightRightCornerRegion.y;
		circle(faceROI, leftRightCorner, 3, 200);
		circle(faceROI, leftLeftCorner, 3, 200);
		circle(faceROI, rightLeftCorner, 3, 200);
		circle(faceROI, rightRightCorner, 3, 200);
  }
  cv::namedWindow("Hello",CV_WINDOW_NORMAL);
  circle(grey,rightPupil,3,2000);
  circle(grey,leftPupil,3,2000);
  imshow("Hello",frame);
  imshow(face_window_name, debugFace);
  //imwrite("frame100.jpg", frame); 
	 
        printf("End of function: findEyes\n");
}


cv::Mat findSkin (cv::Mat &frame) 
{
  cv::Mat input;
  cv::Mat output = cv::Mat(frame.rows,frame.cols, CV_8U);
  cvtColor(frame, input, CV_BGR2YCrCb);
  for (int y = 0; y < input.rows; ++y) 
  {
		const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
		//    uchar *Or = output.ptr<uchar>(y);
		cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
		for (int x = 0; x < input.cols; ++x) 
		{
			cv::Vec3b ycrcb = Mr[x];
//			Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
			if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) 
			{
				Or[x] = cv::Vec3b(0,0,0);
		
			}
		}
  }
  return output;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( cv::Mat frame ) 
{
  printf("Beginning of function: detectAndDisplay\n");
  std::vector<cv::Rect> faces;
  std::vector<cv::Rect> eyes;
  //cv::Mat frame_gray;
  //points4d.resize(4);
  std::vector<cv::Mat> rgbChannels(3);
  cv::split(frame, rgbChannels);
  printf("After split\n");
  cv::Mat frame_gray = rgbChannels[2];
  cv::Mat frame_, eye_tpl;
  //cvtColor( frame, frame_gray, CV_BGR2GRAY );
  //equalizeHist( frame_gray, frame_gray );
  //cv::pow(frame_gray, CV_64F, frame_gray);
  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
  //  findSkin(debugImage);
  printf("After detectMultiscale\n");
 
 
  if (faces.size() > 0) 
  {
    findEyes(frame_gray, faces[0],frame);
  }
  
  printf("End of function: detectAndDisplay\n");
}
//File Reading Changes start
void ProcessInputPoints(cv::String path)
{
	ifstream infile;
	infile.open ("test.txt");
	string line;
	int i=0;
	if (infile.is_open())
		{
		 while (! infile.eof() )
			{
			 NumofPoints++;
			 getline (infile,line);
			 istringstream ss( line );
			
				while (ss)
					{
						Xpo.resize(i);Ypo.resize(i);Zpo.resize(i);ColorR.resize(i);ColorG.resize(i);ColorB.resize(i);IsConnected.resize(i);
						string s;
						if (!getline( ss, s, ',' )) break;
						Eye_num.push_back((atoi(s.c_str())));
						if (!getline( ss, s, ',' )) break;
						Xpo.push_back((stof(s.c_str())));
						if (!getline( ss, s, ',' )) break;
						Ypo.push_back((stof(s.c_str())));
						if (!getline( ss, s, ',' )) break;
						Zpo.push_back((stof(s.c_str())));
						if (!getline( ss, s, ',' )) break;
						ColorR.push_back((atoi(s.c_str())));
						if (!getline( ss, s, ',' )) break;
						ColorG.push_back((atoi(s.c_str())));
						if (!getline( ss, s, ',' )) break;
						ColorB.push_back((atoi(s.c_str())));
						if (!getline( ss, s, ',' )) break;
						IsConnected.push_back((atoi(s.c_str())));
						i++;
					}
			}
		}

}
void computeRPO()
{int i=0;
//Rpo.resize(NumofPoints);
	for(i=0;i<NumofPoints;i++)
	{	
	float rposquare = ((Xpo[i]*Xpo[i])+(Ypo[i]*Ypo[i])+(Zpo[i]*Zpo[i]));
		float rpo= sqrt(rposquare);
		Rpo.push_back(rpo);
	}
}
std::vector<float> ComputeAngle(cv::Point Iris,int EyeBallWidth,int EyeBallHeight,int IrisDispacementOnX,int DisyanceY)
{		
		float AngleX,AngleY;
		float EyeBallScaleX;
		float EyeBallScaleY;
		float DisplacementOnX;
		float DisplacementOnY;
		float RatioOnX;
		float RatioOnY;
		boolean detected;
		//vector<float> Angle;
		EyeBallWidth-=(EyeBallWidth*0.25);
		EyeBallScaleX = EyeBallWidth-EyeBallWidth/2;
		DisplacementOnX=	.75*IrisDispacementOnX - EyeBallWidth/2;  
		
		RatioOnX = DisplacementOnX/EyeBallScaleX;
		if(RatioOnX>-1 && RatioOnX<1)
		{
			detected = true;
		}
		else
		{
				detected = false;
		}
		AngleX = (asinf(RatioOnX) * 180.0/3.14159265);
		
		EyeBallHeight+=EyeBallHeight*0.50;
		EyeBallScaleY = EyeBallHeight-EyeBallHeight/2;	
		DisplacementOnY=DisyanceY-EyeBallHeight/2;
				if(DisplacementOnY/EyeBallScaleY<-1 || DisplacementOnY/EyeBallScaleY>1){
				detected = false;
				}
				else
				{
					detected = true;
				}
		AngleY = (asinf(DisplacementOnY/EyeBallScaleY) * 180.0/3.14159265);
		Angle[0] = AngleX*(3.14159265/180);
		Angle[1]=AngleY*(3.14159265/180);
		if(detected)
		{
			outfile << 1 << ",";
		}
		else 
		{
		outfile << 1 << ",";
		}
		return Angle;

}

void computeRotatedPoints(int x_left ,int y_left,int x_right, int y_right)
{	RotatedPoints_Left.clear();
	RotatedPoints_Right.clear();
	int Rotated_x;int Rotated_Y;
	int iterator;
	for (iterator =0;iterator<NumofPoints;iterator++)
	{   if(Eye_num[iterator]==0)
			{
				Rotated_x =x_left+ interocular_distance_current*Xpo[iterator] + (interocular_distance_current*Rpo[iterator] * sin(Angle[0]) * cos (Angle[0]));
				Rotated_Y =y_left+ interocular_distance_current*Ypo[iterator] + (interocular_distance_current*Rpo[iterator] * sin(Angle[1]) * cos (Angle[1]));
				RotatedPoints_Left.push_back(cv::Point(Rotated_x,Rotated_Y));
			}
		else
			{
				Rotated_x =x_right+ interocular_distance_current*Xpo[iterator] + (interocular_distance_current*Rpo[iterator] * sin(Angle[0]) * cos (Angle[0]));
				Rotated_Y =y_right+ interocular_distance_current*Ypo[iterator] + (interocular_distance_current*Rpo[iterator] * sin(Angle[1]) * cos (Angle[1]));
				RotatedPoints_Right.push_back(cv::Point(Rotated_x,Rotated_Y));
			}
		}
	
}
cv::Mat drawViellance(cv::Point left,cv::Point right,cv::Mat frame)
{	
	int iterator =0;
	int MaxIteration = RotatedPoints_Left.size()*2;
	bool triangle=false;
	for (iterator =0;iterator<RotatedPoints_Left.size();iterator++)
		{
			
			cv::line(frame,left,RotatedPoints_Left[iterator],cv::Scalar(ColorR[iterator], ColorG[iterator], ColorB[iterator]),thickness);
			
			if(triangle)
				{
					triangle=false;
					cv::line(frame,RotatedPoints_Left[iterator-1],RotatedPoints_Left[iterator],cv::Scalar(ColorR[iterator], ColorG[iterator], ColorB[iterator]),thickness);
				}
			if(IsConnected[iterator]==1)
				{
					triangle=true;
					if(iterator==(RotatedPoints_Left.size()-1))
					{
						cv::line(frame,RotatedPoints_Left[iterator],RotatedPoints_Left[0],cv::Scalar(ColorR[iterator], ColorG[iterator], ColorB[iterator]),thickness);
					}
				}
		}
	triangle=false;
	for (iterator =0;iterator<RotatedPoints_Right.size();iterator++)
		{
			
			cv::line(frame,right,RotatedPoints_Right[iterator],cv::Scalar(ColorR[iterator+RotatedPoints_Right.size()], ColorG[iterator+RotatedPoints_Right.size()], ColorB[iterator+RotatedPoints_Right.size()]),thickness);
			if(triangle)
				{
					triangle=false;
					cv::line(frame,RotatedPoints_Right[iterator-1],RotatedPoints_Right[iterator],cv::Scalar(ColorR[iterator+RotatedPoints_Right.size()], ColorG[iterator+RotatedPoints_Right.size()], ColorB[iterator+RotatedPoints_Right.size()]),thickness);
				}
			
			if(IsConnected[iterator]==1)
				{
					triangle=true;
					if(iterator==(RotatedPoints_Right.size()-1))
					{
						cv::line(frame,RotatedPoints_Right[iterator],RotatedPoints_Right[0],cv::Scalar(ColorR[iterator+RotatedPoints_Right.size()], ColorG[iterator+RotatedPoints_Right.size()], ColorB[iterator+RotatedPoints_Right.size()]),thickness);
					}
				}

		}

	return frame;
}

//File Reading Changes Ends

 

