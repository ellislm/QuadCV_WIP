#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

bool CalibrateMode = true;

int iLowR = 0;
int iHighR = 179; 
int iLowG = 100;
int iHighG = 255;
int iLowB = 100;
int iHighB = 255;
int minRadius = 5;

Mat imgOriginal;

const float pi = 3.14159;
const int MAX_NUM_OBJECTS = 50;

struct marker
{
  int x, y;
  int area;
};

string intToString(int number)
{

  std::stringstream ss;
  ss << number;
  return ss.str();

}
static void onMouse( int event, int x, int y, int f, void* )
{
 Mat image=imgOriginal.clone();
 Vec3b rgb=image.at<Vec3b>(y,x);
 int B=rgb.val[0];
 int G=rgb.val[1];
 int R=rgb.val[2];

  Mat HSV;
  Mat RGB=image(Rect(x,y,1,1));
  cvtColor(RGB, HSV,CV_BGR2HSV);

    Vec3b hsv=HSV.at<Vec3b>(0,0);
    int H=hsv.val[0];
    int S=hsv.val[1];
    int V=hsv.val[2];

    char name[30];
    sprintf(name,"B=%d",B);
    putText(image,name, Point(150,40) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );

    sprintf(name,"G=%d",G);
    putText(image,name, Point(150,80) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );

    sprintf(name,"R=%d",R);
    putText(image,name, Point(150,120) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );

    sprintf(name,"H=%d",H);
    putText(image,name, Point(25,40) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );

    sprintf(name,"S=%d",S);
    putText(image,name, Point(25,80) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );

    sprintf(name,"V=%d",V);
    putText(image,name, Point(25,120) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );

    sprintf(name,"X=%d",x);
    putText(image,name, Point(25,300) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,0,255), 2,8,false );

    sprintf(name,"Y=%d",y);
    putText(image,name, Point(25,340) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,0,255), 2,8,false );
    imshow("Original",image);//show original image
}
 //imwrite("hsv.jpg",image);
void createWindows()
{

  namedWindow("Control", CV_WINDOW_AUTOSIZE);

  cvCreateTrackbar("LowR", "Control", &iLowR, 179);//Red(0-255)
  cvCreateTrackbar("HighR","Control", &iHighR, 179);

  cvCreateTrackbar("LowG", "Control", &iLowG, 255);//Green(0-255)
  cvCreateTrackbar("HighGS","Control", &iHighG, 255);

  cvCreateTrackbar("LowB", "Control", &iLowB, 255);//Green(0-255)
  cvCreateTrackbar("HighB","Control", &iHighB, 255);

  cvCreateTrackbar("Radius","Control",&minRadius,20);
   return;
}
void drawCrosshair(vector<marker> markerVec, Mat &frame)
{
  for(int i = 0; i<markerVec.size(); i++)
  {
    int x = markerVec.at(i).x;
    int y = markerVec.at(i).y;
    int radius = sqrt(markerVec.at(i).area/pi);
    
    circle(frame,Point(x,y),radius,Scalar(0,255,0),2);
    putText(frame,intToString(x)+","+intToString(y),Point(x,y),1,1,Scalar(0,255,0),2);
  }
}

void morphOps(Mat &imgThresholded)
{
  //morphological openings (remove small objects from the foreground)
  erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(minRadius,minRadius)));
  dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(minRadius,minRadius)));

  //morphological closings (fill small holes in the foreground)
  dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(minRadius,minRadius)));
  erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(minRadius,minRadius)));
}
void trackFilteredObject(Mat threshold,Mat HSV, Mat &cameraFeed, vector<marker> &markerVec)
{
	int x,y;
//	vector<marker> markerVec;
	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );
	//use moments method to find our filtered object
//	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if(numObjects<MAX_NUM_OBJECTS){
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {
				marker marktemp;
				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if(area>minRadius*minRadius)
        {
					marktemp.x = moment.m10/area;
					marktemp.y = moment.m01/area;
					marktemp.area = area;
					markerVec.push_back(marktemp);
					objectFound = true;
				}
        else objectFound = false;

			}
			//let user know you found an object
			if(objectFound ==true)
      {
			//draw object location on screen
//			drawCrosshair(markerVec,cameraFeed);
      }

		}else putText(cameraFeed,"TOO MUCH NOISE! ADJUST FILTER",Point(0,50),1,2,Scalar(0,0,255),2);
	}
}

int main(int argc, char** argv)
{
  VideoCapture cap("quadvid.avi");
    Mat imgThresholded;
    Mat imgHSV;
    Mat upperRed;
//    Mat imgOriginal;

  if (!cap.isOpened())
  {
    cout<<"Cannot load camera"<< endl;
    return -1;
  }

  cap.set(CV_CAP_PROP_FPS, 60); //setting capture fps to 60

  createWindows();
  //grabbing first frame of video
  bool bSuccess = cap.read(imgOriginal);

  while (true)
  {
    vector<marker> markerVec;
    if(waitKey())
    {
        bSuccess = cap.read(imgOriginal);
    if (!bSuccess) //break loop if not successful
    {
      cout << "Cannot read a frame from video stream" << endl;
      break;
    }
  } 
    cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);
  //  upperRed = imgHSV.clone();
    inRange(imgHSV, Scalar(0, iLowG, iLowB), Scalar(10, iHighG, iHighB), imgThresholded);
    morphOps(imgThresholded);
    trackFilteredObject(imgThresholded,imgHSV, imgOriginal, markerVec);
    inRange(imgHSV, Scalar(160, iLowB), Scalar(179, iHighG, iHighB),imgThresholded);

    morphOps(imgThresholded);
    trackFilteredObject(imgThresholded,imgHSV, imgOriginal,markerVec);
  
	  drawCrosshair(markerVec,imgOriginal);
    imshow("HSV Image",imgHSV);

    imshow("Thresholded Image", imgThresholded); //show thresholded image
    imshow("Original",imgOriginal);//show original image
    setMouseCallback("Original",onMouse, 0);
    if(waitKey(30) == 27) //wait for esc key press for 30ms, if esc key is pressed, break loop
    {
      cout << "esc key pressed by user" << endl;
      break;
    }

  }

  return 0;

}
