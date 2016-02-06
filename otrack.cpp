#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


int iLowH = 0;
int iHighH = 179;
int iLowS = 0;
int iHighS = 255;
int iLowV = 0;
int iHighV = 255;

const int MAX_NUM_OBJECTS = 50;
const int MIN_OBJECT_AREA = 20*20;

string intToString(int number)
{

  std::stringstream ss;
  ss << number;
  return ss.str();

}
void createWindows()
{

  namedWindow("Control", CV_WINDOW_AUTOSIZE);

  cvCreateTrackbar("LowH", "Control", &iLowH, 179);//Hue(0-179)
  cvCreateTrackbar("HighH","Control", &iHighH, 179);

  cvCreateTrackbar("LowS", "Control", &iLowS, 255);//Saturation(0-255)
  cvCreateTrackbar("HighS","Control", &iHighS, 255);

  cvCreateTrackbar("LowV", "Control", &iLowV, 255);//Value(0-255)
  cvCreateTrackbar("HighV","Control", &iHighV, 255);
  return;
}
void drawCrosshair(int x, int y,int obj_radius, Mat &frame)
{
  circle(frame,Point(x,y),obj_radius,Scalar(0,255,0),2);
  putText(frame,intToString(x)+","+intToString(y),Point(x,y+30),1,1,Scalar(0,255,0),2);
}

void morphOps(Mat &imgThresholded)
{
  //morphological openings (remove small objects from the foreground)
  erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
  dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));

  //morphological closings (fill small holes in the foreground)
  dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
  erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
}
void trackFilteredObject(Mat threshold,Mat HSV, Mat &cameraFeed)
{

	int x,y;

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

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if(area>MIN_OBJECT_AREA){
					x = moment.m10/area;
					y = moment.m01/area;

				

					objectFound = true;

				}else objectFound = false;


			}
			//let user know you found an object
			if(objectFound ==true)
      {
				//draw object location on screen
			drawCrosshair(x,y,20,cameraFeed);
      }

		}else putText(cameraFeed,"TOO MUCH NOISE! ADJUST FILTER",Point(0,50),1,2,Scalar(0,0,255),2);
	}
}

int main(int argc, char** argv)
{
  VideoCapture cap(0);

    Mat imgThresholded;
    Mat imgHSV;
    Mat imgOriginal;

  if (!cap.isOpened())
  {
    cout<<"Cannot load camera"<< endl;
    return -1;
  }

  cap.set(CV_CAP_PROP_FPS, 60); //setting capture fps to 60

  createWindows();

  while (true)
  {

    bool bSuccess = cap.read(imgOriginal);

    if (!bSuccess) //break loop if not successful
    {
      cout << "Cannot read a frame from video stream" << endl;
      break;
    }

    cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);

    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);
    morphOps(imgThresholded);

    imshow("Thresholded Image", imgThresholded); //show thresholded image
    imshow("Original",imgOriginal);//show original image
    imshow("HSV Image",imgHSV);
    trackFilteredObject(imgThresholded,imgHSV, imgOriginal);
    if(waitKey(30) == 27) //wait for esc key press for 30ms, if esc key is pressed, break loop
    {
      cout << "esc key pressed by user" << endl;
      break;
    }

  }

  return 0;

}
