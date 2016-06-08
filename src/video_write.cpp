#include<iostream>
#include<fstream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int x_res = 320;
int y_res = 240;
int frame_rate=30;
string file_name = "video/picam.avi";


//Object used to represent markers
int main(int argc, char** argv)
{
   Mat imgOriginal;
   VideoCapture cap(0);
   VideoWriter video;
   video.open(file_name, CV_FOURCC('L','A','G','S'), frame_rate,Size (x_res,y_res), true );

  if (!cap.isOpened() || !video.isOpened())
  { 
    video.release();
    cout<<"Cannot load camera or video writer"<< endl;
    return -1;
  }
  cap.set(CV_CAP_PROP_FPS, frame_rate); //setting capture fps to 60
  cap.set(CV_CAP_PROP_FRAME_WIDTH,x_res);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, y_res);

  cout << "Starting Stream" << endl;
  while (true)
  {
    bool bSuccess = cap.read(imgOriginal);

       if (!bSuccess) //break loop if not successful
    {
      cout << "Cannot read a frame from video stream" << endl;
      video.release();
      break;
    }
    video.write(imgOriginal);
    imshow("Original",imgOriginal);//show original image
    if(waitKey(30) == 27) //wait for esc key press for 30ms, if esc key is pressed, break loop
    {
      cout << "esc key pressed by user" << endl;
      video.release();
      break;

    }

  }

 return 0;
}
