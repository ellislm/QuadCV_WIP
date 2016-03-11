#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

bool CalibrateMode = true;
const float pi = 3.14159;
int iLowR = 0;
int iHighR = 255; 
int iLowG = 100;
int iHighG = 255;
int iLowB = 100;
int iHighB = 255;
int minRadius = 5;
double l_mean = 260;
//vector<double> x_store, y_store;
double phimat[8][8] = 
{
  { 1, 1, 0, 0, 0,0, 0, 0,},
  { 0, 1, 0, 0, 0,0, 0, 0,},
  { 0, 0, 1, 1, 0,0, 0, 0,},
  { 0, 0, 0, 1, 0,0, 0, 0,},
  { 0, 0, 0, 0, 1,1, 0, 0,},
  { 0, 0, 0, 0, 0,1, 0, 0,},
  { 0, 0, 0, 0, 0,0, 1, 1,},
  { 0, 0, 0, 0, 0,0, 0, 1,},
  };
double hmat[4][8] =
{
  {1, 0, 0, 0, 0, 0, 0, 0, },
  {0, 0, 1, 0, 0, 0, 0, 0, },
  {0, 0, 0, 0, 1, 0, 0, 0, },
  {0, 0, 0, 0, 0, 0, 1, 0, },
};
double zmat[4];
double r = 0.25;
double qLocmat[4] = {9.7, 10.5,1.4,-5/360*pi};
Mat imgOriginal;
int x_pix=640;
int y_pix = 480;
int frame_rate=60;
Mat_<double>x(1,3);
Mat_<double>v(1,3);
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
double median( cv::Mat channel )
{
		double m = (channel.rows*channel.cols) / 2;
		int bin = 0;
		double med = -1.0;

		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true;
		bool accumulate = false;
		cv::Mat hist;
		cv::calcHist( &channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

		for ( int i = 0; i < histSize && med < 0.0; ++i )
		{
				bin += cvRound( hist.at< float >( i ) );
				if ( bin > m && med < 0.0 )
						med = i;
		}

		return med;
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
void createWindows()
{

  namedWindow("Control", CV_WINDOW_AUTOSIZE);

  cvCreateTrackbar("LowR", "Control", &iLowR, 255);//Red(0-255)
  cvCreateTrackbar("HighR","Control", &iHighR, 255);

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
void matLabCode(vector<marker> mVec)
{
  int bin_cntr = 1;
  int length = mVec.size();
  vector<double> l;
//  vector<double> xs;
//  vector<double> ys;
  Mat_<double> qLocP(1,2);
  Mat_<double> pRed(1,2);
  Mat_<double> pRed2(1,2);
  Mat_<double> eP(1,1);
  Mat_<double> qLoc(1,4, qLocmat);
  Mat_<double> qLocPsum(1,2);
	Mat_<double> pm = Mat_<double>::eye(8, 8);
  Mat_<double> workMat(1,2);//used for temp calculations
  double ePsum;

	double xmmatdata[8] = {10, 0, 10, 0, 1.4, 0, 0, 0};
	Mat_<double> xm(1,8,xmmatdata);
	xm = xm.t();
	Mat_<double> q = pm.clone()*0.1;

	double l_sum;
  for (int i = 0; i < l.size(); i++)
  {
    if(l[i]< l_mean*1.3 && l[i] > l_mean*0.7)
    {
      l_sum = l_sum+l[i]; 
      bin_cntr++;
    }
  }
  l_mean = l_sum/bin_cntr;
  for(int i = 0; i<length; i++)
  {
        workMat(1) = qLoc(1) + ((mVec[i].x - x_pix)/l_mean)*cos(qLoc(4)) - ((mVec[i].y - y_pix)/l_mean)*sin(qLoc(4));
        workMat(2) = qLoc(2) + ((mVec[i].y - y_pix)/l_mean)*sin(qLoc(4)) + ((mVec[i].y - y_pix)/l_mean)*cos(qLoc(4));
        pRed.push_back(workMat);
        
        workMat(1) = round(pRed(i,1));// % correct marker locations
        workMat(2) = round(pRed(i,2));// % correct marker locations
        pRed2.push_back(workMat);

  //      rgbFrame = step(htextinsCent, rgbFrame, [uint16(pRed(i,1)) uint16(pRed(i,2))], [mVec[i].x mVec[i].y]);
        
        eP.push_back(sqrt((pRed(i,1) - pRed2(i,1))*(pRed(i,1) - pRed2(i,1))+
                          (pRed(i,1) - pRed2(i,1))*(pRed(i,1) - pRed2(i,1))));
        qLocP(i,1) = pRed2(i,1) - ((mVec[i].x - x_pix)/l_mean)*cos(qLoc(4)) + ((mVec[i].y - y_pix)/l_mean)*sin(qLoc(4));
        qLocP(i,2) = pRed2(i,2) - ((mVec[i].y - y_pix)/l_mean)*sin(qLoc(4)) - ((mVec[i].y - y_pix)/l_mean)*cos(qLoc(4));        
        
        qLocPsum(1) = qLocPsum(1) + (1/eP(i)) * qLocP(i,1);
        qLocPsum(2) = qLocPsum(2) + (1/eP(i)) * qLocP(i,2);
        ePsum = ePsum + (1/eP(i));
    }

		Mat angvar(1,1,CV_32F);
    double dx;
    double dy;
    for(int i=0; i < pRed2.rows; i++)
    {
      for (int j=0;i < pRed2.rows; i++)
      {
        if (pRed2(i,1) == pRed2(j,1) + 1 && pRed2(i,2) == pRed2(j,2)){
                dx = pRed(i,1) - pRed(j,1);
                dy = pRed(i,2) - pRed(j,2);
                angvar.push_back(atan2(-dy,dx)*180/pi);
            }
            if (pRed2(i,2) == pRed2(j,2) + 1 && pRed2(i,1) == pRed2(j,1)){
                dx = mVec[i].x - mVec[j].x;
                dy = mVec[i].y - mVec[j].y;
                angvar.push_back(atan2(dx,dy)*180/pi);

            }
      }
    }

    eP = eP.t();
    qLoc(1) = qLocPsum(1) / ePsum;
    qLoc(2) = qLocPsum(2) / ePsum;
    qLoc(3) = 322.5806*1/l_mean;
    qLoc(4) = median(angvar)*pi/180;
    qLoc = qLoc.t();

    Mat h = Mat(4,8, CV_32F, &hmat);
    Mat phi = Mat(8,8, CV_32F, &phimat);

		Mat ka = pm*h.t()/(h*pm*h.t() + r); for(int i = 0; i < qLocP.rows; i++)
    {
      zmat[0] = zmat[0] + qLocP(i,1)/qLocP.rows;
      zmat[1] = zmat[1] + qLocP(i,2)/qLocP.rows;
    }
    zmat[2] = 322.5806/l_mean;
    zmat[3] = qLoc(4);
    Mat_<double> z(1,4,zmat);
    z = z.t();
		Mat_<double> xh = xm+ka*(z-h*xm);
    Mat_<double> p = Mat_<double>::eye(8,8); 
    xm = phi*xh;
    pm = phi*p*phi.t()+q;
    
    qLoc(0) = xh(0) + xh(1)*1/frame_rate;
    qLoc(1) = xh(2) + xh(3)*1/frame_rate;
    qLoc(2) = xh(4) + xh(5)*1/frame_rate;
    qLoc(3) = xh(6) + xh(7)*1/frame_rate;
    double temp[3] = {qLoc(0),qLoc(1),qLoc(2)};
    x.push_back(Mat_<double>(1,3, temp));
    double duh[3] = {xh(1), xh(3), xh(5)};
    v.push_back(Mat_<double>(1,3,duh));
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
    cvtColor(imgOriginal, imgHSV, COLOR_BGR2RGB);
  //  upperRed = imgHSV.clone();
    inRange(imgHSV, Scalar(iLowR, iLowB), Scalar(iHighR, iHighG, iHighB),imgThresholded);
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
