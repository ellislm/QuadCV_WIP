#include<iostream>
#include<fstream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

bool CalibrateMode = true;
const float pi = 3.14159;
//the following are initial values for RGB filtering
int iLowR = 250;
int iHighR = 255; 
int iLowG = 0;
int iHighG = 255;
int iLowB = 0;
int iHighB = 255;
int minRadius = 3;
int maxRadius = 30;
double l_mean = 260;

ofstream logfile; //initializing log file

//Establishing matrices for use in matlab code

double phimat[64] = 
{
   1, 1, 0, 0, 0,0, 0, 0,
   0, 1, 0, 0, 0,0, 0, 0,
   0, 0, 1, 1, 0,0, 0, 0,
   0, 0, 0, 1, 0,0, 0, 0,
   0, 0, 0, 0, 1,1, 0, 0,
   0, 0, 0, 0, 0,1, 0, 0,
   0, 0, 0, 0, 0,0, 1, 1,
   0, 0, 0, 0, 0,0, 0, 1,
  };
double hmat[32] =
{
  1, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 1, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 1, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 1, 0,
};
double r = 0.25;
double qLocmat[4] = {9.7, 10.5,1.4,-5/360*pi};

Mat_<double> qLoc(1,4, qLocmat);
Mat imgOriginal;

int x_pix=640;
int y_pix = 480;
int frame_rate=60;

Mat_<double>x(1,3);
Mat_<double>v(1,3);

	double xmmatdata[8] = {10, 0, 10, 0, 1.4, 0, 0, 0};
	Mat_<double> xm(1,8,xmmatdata);
const int MAX_NUM_OBJECTS = 50;

//Object used to represent markers
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
void createWindows()
{

  namedWindow("Control", CV_WINDOW_AUTOSIZE);

  cvCreateTrackbar("Red Min", "Control", &iLowR, 255);//Red(0-255)
  cvCreateTrackbar("Red Max","Control", &iHighR, 255);

  cvCreateTrackbar("Green Min", "Control", &iLowG, 255);//Green(0-255)
  cvCreateTrackbar("Green Max","Control", &iHighG, 255);

  cvCreateTrackbar("Blue Min", "Control", &iLowB, 255);//Green(0-255)
  cvCreateTrackbar("Blue Max","Control", &iHighB, 255);

  cvCreateTrackbar("Radius Min","Control",&minRadius,30);
  cvCreateTrackbar("Radius Max","Control",&maxRadius,30);
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
				if((area > pi*minRadius*minRadius) && (area < pi*maxRadius*maxRadius))
        {
					marktemp.x = moment.m10/area;
					marktemp.y = moment.m01/area;
					marktemp.area = area;
					markerVec.push_back(marktemp);
					objectFound = true;
				}
        else objectFound = false;

			}

		}else putText(cameraFeed,"TOO MUCH NOISE! ADJUST FILTER",Point(0,50),1,2,Scalar(0,0,255),2);
	}
}
double median(Mat Input)
{    
Input = Input.reshape(0,1); // spread Input Mat to single row
vector<double> vecFromMat;
Input.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat
std::nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());
return vecFromMat[vecFromMat.size() / 2];
}

void matLabCode(vector<marker> mVec)
{
  double bin_cntr = 1;
  int length = mVec.size();
  vector<double> l;
  l_mean = 260;
  //  vector<double> xs;
//  vector<double> ys;
  Mat_<double> qLocP(1,2);
  Mat_<double> pRed(1,2);
  Mat_<double> pRed2(1,2);
  Mat_<double> eP(1,1);
  Mat_<double> qLocPsum(1,2);
	Mat_<double> pm = Mat_<double>::eye(8, 8);
  Mat_<double> workMat(1,2);//used for temp calculations
  Mat_<double> zeros(1,2);
  zeros(0) = 0;
  zeros(1) = 0;
double zmat[4];
  double ePsum;

	Mat_<double> q = pm.clone()*0.1;

	double l_sum;

  for (int i = 0; i < length; i++)
  {
    for(int j = i+1; j<length; j++)
    {
      l.push_back(sqrt(pow((mVec[i].x - mVec[j].x),2) + pow((mVec[i].y - mVec[j].y),2)));
    }
  }

  for (int i = 0; i < l.size(); i++)
  {
    if((l[i]< l_mean*1.3) && (l[i] > l_mean*0.7))
    {
      l_sum = l_sum+l[i]; 
      bin_cntr++;
    }
  }
  l_mean = l_sum/bin_cntr;
  qLocPsum(0) = 0; qLocPsum(1) = 0;
  for(int i = 0; i<length; i++)
  {
        pRed(i,0) = qLoc(0) + ((mVec[i].x - x_pix)/l_mean)*cos(qLoc(3)) - ((mVec[i].y - y_pix)/l_mean)*sin(qLoc(3));
        pRed(i,1) = qLoc(1) + ((mVec[i].y - y_pix)/l_mean)*sin(qLoc(3)) + ((mVec[i].y - y_pix)/l_mean)*cos(qLoc(3));
        pRed2(i,0) = round(pRed(i,0));// % correct marker locations
        pRed2(i,1) = round(pRed(i,1));// % correct marker locations
  //      rgbFrame = step(htextinsCent, rgbFrame, [uint16(pRed(i,1)) uint16(pRed(i,2))], [mVec[i].x mVec[i].y]);
        
        eP.push_back(sqrt((pRed(i,0) - pRed2(i,0))*(pRed(i,0) - pRed2(i,0))+
                          (pRed(i,1) - pRed2(i,1))*(pRed(i,1) - pRed2(i,1))));
        eP(i) = eP(i+1);
        qLocP(i,0) = pRed2(i,0) - ((mVec[i].x - x_pix)/l_mean)*cos(qLoc(3)) + ((mVec[i].y - y_pix)/l_mean)*sin(qLoc(3));
        qLocP(i,1) = pRed2(i,1) - ((mVec[i].y - y_pix)/l_mean)*sin(qLoc(3)) - ((mVec[i].y - y_pix)/l_mean)*cos(qLoc(3));        
        qLocPsum(0) = qLocPsum(0) + (1/eP(i)) * qLocP(i,0);
        qLocPsum(1) = qLocPsum(1) + (1/eP(i)) * qLocP(i,1);
        ePsum = ePsum + (1/eP(i));
        if(i!= length - 1)
        {
        pRed.push_back(zeros);
        pRed2.push_back(zeros);
        qLocP.push_back(zeros); 
        }
    }
		Mat_<double> angvar(1,1);
    double dx;
    double dy;
    for(int i=0; i < pRed2.rows; i++)
    {
      for (int j=0;i < pRed2.rows; i++)
      {
        if (pRed2(i,0) == pRed2(j,0) + 1 && pRed2(i,1) == pRed2(j,0)){
                dx = pRed(i,0) - pRed(j,0);
                dy = pRed(i,1) - pRed(j,1);
                angvar.push_back(atan2(-dy,dx)*180/pi);
            }
            if (pRed2(i,1) == pRed2(j,1) + 1 && pRed2(i,0) == pRed2(j,0)){
                dx = mVec[i].x - mVec[j].x;
                dy = mVec[i].y - mVec[j].y;
                angvar.push_back(atan2(dx,dy)*180/pi);
            }
      }
    }

    eP = eP.t();
    qLoc(0) = qLocPsum(0) / ePsum;
    qLoc(1) = qLocPsum(1) / ePsum;
    qLoc(2) = 322.5806/l_mean;
    qLoc(3) = median(angvar)*pi/180; 
    qLoc = qLoc.t();
    Mat_<double> h = Mat(4,8,CV_64F, &hmat);
    Mat_<double> phi = Mat(8,8,CV_64F, &phimat);
    Mat num = pm*h.t();
    Mat denom = h*pm*h.t() + r;

		Mat_<double> ka = num*denom.inv(DECOMP_LU); 
    for(int i = 0; i < qLocP.rows; i++)
    {
      zmat[0] = zmat[0] + qLocP(i,0)/qLocP.rows;
      zmat[1] = zmat[1] + qLocP(i,1)/qLocP.rows;
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
    //x.push_back(Mat_<double>(1,3, temp));
    //double duh[3] = {xh(1), xh(3), xh(5)};
    //v.push_back(Mat_<double>(1,3,duh));
    cout<<qLoc<<endl;
    logfile << qLoc(0) << "," << qLoc(1) << "," << qLoc(2) << ","
      << xh(1) << "," << xh(3) << "," << xh(5) << endl;
    
}
int main(int argc, char** argv)
{
  VideoCapture cap("quadvid.avi");
    Mat imgThresholded;
    Mat imgHSV;
    Mat upperRed;
//    Mat imgOriginal;
  logfile.open("data.csv");
  if (!cap.isOpened())
  { 
    cout<<"Cannot load camera"<< endl;
    return -1;
  }

	xm = xm.t();
	cap.set(CV_CAP_PROP_FPS, 60); //setting capture fps to 60

  createWindows();
  //grabbing first frame of video
  bool bSuccess = cap.read(imgOriginal);

  while (true)
  {
    vector<marker> markerVec;
    if(argc > 1)
    {
    if(waitKey() && string(argv[1]) == "-log")
    {
        bSuccess = cap.read(imgOriginal);
    }
    }
    else
    {
       bSuccess = cap.read(imgOriginal);
    }
       if (!bSuccess) //break loop if not successful
    {
      cout << "Cannot read a frame from video stream" << endl;
      logfile.close();
      break;
    }
     
    cvtColor(imgOriginal, imgHSV, COLOR_BGR2RGB);
  //  upperRed = imgHSV.clone();
    inRange(imgHSV, Scalar(iLowR, iLowB), Scalar(iHighR, iHighG, iHighB),imgThresholded);
    morphOps(imgThresholded);
    trackFilteredObject(imgThresholded,imgHSV, imgOriginal,markerVec);

	  drawCrosshair(markerVec,imgOriginal);
    matLabCode(markerVec);
    //The next line previously showed thresholded image, not necessary with marker crosshairs
//    imshow("Thresholded Image", imgThresholded); //show thresholded image
    imshow("Original",imgOriginal);//show original image
    setMouseCallback("Original",onMouse, 0);//Used to print color info on screen under mouse cursor
    if(waitKey(30) == 27) //wait for esc key press for 30ms, if esc key is pressed, break loop
    {
      cout << "esc key pressed by user" << endl;
      logfile.close();
      break;

    }

  }

  return 0;
  
}
