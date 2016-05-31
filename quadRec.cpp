#include<iostream>
#include<fstream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

bool CalibrateMode = true;
const float pi = 3.14159;
//the following are initial values for RGB filtering
int iLowR = 0;
int iHighR = 10; 
int iLowR2 = 170;
int iHighR2 = 179; 
int iLowG = 75;
int iHighG = 255;
int iLowB = 75;
int iHighB = 255;
int minRadius = 1;
int maxRadius = 15;

int frame_num = 0;

int x_res = 320;
int y_res = 240;
double l_mean = x_res/2.4;
ofstream logfile; //initializing log file

//Establishing matrices for use in matlab code

double phimat[64] = 
{1,1,0,0,0,0,0,0,
 0,1,0,0,0,0,0,0,
 0,0,1,1,0,0,0,0,
 0,0,0,1,0,0,0,0,
 0,0,0,0,1,1,0,0,
 0,0,0,0,0,1,0,0,
 0,0,0,0,0,0,1,1,
 0,0,0,0,0,0,0,1
};
double hmat[32] =
{
 1,0,0,0,0,0,0,0,
 0,0,1,0,0,0,0,0,
 0,0,0,0,1,0,0,0,
 0,0,0,0,0,0,1,0
};

Mat_<double> h = Mat(4,8,CV_64F, &hmat);
Mat_<double> phi = Mat(8,8,CV_64F, &phimat);

double r = 0.25;
double qLocmat[4] = {9.7, 10.5,1.4,-5/360*pi};

Mat_<double> qLoc(1,4, qLocmat);
Mat imgOriginal;

Mat_<double> eye = Mat_<double>::eye(8,8); 
int x_pix=x_res/2;
int y_pix = y_res/2;
int frame_rate=30;
clock_t time_previous;

	Mat_<double> pm = Mat_<double>::eye(8,8);
	double xmmatdata[8] = {10, 0, 10, 0, 1.4, 0, 0, 0};
	Mat_<double> xm(1,8,xmmatdata);
  Mat_<double> q = eye*0.1;
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
void createWindows()
{

  namedWindow("Control", CV_WINDOW_AUTOSIZE);

  cvCreateTrackbar("LowR", "Control", &iLowR, 179);//Red(0-255)
  cvCreateTrackbar("HighR","Control", &iHighR, 179);

  cvCreateTrackbar("LowR2", "Control", &iLowR2, 179);//Red(0-255)
  cvCreateTrackbar("HighR2","Control", &iHighR2, 179);

  cvCreateTrackbar("LowS", "Control", &iLowG, 255);//Green(0-255)
  cvCreateTrackbar("HighS","Control", &iHighG, 255);

  cvCreateTrackbar("LowV", "Control", &iLowB, 255);//Green(0-255)
  cvCreateTrackbar("HighV","Control", &iHighB, 255);

  cvCreateTrackbar("Radius Min","Control",&minRadius,30);
  cvCreateTrackbar("Radius Max","Control",&maxRadius,200);
   return;
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
double median(vector<double> v)
{
size_t n = v.size() / 2;
nth_element(v.begin(), v.begin()+n, v.end());
int vn = v[n];
if(v.size()%2 == 1)
{
  return vn;
}else
 {
 std::nth_element(v.begin(), v.begin()+n-1, v.end());
  return 0.5*(vn+v[n-1]);
   }
}
//double median(Mat Input)
//{    
//Input = Input.reshape(0,1); // spread Input Mat to single row
//vector<double> vecFromMat;
//Input.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat
//std::nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());
//return vecFromMat[vecFromMat.size() / 2];
//}
void matLabCode(vector<marker> mVec)
{
  double bin_cntr = 0;
  int length = mVec.size();
  if(length == 0) return;
  vector<double> l;
  //  vector<double> xs;
  //  vector<double> ys;
  Mat_<double> qLocP(1,2);
  Mat_<double> pRed(1,2);
  Mat_<double> pRed2(1,2);
  Mat_<double> eP(1,1);
  Mat_<double> qLocPsum(1,2);
  Mat_<double> workMat(1,2);//used for temp calculations
  Mat_<double> zeros(1,2);
  Mat_<double> expMat(1,2);
  zeros(0) = 0;
  Mat_<double> zero1(1,1);
  zeros(1) = 0;
  double zmat[4] = {0,0,0,0};
  double ePsum = 0;


	double l_sum = 0;
  for (int i = 0; i < length -1; i++)
  {
    for(int j = i+1; j<length; j++)
    {
      l.push_back(sqrt(pow((mVec[i].x - mVec[j].x),2) + pow((mVec[i].y - mVec[j].y),2)));
    }
  }
  for (int i = 0; i < l.size(); i++)
  {
    if((l[i]< l_mean*1.5) && (l[i] > l_mean*0.5))
    {
      l_sum = l_sum+l[i]; 
      bin_cntr++;
    }
  }
  if(bin_cntr<1) return;
  l_mean = l_sum/bin_cntr;
  cout<<l_mean<<endl;
 qLocPsum(0) = 0; qLocPsum(1) = 0;
  pRed(0) = 0; pRed2(0) = 0; pRed(1) = 0; pRed(2) = 0;
  qLocP(0) = 0; qLocP(1)=0; eP(0) = 0;

  for(int i = 0; i<length; i++)
  {
        pRed(i,0) = qLoc(0) + ((mVec[i].x - x_pix)/l_mean)*cos(qLoc(3)) - ((mVec[i].y - y_pix)/l_mean)*sin(qLoc(3));
        pRed(i,1) = qLoc(1) + ((mVec[i].x - x_pix)/l_mean)*sin(qLoc(3)) + ((mVec[i].y - y_pix)/l_mean)*cos(qLoc(3));
        pRed2(i,0) = round(pRed(i,0)); 
        pRed2(i,1) = round(pRed(i,1)); 

        eP(i) = sqrt(pow((pRed(i,0) - pRed2(i,0)),2)+pow((pRed(i,1) - pRed2(i,1)),2)); 
        qLocP(i,0) = pRed2(i,0) - ((mVec[i].x - x_pix)/l_mean)*cos(qLoc(3)) + ((mVec[i].y - y_pix)/l_mean)*sin(qLoc(3));
        qLocP(i,1) = pRed2(i,1) - ((mVec[i].x - x_pix)/l_mean)*sin(qLoc(3)) - ((mVec[i].y - y_pix)/l_mean)*cos(qLoc(3));        

        qLocPsum(0) = qLocPsum(0) + (1/eP(i)) * qLocP(i,0);
        qLocPsum(1) = qLocPsum(1) + (1/eP(i)) * qLocP(i,1);
        ePsum = ePsum + (1/eP(i));
        if(i!= length - 1)
        {
        pRed.push_back(zeros);
        pRed2.push_back(zeros);
        qLocP.push_back(zeros); 
        eP.push_back(zero1);
        expMat.push_back(zeros);
         }
     }
    vector<double> angvar(1,1); 
    double dx = 0;
    double dy = 0;

    for(int i=0; i < pRed2.rows; i++)
    {
      for (int j=0;j < pRed2.rows; j++)
      {
        if (pRed2(i,0) == pRed2(j,0) + 1 && pRed2(i,1) == pRed2(j,1)){
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
    if(ePsum==0) return;
    qLoc(0) = qLocPsum(0) / ePsum;
    qLoc(1) = qLocPsum(1) / ePsum;
    qLoc(2) = 322.5806/l_mean;
    if(angvar.size() != 0) qLoc(3) = median(angvar)*pi/180; 
    Mat_<double> num = pm*h.t();
    Mat_<double> denom = h*pm*h.t() + r;
    Mat_<double> ka = num*denom.inv(DECOMP_LU);

    for(int i = 0; i < qLocP.rows; i++)
    {
      zmat[0] = zmat[0] + qLocP(i,0)/qLocP.rows;
      zmat[1] = zmat[1] + qLocP(i,1)/qLocP.rows;
    }

    zmat[2] = 322.5806/l_mean;
    zmat[3] = qLoc(3);
     
    Mat_<double> z(1,4,zmat);
    z = z.t();
		Mat_<double> xh = xm+ka*(z-h*xm);
    if(xh(0) != xh(0)) return;
    Mat_<double> p = (eye-ka*h)*pm;
    xm = phi*xh;
    pm = phi*p*phi.t()+q;
    
    qLoc(0) = xh(0) + xh(1)*1/frame_rate;
    qLoc(1) = xh(2) + xh(3)*1/frame_rate;
    qLoc(2) = xh(4) + xh(5)*1/frame_rate;
    qLoc(3) = xh(6) + xh(7)*1/frame_rate;
    //cout<<qLoc<<endl; 
    //x.push_back(Mat_<double>(1,3, temp));
    //double duh[3] = {xh(1), xh(3), xh(5)};
    //v.push_back(Mat_<double>(1,3,duh));
    logfile << qLoc(0)*.3048 << "," << qLoc(1)*0.3048 << "," << qLoc(2) << ","
    <<qLoc(3)<<"," << xh(1) << "," << xh(3) << "," << xh(5) << endl;
    
}
int main(int argc, char** argv)
{
   time_previous = clock();
  VideoCapture cap("quadvid.avi");
    Mat imgThresholded;
    Mat imgHSV;
    Mat lowerRed;
    Mat upperRed;
//    Mat imgOriginal;
//  const string filename = argv[1];
  logfile.open("data.csv");
  if (!cap.isOpened())
  { 
    cout<<"Cannot load camera"<< endl;
    return -1;
  }
xm = xm.t();
  cap.set(CV_CAP_PROP_FPS, 30); //setting capture fps to 60
  cap.set(CV_CAP_PROP_FRAME_WIDTH,x_res);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, y_res);
  createWindows();
  
  //grabbing first frame of video
  cout << "Starting Stream" << endl;
  while (true)
  {
    bool bSuccess = cap.read(imgOriginal);
    
       if (!bSuccess) //break loop if not successful
    {
      cout << "Cannot read a frame from video stream" << endl;
      logfile.close();
      break;
    }
    cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);
  //  upperRed = imgHSV.clone();
    inRange(imgHSV, Scalar(iLowR, iLowG, iLowB), Scalar(iHighR, iHighG, iHighB), lowerRed);
    inRange(imgHSV, Scalar(iLowR2, iLowB), Scalar(iHighR2, iHighG, iHighB),upperRed);
    addWeighted(lowerRed,1.0,upperRed,1.0,0.0,imgThresholded);
    morphOps(imgThresholded);
//    GaussianBlur(imgThresholded, imgThresholded, Size(9,9),2,2);
    vector<Vec3f> circles;
    vector<marker> markerVec;
    HoughCircles(imgThresholded, circles, CV_HOUGH_GRADIENT, 1, 100, 200,2,minRadius,maxRadius);
    if(circles.size() != 0){
    for(size_t current_circle = 0; current_circle < circles.size(); current_circle++)
    {
    marker marktemp;

 	  marktemp.x = circles[current_circle][0];
	  marktemp.y = circles[current_circle][1];
          markerVec.push_back(marktemp);
	Point center(round(circles[current_circle][0]),round(circles[current_circle][1]));
    int radius = round(circles[current_circle][2]);
    circle(imgOriginal, center, radius, Scalar(0,255,0),5);

    }
    matLabCode(markerVec);
    }
//    video.write(imgOriginal);
//    imshow("Thresholded Image", imgThresholded); //show thresholded image
    imshow("Original",imgOriginal);//show original image
    //frame_rate = CLOCKS_PER_SEC/((float)(clock() - time_previous));
    time_previous = clock();
    if(waitKey(30) == 27) //wait for esc key press for 30ms, if esc key is pressed, break loop
    {
      cout << "esc key pressed by user" << endl;
      logfile.close();
      break;

    }

  }

 return 0;
}
