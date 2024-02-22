/*
Project :- Real-Time 2D Object Recognition
@ Author:- Hussain Kanchwala
@ Date  :- Start: - 02/19/24 End:- 02/25/24
*/
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../header_files/objfun.h"
using namespace std;
using namespace cv;

int main(){
    /*
    string path;
    cout<<"Please enter the image path "<<endl;
    getline(cin, path);
    Mat frame = imread(path);
    Mat currentframe;
    Mat dst;
    Mat gray;
    Mat segment_output;
    int minarea;
    cout<<"Enter min segmentation area "<<endl;
    cin>>minarea;
    vector<Vec3b> color_components;
    create_color_vector(color_components);
    
    // Thresholding
    cvtColor(frame,gray,COLOR_BGR2GRAY);
    int th = thresh(gray);
    thresh_custom(th,gray,currentframe);
    imshow("img", currentframe);
    waitKey(0);

    // Cleanup
    cleanup(currentframe, dst);
    imshow("clean", dst);
    waitKey(0);

    // Image Segmentation
    segment_image(dst,color_components, segment_output, minarea);
    imshow("segment", segment_output);
    waitKey(0);
    */
    string path;
    cout<<"Please enter the image path "<<endl;
    getline(cin, path);
    Mat src = imread(path);
    Mat gray;
    Mat dst = Mat::zeros(src.size(),CV_8UC3);
    cvtColor( src, gray, cv::COLOR_BGR2GRAY ); //converting to grayscale
    GaussianBlur( gray, gray, cv::Size(5,5),1); //applying blur 
    Mat cannyOut;
    Canny( gray, cannyOut, 50, 200 ); //applying canny edge detector
    vector<vector<Point>> contour;
    //finding the contour points of different regions
    cv::findContours( cannyOut, contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    //vector<Point> cont = contour[0];

    cv::drawContours(dst, contour, 1, cv::Scalar(0, 255, 0), 2);
    imshow("cont",dst);
    waitKey(0);

    return 0;
}