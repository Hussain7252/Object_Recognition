#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// OTSU binary thresholding
void thresh(Mat frame, Mat &currentframe);

//Thresholded Image cleanup
void cleanup(Mat frame, Mat &currentframe);