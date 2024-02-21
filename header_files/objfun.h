#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// OTSU binary thresholding
void thresh(Mat frame, Mat &currentframe);

// Thresholded Image cleanup
void cleanup(Mat frame, Mat &currentframe);

void segment_image(Mat frame, vector<Vec3b> &color_components, Mat &segment_output);

void create_color_vector(vector<Vec3b> &color_components);