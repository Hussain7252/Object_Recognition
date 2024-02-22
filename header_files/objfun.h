#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// OTSU binary thresholding
int thresh(Mat frame); // Pass the gray frame as input

// Thresholding the gray scale image
void thresh_custom(int th,Mat frame,Mat& currentframe); // Inputs are threshold, Gray frame and frame to store the thresholded frame

// Thresholded Image cleanup
void cleanup(Mat frame, Mat &currentframe);

// Custom Cleanup
void cleanup_custom(Mat frame, Mat &currentframe);

// Dialation Implementation
void dilate_custom(Mat &src, Mat &dst, Mat &d_kernel);

// Erosion Implementation
void erode_custom(Mat &src, Mat &dst, Mat &e_kernel);

// Create colors for the segmented image
void create_color_vector(vector<Vec3b> &color_components);

// Image Segmentation
int segment_image(Mat frame, Mat &region_map,const vector<Vec3b> &color_components, Mat &segment_output, const int min_area,vector<int> &major_regions);

// Compute Features
vector<float> computeFeatures(const Mat &regionMap, int regionId, const Mat &segmented_img);
