#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../header_files/objfun.h"
using namespace std;
using namespace cv;


// Given the threshold generates the thresholded image from gray scale image
void thresh_custom(int th,Mat frame,Mat& currentframe)
{
    Mat dst = Mat::zeros(frame.size(), frame.type());
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)                                                      
        {
            if (frame.at<uchar>(i,j) < th)
            {
                dst.at<uchar>(i,j) = 255;
            }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            else
            {
                dst.at<uchar>(i,j) = 0;
            }
        }
    }
    currentframe = dst.clone();
}

// Implementing OTSU dynamic thresholding
int thresh(Mat frame)
{
    int histogram[256] = {0};
    GaussianBlur(frame, frame, Size(5, 5), 0); // Smooth the gray scale
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            histogram[frame.at<uchar>(i, j)]++;
        }
    }
    // Implement OTSU threshold calculation
    // Calculate the total number of pixels
    int total = frame.rows * frame.cols;

    float sum = 0;
    for (int i = 0; i < 256; i++)
    {
        sum += i * histogram[i];
    }

    float sumB = 0, wB = 0, wF = 0, varMax = 0;
    float th = 0;
    for (int t = 0; t < 256; t++) 
    {
        wB += histogram[t]; // Weight Background
        if (wB == 0)
            continue;

        wF = total - wB; // Weight Foreground
        if (wF == 0)
            break;

        sumB += (float)(t * histogram[t]);

        float mB = sumB / wB;         // Mean Background
        float mF = (sum - sumB) / wF; // Mean Foreground

        // Calculate Between Class Variance
        float varBetween = wB * wF * (mB - mF) * (mB - mF);

        // Check if new maximum found
        if (varBetween > varMax)
        {
            varMax = varBetween;
            th = t;
        }
    }
    cout << th << endl;
    return th;
}


// Non custom implementation of Erosion annd Dialation
void cleanup(Mat frame, Mat &currentframe)
{

    Mat d_kernel = getStructuringElement(MORPH_RECT, Size(8, 8));
    Mat e_kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat dst;
    dilate(frame, dst, d_kernel, Point(-1, -1), 1);
    erode(dst, currentframe, e_kernel, Point(-1, -1), 1);
}

// Dialte Custom Implementation
void dilate_custom(Mat &src, Mat &dst, Mat &d_kernel)
{
    dst = src.clone();
    int row_mid = d_kernel.rows / 2;
    int col_mid = d_kernel.cols / 2;

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            uchar max_val = 0;
            for (int ki = 0; ki < row_mid; ki++)
            {
                for (int kj = 0; kj < col_mid; kj++)
                {
                    int x = i + ki - row_mid;
                    int y = j + kj - col_mid;
                    if (x >= 0 && x < src.rows && y >= 0 && y < src.cols)
                    {
                        max_val = max(max_val, src.at<uchar>(x, y));
                    }
                }
            }
            dst.at<uchar>(i, j) = max_val;
        }
    }
}

// Erosion Custom Implementation
void erode_custom(Mat &src, Mat &dst, Mat &e_kernel)
{
    dst = src.clone();
    int row_mid = e_kernel.rows / 2;
    int col_mid = e_kernel.cols / 2;

    // Iterate source image
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            bool erode_pixel = true;
            // Iterate kernel
            for (int ki = 0; ki < row_mid; ki++)
            {
                for (int kj = 0; kj < col_mid; kj++)
                {
                    // Calculate target position
                    int x = i + ki - row_mid;
                    int y = j + kj - col_mid;

                    // Check for erosion condition
                    if (x >= 0 && x < src.rows && y >= 0 && y < src.cols && src.at<uchar>(x, y) == 0)
                    {
                        erode_pixel = false;
                        break;
                    }
                }
                if (!erode_pixel)
                {
                    break;
                }
            }
            // Apply erosion condition
            dst.at<uchar>(i, j) = erode_pixel ? 255 : 0;
        }
    }
}

// 
void cleanup_custom(Mat frame, Mat &currentframe)
{
    Mat d_kernel = Mat::ones(Size(8, 8), CV_8U);
    Mat e_kernel = Mat::ones(Size(3, 3), CV_8U);
    Mat dst;
    dilate_custom(frame, dst, d_kernel);
    erode_custom(dst, currentframe, e_kernel);
}
/*
// Given a Binary Cleaned up Image, do the segmentation.
// In the segmented Image only show the regions that are greater than the user specified value
void segment_image(Mat cleanedup, Mat &segmented_output,const int min_area)
{

    // Use  these two lines to further continue for task 4 in main program itself, I did the visual display
    Mat img_labels, img_stats, centroids;
    Mat dst = Mat::zeros(cleanedup.size(),CV_8UC3);
    int label_count = connectedComponentsWithStats(cleanedup, img_labels, img_stats, centroids);
    vector<Vec3b> colors(label_count);
    colors[0] = Vec3b(0,0,0); // Background Color
    for(int label =1; label<label_count;label++ ){
        colors[label] = Vec3b((rand()&255),(rand()&255),(rand()&255));
    }
    for(int r=0;r<img_labels.rows;r++){
        for(int c=0;c<img_labels.cols;c++){
            int label = img_labels.at<int>(r,c);
            int* stat = img_stats.ptr<int>(label);
            int area = stat[ConnectedComponentsTypes::CC_STAT_AREA];
            if(area>=min_area){
                dst.at<Vec3b>(r,c) = colors[label];
            }
        }    
    }
    segmented_output = dst.clone();
}
*/

// fixed predefined set of colors generate once per program
void create_color_vector(vector<Vec3b> &color_components) {
    // Ensure the vector is empty before adding new colors
    color_components.clear();
    
    // Fixed list of 30 colors at start of the program
    for (int i = 0; i < 30; i++) {
        color_components.push_back(cv::Vec3b(rand() % 256, rand() % 256, rand() % 256));
    }
    // Background color
    color_components[0] = Vec3b(0, 0, 0);
}

// Image Segmentation
void segment_image(Mat frame, const vector<Vec3b> &color_components, Mat &segment_output, const int min_area) {
    // Ensure the output image has the same dimensions as the input, but with 3 channels for color
    segment_output = Mat::zeros(frame.size(), CV_8UC3);

    Mat img_labels, img_stats, centroids;
    int label_count = connectedComponentsWithStats(frame, img_labels, img_stats, centroids, 8, CV_32S);

    // Create a flag array to mark labels that meet the area requirement
    vector<bool> valid_label(label_count, false);

    for (int i = 1; i < label_count; i++) { // Start from 1 to skip background
        int curr_area = img_stats.at<int>(i, CC_STAT_AREA);
        if (curr_area > min_area) {
            valid_label[i] = true;
        }
    }

    // Color valid segments in one pass
    for (int x = 0; x < img_labels.rows; x++) {
        for (int y = 0; y < img_labels.cols; y++) {
            int curr_label = img_labels.at<int>(x, y);
            if (valid_label[curr_label]) {
                segment_output.at<Vec3b>(x, y) = color_components[curr_label % color_components.size()];
            }
        }
    }
}

// 