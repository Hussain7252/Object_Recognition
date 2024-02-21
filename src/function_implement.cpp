#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../header_files/objfun.h"
using namespace std;
using namespace cv;

void thresh_custom(Mat frame, Mat &currentframe)
{
    cvtColor(frame, currentframe, COLOR_BGR2GRAY); // Convert to gray scale
    for (int i = 0; i < currentframe.rows; i++)
    {
        uchar *dst_row = currentframe.ptr<uchar>(i);
        for (int j = 0; j < currentframe.cols; j++)
        {
            if (dst_row[j] < 100)
            {
                dst_row[j] = 255;
            }
            else
            {
                dst_row[j] = 0;
            }
        }
    }
}
void thresh(Mat frame, Mat &currentframe)
{
    int histogram[256] = {0};
    cvtColor(frame, frame, COLOR_BGR2GRAY);    // Convert to gray scale
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
    // Apply the threshold
    threshold(frame, currentframe, th, 255, THRESH_BINARY_INV);
}

void cleanup(Mat frame, Mat &currentframe)
{

    Mat d_kernel = getStructuringElement(MORPH_CROSS, Size(15, 15));
    Mat e_kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat dst;
    dilate(frame, dst, d_kernel, Point(-1, -1), 1);
    erode(dst, currentframe, e_kernel, Point(-1, -1), 1);
}

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

void cleanup_custom(Mat frame, Mat &currentframe)
{
    Mat d_kernel = Mat::ones(Size(16, 16), CV_8U);
    Mat e_kernel = Mat::ones(Size(3, 3), CV_8U);
    Mat dst;
    dilate_custom(frame, dst, d_kernel);
    erode_custom(dst, currentframe, e_kernel);
}

// fixed predefined set of colors generate once per program
void create_color_vector(vector<Vec3b> &color_components)
{
    // fixed list of 30 colors at start of the program
    for (int i = 0; i < 30; i++)
    {
        color_components.push_back(cv::Vec3b(rand() % 256, rand() % 256, rand() % 256));
    }
    // background color
    color_components[0] = Vec3b(0, 0, 0);
}

void segment_image(Mat frame, vector<Vec3b> &color_components, Mat &segment_output)
{

    // Use  these two lines to further continue for task 4 in main program itself, I did the visual display
    Mat img_labels, img_stats, centroids;
    int label_count = connectedComponentsWithStats(frame, img_labels, img_stats, centroids);
    int min_area = 400;
    for (int i = 1; i < label_count; i++)
    {
        int curr_area = img_stats.at<int>(i, CC_STAT_AREA);
        std::cout << curr_area << endl;
        if (curr_area > min_area)
        {
            for (int x = 0; x < img_labels.rows; x++)
            {
                for (int y = 0; y < img_labels.cols; y++)
                {
                    int curr_label = img_labels.at<int>(x, y);
                    if (curr_label == i)
                    {
                        segment_output.at<Vec3b>(x, y) = color_components[curr_label];
                    }
                }
            }
        }
    }
}