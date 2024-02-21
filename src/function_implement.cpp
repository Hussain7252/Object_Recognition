#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../header_files/objfun.h"
using namespace std;
using namespace cv;

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

    for (int i = 0; i < img_labels.rows; i++)
    {
        for (int j = 0; j < img_labels.cols; j++)
        {
            int curr_label = img_labels.at<int>(i, j);
            if (curr_label > 0 && curr_label <= label_count)
            {
                segment_output.at<Vec3b>(i, j) = color_components[curr_label];
            }
        }
    }
}