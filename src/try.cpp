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

int main()
{
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
    /*
     string path;
     cout << "Please enter the image path " << endl;
     getline(cin, path);
     Mat src = imread(path);
     Mat gray;
     Mat dst = Mat::zeros(src.size(), CV_8UC3);
     cvtColor(src, gray, cv::COLOR_BGR2GRAY);     // converting to grayscale
     GaussianBlur(gray, gray, cv::Size(5, 5), 1); // applying blur
     Mat cannyOut;
     Canny(gray, cannyOut, 50, 200); // applying canny edge detector
     vector<vector<Point>> contour;
     // finding the contour points of different regions
     cv::findContours(cannyOut, contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
     // vector<Point> cont = contour[0];

     cv::drawContours(dst, contour, 1, cv::Scalar(0, 255, 0), 2);
     imshow("cont", dst);
     waitKey(0);

     return 0;
     */
    vector<float> v1 = {23.4, 45.7, 89.3};
    vector<float> v2{34.2, 0.4, 99.8};
    pair<float, bool> result(compute_euclidean(v1, v2, 100.0));
    cout << result.first << " bool : " << result.second << endl;
    result = compute_similarity(v1, v2, 100.0);
    cout << result.first << " bool : " << result.second << endl;
}
/*

int segment_image(Mat frame, Mat &region_map, const vector<Vec3b> &color_components, Mat &segment_output, const int min_area, vector<int> &major_regions);


// Suppose we get the main regions then do the necessary
int segment_image(Mat frame, Mat &img_labels, const vector<Vec3b> &color_components, Mat &segment_output, const int min_area, vector<int> &major_regions)
{
    // Ensure the output image has the same dimensions as the input, but with 3 channels for color
    segment_output = Mat::zeros(frame.size(), CV_8UC3);
    Mat img_stats, centroids;
    int label_count = connectedComponentsWithStats(frame, img_labels, img_stats, centroids, 8, CV_32S);

    // Biggest Region in frame
    int biggest_region;
    int min_ar = INT_MIN;

    // Create a flag array to mark labels that meet the area requirement
    vector<bool> valid_label(label_count, false);
    for (int i = 1; i < label_count; i++)
    { // Start from 1 to skip background
        int curr_area = img_stats.at<int>(i, CC_STAT_AREA);
        if (curr_area > min_area)
        {
            valid_label[i] = true;
            major_regions.push_back(i);
            if (curr_area > min_ar)
            {
                biggest_region = i;
                min_ar = curr_area;
            }
        }
    }

    // Color valid segments in one pass
    for (int x = 0; x < img_labels.rows; x++)
    {
        for (int y = 0; y < img_labels.cols; y++)
        {
            int curr_label = img_labels.at<int>(x, y);
            if (valid_label[curr_label])
            {
                segment_output.at<Vec3b>(x, y) = color_components[curr_label % color_components.size()];
            }
        }
    }
    return biggest_region;
}

*/