/*
Project :- Real-Time 2D Object Recognition
@ Author:- Hussain Kanchwala, Abdulaziz Suria
@ Date  :- Start: - 02/19/24 End:- 02/25/24
This file takes in an image and segmentes the image and finds the features
*/
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../header_files/objfun.h"
using namespace std;
using namespace cv;

int main()
{
    // get the path of the image
    string path;
    cout << "Please enter the image path " << endl;
    getline(cin, path);

    // Read the image path into a variable
    Mat frame = imread(path);
    // Variables initialized to hold different stages of object recognition
    Mat currentframe;
    Mat dst;
    Mat gray;
    Mat segment_output;
    Mat regionmap;

    // Minimum area of segmented region
    //int top_n = 4;
    int min_area;
    cout<<"Enter minimum area";
    cin>>min_area;
    vector<int> major_regions;

    // Default color components for the segmentation colors
    vector<Vec3b> color_components;
    create_color_vector(color_components);

    // Thresholding
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    int thresh_val = get_otsu_thresh(gray);
    thresh_custom(thresh_val, gray, currentframe);
    namedWindow("Thresholded", WINDOW_NORMAL); // WINDOW_NORMAL allows the window to be resizable
    resizeWindow("Thresholded", 640, 480);
    resize(currentframe, currentframe, Size(640, 480));
    imshow("Thresholded", currentframe);
    waitKey(0);

    // Cleanup
    cleanup(currentframe, dst);
    namedWindow("Cleanedup", WINDOW_NORMAL); // WINDOW_NORMAL allows the window to be resizable
    resizeWindow("Cleanedup", 640, 480);
    resize(dst,dst, Size(640, 480));
    imshow("Cleanedup", dst);
    waitKey(0);

    // Image Segmentation
    int biggest = segment_image(dst, regionmap, color_components, segment_output, min_area,major_regions);
    namedWindow("segment", WINDOW_NORMAL); // WINDOW_NORMAL allows the window to be resizable
    resizeWindow("segment", 640, 480);
    resize(segment_output,segment_output, Size(640, 480));
    imshow("segment",segment_output);


    // Feature Vector Generation
    vector<float> featurevector = computeFeatures(regionmap, biggest, segment_output);
    namedWindow("Feature", WINDOW_NORMAL); // WINDOW_NORMAL allows the window to be resizable
    resizeWindow("Feature", 640, 480);
    resize(segment_output,segment_output, Size(640, 480));
    imshow("Feature", segment_output);
    waitKey(0);

    return 0;
}