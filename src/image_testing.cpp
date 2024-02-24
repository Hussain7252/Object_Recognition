/*
Project :- Real-Time 2D Object Recognition
@ Author:- Hussain Kanchwala, Abdulaziz Suria
@ Date  :- Start: - 02/19/24 End:- 02/25/24
This file takes in an image and recognizes the object provided in the given image (if present in DB)
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
    imshow("Threshold Image", currentframe);
    waitKey(0);

    // Cleanup
    cleanup(currentframe, dst);
    imshow("Cleaned Image", dst);
    waitKey(0);

    // Image Segmentation
    int biggest = segment_image(dst, regionmap, color_components, segment_output, min_area,major_regions);

    // Feature Vector Generation
    vector<float> featurevector = computeFeatures(regionmap, biggest, segment_output);
    imshow("Segmented Image", segment_output);
    waitKey(0);

    return 0;
}