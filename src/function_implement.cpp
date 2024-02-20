#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../header_files/objfun.h"
using namespace std;
using namespace cv;

void thresh(Mat frame, Mat &currentframe){
    int histogram[256]={0};
    cvtColor(frame,frame,COLOR_BGR2GRAY); // Convert to gray scale
    GaussianBlur(frame,frame,Size(5,5),0); // Smooth the gray scale
    for(int i=0;i<frame.rows;i++){
        for(int j=0;j<frame.cols;j++){
            histogram[frame.at<uchar>(i,j)]++;
        }
    }
    // Implement OTSU threshold calculation
        // Calculate the total number of pixels
    int total = frame.rows * frame.cols;

    float sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += i * histogram[i];
    }

    float sumB = 0, wB = 0, wF = 0, varMax = 0;
    float th = 0;
    for (int t = 0; t < 256; t++) {
        wB += histogram[t];               // Weight Background
        if (wB == 0) continue;

        wF = total - wB;                   // Weight Foreground
        if (wF == 0) break;

        sumB += (float)(t * histogram[t]);

        float mB = sumB / wB;              // Mean Background
        float mF = (sum - sumB) / wF;      // Mean Foreground

        // Calculate Between Class Variance
        float varBetween = wB * wF * (mB - mF) * (mB - mF);

        // Check if new maximum found
        if (varBetween > varMax) {
            varMax = varBetween;
            th = t;
        }
    }
    cout<<th<<endl;
    // Apply the threshold
    threshold(frame, currentframe, th, 255, THRESH_BINARY_INV);
}

void cleanup(Mat frame, Mat &currentframe){

    Mat d_kernel = getStructuringElement(MORPH_CROSS,Size(4,4));
    Mat e_kernel = getStructuringElement(MORPH_RECT,Size(5,5));
    Mat dst;
    dilate(frame,dst,d_kernel,Point(-1,-1),1);
    erode(dst,currentframe,e_kernel,Point(-1,-1),1);
}

