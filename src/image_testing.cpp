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


    return 0;
}