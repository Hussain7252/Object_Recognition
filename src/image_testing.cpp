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

int main(){
    string path;
    getline(cin,path);
    Mat frame=imread(path);
    Mat currentframe;
    Mat dst;
    thresh(frame,currentframe);
    imshow("img",currentframe);
    waitKey(0);

    cleanup(currentframe,dst);
    imshow("clean",dst);
    waitKey(0);

    return 0;
}