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

// Turns on your device default camera
void video_turnon() {
    VideoCapture capdev(0);
    if (!capdev.isOpened()) {
        cout<<"Unable to open video device"<<"\n";
    }

    int frameWidth = capdev.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = capdev.get(CAP_PROP_FRAME_HEIGHT);
    double fps = capdev.get(CAP_PROP_FPS);
    int totalFrames = capdev.get(CAP_PROP_FRAME_COUNT);
    std::cout << "Video Details:" << std::endl;
    std::cout << "Frame Width: " << frameWidth << std::endl;
    std::cout << "Frame Height: " << frameHeight << std::endl;
    std::cout << "Frames Per Second (fps): " << fps << std::endl;
    std::cout << "Total Number of Frames: " << totalFrames << std::endl;
    namedWindow("Video", 1); // identifies a window
    Mat frame;
    Mat th_frame;
    Mat clean;
    while (true) {
        capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            cout<<"frame is empty"<<"\n";
            break;
        }
        thresh(frame,th_frame);
        cleanup(th_frame,clean);
        imshow("Video",clean);
        int key = waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q'){
            break;
        }
    }
    capdev.release();
    destroyAllWindows();
}

int main() {
    video_turnon();
    return 0;
}