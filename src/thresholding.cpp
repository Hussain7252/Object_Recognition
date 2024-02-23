/*
Project :- Real-Time 2D Object Recognition
@ Author:- Hussain Kanchwala, Abdulaziz Suria
@ Date  :- Start: - 02/19/24 End:- 02/25/24
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../header_files/objfun.h"
using namespace std;
using namespace cv;

// Turns on your device default camera
void video_turnon()
{
    VideoCapture capdev(0);
    if (!capdev.isOpened())
    {
        cout << "Unable to open video device"
             << "\n";
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
    Mat gray;
    Mat th_frame;
    Mat clean_frame;
    int top_n = 4;
    // cout<<"Please enter the  top segmentation regions to display"<<endl;
    // cin>>min_area;
    vector<Vec3b> color_components;
    create_color_vector(color_components);

    while (true)
    {
        capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty())
        {
            cout << "frame is empty"
                 << "\n";
            break;
        }
        // key press
        int key = waitKey(1);

        //  For segmentation
        Mat segment_output;
        Mat region_map;
        vector<int> major_regions;
        //
        // COnvert to gray
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        // Get the dynamic threshold
        int th = get_otsu_thresh(gray);
        // Use the dynamic threshold to get the thresholded frame
        thresh_custom(th, gray, th_frame);
        // Clean the thresholded frame
        // For cleanup_custom
        cleanup(th_frame, clean_frame);
        // Group the regions
        int biggest_region = segment_image(clean_frame, region_map, color_components, segment_output, top_n, major_regions);
        // Feature Vector for biggest region
        vector<float> featurevector = computeFeatures(region_map, biggest_region, segment_output);
        // Store in CSV on press of N button
        /*
        if (key == 'n' || key == 'N'){
            string lab;
            cout<<"Please enter the  label of the  item"<<endl;
            getline(cin,lab);
            append_image_data_csv(featurefile,lab,featurevector,0);
        }
        */
        // Display the video
        imshow("Video", segment_output);

        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
    }
    capdev.release();
    destroyAllWindows();
}

int main()
{
    video_turnon();
    return 0;
}