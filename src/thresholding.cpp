/*
Project :- Real-Time 2D Object Recognition
@ Author:- Hussain Kanchwala, Abdulaziz Suria
@ Date  :- Start: - 02/19/24 End:- 02/25/24
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../header_files/objfun.h"
#include "../header_files/csv_util.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem> 
namespace fs = std::filesystem; 
using namespace std;
using namespace cv;

// Turns on your device default camera
int video_turnon()
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
    // Different Mat variables
    Mat frame;
    Mat gray;
    Mat th_frame;
    Mat clean_frame;
    int top_n = 4;
    // cout<<"Please enter the  top segmentation regions to display"<<endl;
    // cin>>min_area;

    vector<Vec3b> color_components;
    create_color_vector(color_components);
    // Filename to store feature vectors
    string file_path;
    cout<<"Please enter the file name where features have to be stored and retrieved"<<endl;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    getline(cin,file_path);
    // if (!file_path.ends_with(".csv")){
    //     cout<<"Enter valid feature.csv file"<<endl;
    //     return -1;
    //}
    char* filepath = new char[file_path.length() + 1];
    strcpy(filepath, file_path.c_str());


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
        if (key == 'n' || key == 'N'){
            string lab;
            cout<<"Please enter the  label of the  item"<<endl;
            getline(cin,lab);
            char* name = new char[lab.length()+1];
            strcpy(name, lab.c_str());
            append_image_data_csv(filepath,name,featurevector,0);
        }


        // Display the video
        imshow("Video", segment_output);

        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
    }
    capdev.release();
    destroyAllWindows();
    return 0;
}

int main()
{
    video_turnon();
    return 0;
}