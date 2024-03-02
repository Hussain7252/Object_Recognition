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
#include <algorithm>
namespace fs = std::filesystem; 
using namespace std;
using namespace cv;

// Turns on your device default camera
int video_turnon()
{
    string ip = "http://10.0.0.232:8080/video";
    VideoCapture capdev(ip);
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
    
    //Select the Feature Vector Model
    dnn::Net net;
    int classifier_mode;
    cout << "Please select the classifier mode: " << endl;
    cout << "1 - Moment based\n2 - DNN based\n";
    cin >> classifier_mode;
    if (classifier_mode != 1 && classifier_mode != 2)
    {
        cout << "Invalid mode selected" << endl;
        return (0);
    }
    if (classifier_mode == 2)
    {
        // read the network
        net = dnn::readNet("../src/or2d-normmodel-007.onnx");
        cout<<"Network read successfully"<<endl;
    }
    
    // Different Mat variables
    Mat frame;
    Mat gray;
    Mat th_frame;
    Mat clean_frame;
    float threshold;
    if(classifier_mode==1){
        threshold = 1;
    }else{
        threshold = 0.3;
    }
    int min_area;
    cout<<"Please enter the min area to segment"<<endl;
    cin>>min_area;

    vector<Vec3b> color_components;
    create_color_vector(color_components);
    // Filename to store feature vectors
    string file_path;
    cout<<"Please enter the file name where features have to be stored and retrieved"<<endl;
    cout<<"Make sure to enter correct filepath for DNN and Nearest Neighbout Feature Vector"<<endl;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    getline(cin,file_path);
    // if (!file_path.ends_with(".csv")){
    //     cout<<"Enter valid feature.csv file"<<endl;
    //     return -1;
    //}
    char* filepath = new char[file_path.length() + 1];
    strcpy(filepath, file_path.c_str());
    //Declare a confusion matrix to update globally
    //Key is the true value and value is the map of what the predictions can be 
    map<string,map<string, int>> confusionMatrix;
    map<string, int> labelToIndex;
    vector<string> indexToLabel;

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
        string text;
        //
        // COnvert to gray
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        // Get the dynamic threshold
        int th = get_otsu_thresh(gray);
        // Use the dynamic threshold to get the thresholded frame
        thresh_custom(th, gray, th_frame);
        // Clean the thresholded frame
        // For cleanup_custom
        cleanup_custom(th_frame,clean_frame);
        //cleanup(th_frame, clean_frame);
        // Group the regions
        int biggest_region = segment_image(clean_frame, region_map, color_components, segment_output, min_area,major_regions);
        //Creation of Feature vector based on classifiermode
        // For Nearest Neighbour make a feature vector for biggest region
        vector<float> featurevector;
        if (classifier_mode==1){
            featurevector = computeFeatures(region_map, biggest_region, segment_output);
        }
        else{
            Mat emb;
            //get bounding box around the main object in file
            //Rect bounding_box;
            //Mat region = region_map == biggest_region;
            // Calculate bounding box
            //vector<Point> regionPoints;
            //findNonZero(region, regionPoints);
            // Calculate the bounding box for the points of the largest region
            //if (!regionPoints.empty()) {
            //   bounding_box = boundingRect(regionPoints);
            cv::Rect bbox( 0, 0, clean_frame.cols, clean_frame.rows );
            getEmbedding(clean_frame,emb,bbox,net,0);
            featurevector.assign((float*)emb.datastart, (float*)emb.dataend);
            //}
        }
        // Calculate Error
        if(!(fs::exists(filepath))|| fs::is_empty(filepath)){
            cout<<"Nothing to compare with making a file system"<<endl;
            key = 'N';
            text = "Unknown";
            // Position for the text (top-right corner)
            int fontFace = FONT_HERSHEY_SIMPLEX;
            double fontScale = 1;
            int thickness = 2;
            int baseline=0;
            Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
            Point textOrg(segment_output.cols - textSize.width - 10, textSize.height + 10);
            // Add the text to the image
            putText(segment_output, text, textOrg, fontFace, fontScale, Scalar(0,0,255), thickness);
        }
        else{ //extract the data from file and store it into a database
            vector<vector<float>> database;
            vector<char *> objectname;
            read_image_data_csv(filepath,objectname,database,0);
            vector<float> error;
            if(classifier_mode == 2){
                error = compute_similarity(featurevector,database);
            }else{
                error = distanceMetric(featurevector,database);
            }
            auto minIt = min_element(error.begin(),error.end());
            int minIndex = distance(error.begin(),minIt);
            float e = *minIt;
            cout<<"The error is "<<e<<endl;
            if(e<threshold){
                text = string(objectname[minIndex]);
                int fontFace = FONT_HERSHEY_SIMPLEX;
                double fontScale = 1;
                int thickness = 2;
                int baseline=0;
                Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
                Point textOrg(segment_output.cols - textSize.width - 10, textSize.height + 10);
                // Add the text to the image
                putText(segment_output, text, textOrg, fontFace, fontScale, Scalar(0,0,255), thickness);
            }else{
                cout<<"This object is not found in databse"<<endl;
                key='N';
                text = "Unknown";
                // Position for the text (top-right corner)
                int fontFace = FONT_HERSHEY_SIMPLEX;
                double fontScale = 1;
                int thickness = 2;
                int baseline=0;
                Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
                Point textOrg(segment_output.cols - textSize.width - 10, textSize.height + 10);
                // Add the text to the image
                putText(segment_output, text, textOrg, fontFace, fontScale, Scalar(0,0,255), thickness);
            }
        }


        // Make the confusion matrix
        if(key == 'C' || key=='c'){
            string truelabel;
            cout<<"Enter the true label";
            getline(cin,truelabel);
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            // Make a function that updates the confusion matrix 
            // Declare the confusion matrix globally
            updateConfusionMatrix(truelabel,text,confusionMatrix,labelToIndex,indexToLabel);
        }
        // See the confusion matrix
        if(key == 'S' || key =='s'){
            cout << "Confusion Matrix:\n\n";
            cout << "True Label ---> Predicted Label: Count\n";
            cout << "---------------------------------------\n";
            string tl;
            string pl;
            for (int i = 0; i < indexToLabel.size(); i++) {
                tl = indexToLabel[i];
                for (int j = 0; j < indexToLabel.size(); j++) {
                    pl = indexToLabel[j];
                    // Accessing the count for the current trueLabel-predictedLabel pair
                    int count = confusionMatrix[tl][pl];
                    // Printing only the mappings that have a non-zero count
                    cout << tl << " ---> " << pl << ": " << count << endl;
                }
            }
            key = 'q';
        }

        // Store in CSV on press of N button
        if (key == 'n' || key == 'N'){
            string lab;
            cout<<"Please enter the  label of the  item"<<endl;
            getline(cin,lab);
            char* name = new char[lab.length()+1];
            strcpy(name, lab.c_str());
            append_image_data_csv(filepath,name,featurevector,0);
        }

        // Save the Classified frame if user wants
        if(key == 'x' || key == 'X'){
            // Define the folder path and file name
            string folderPath = "/home/hussain/computer_vision/CourseWork/Project3/report_pictures"; // Make sure this directory exis
            cout<<"Enter image name"<<endl;
            string fileName;
            getline(cin,fileName);
            string fullPath = folderPath + fileName;
            imwrite(fullPath,segment_output);
        }
        namedWindow("Video", WINDOW_NORMAL); // WINDOW_NORMAL allows the window to be resizable
        resizeWindow("Video", 640, 480);
        Mat resizedFrame;
        resize(segment_output, resizedFrame, Size(640, 480));

        // Display the video
        imshow("Video", resizedFrame);
        // Stop the video when press ESC or Q
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