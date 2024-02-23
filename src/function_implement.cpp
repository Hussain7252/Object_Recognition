/*
@ Author:- Hussain Kanchwala, Abdulaziz Suria
@ Date  :- Start: - 02/19/24 End:- 02/25/24
@ Description : A file consisting of all function implementations of the object recognition task. It also has custom implementation of Thresholding and cleanup as well.
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../header_files/objfun.h"
#include <unordered_set>
using namespace std;
using namespace cv;

// Given the threshold generates the thresholded image from gray scale image
void thresh_custom(int th, Mat frame, Mat &currentframe)
{
    Mat dst = Mat::zeros(frame.size(), frame.type());
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            if (frame.at<uchar>(i, j) < th)
            {
                dst.at<uchar>(i, j) = 255;
            }
            else
            {
                dst.at<uchar>(i, j) = 0;
            }
        }
    }
    currentframe = dst.clone();
}

// Implementing OTSU dynamic thresholding
int get_otsu_thresh(Mat frame)
{
    int histogram[256] = {0};
    GaussianBlur(frame, frame, Size(5, 5), 0); // Smooth the gray scale
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            histogram[frame.at<uchar>(i, j)]++;
        }
    }
    // Implement OTSU threshold calculation
    // Calculate the total number of pixels
    int total = frame.rows * frame.cols;

    float sum = 0;
    for (int i = 0; i < 256; i++)
    {
        sum += i * histogram[i];
    }

    float sumB = 0, wB = 0, wF = 0, varMax = 0;
    float th = 0;
    for (int t = 0; t < 256; t++)
    {
        wB += histogram[t]; // Weight Background
        if (wB == 0)
            continue;

        wF = total - wB; // Weight Foreground
        if (wF == 0)
            break;

        sumB += (float)(t * histogram[t]);

        float mB = sumB / wB;         // Mean Background
        float mF = (sum - sumB) / wF; // Mean Foreground

        // Calculate Between Class Variance
        float varBetween = wB * wF * (mB - mF) * (mB - mF);

        // Check if new maximum found
        if (varBetween > varMax)
        {
            varMax = varBetween;
            th = t;
        }
    }
    return th;
}

// Non custom implementation of Erosion annd Dialation
void cleanup(Mat frame, Mat &currentframe)
{

    Mat d_kernel = getStructuringElement(MORPH_RECT, Size(8, 8));
    Mat e_kernel = getStructuringElement(MORPH_RECT, Size(8, 8));
    Mat e_kernel_e = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat d_kernel_e = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat dst;
    dilate(frame, dst, d_kernel, Point(-1, -1), 1);
    erode(dst, dst, e_kernel, Point(-1, -1), 1);
    erode(dst, dst, e_kernel_e, Point(-1, -1), 1);
    dilate(dst, dst, d_kernel_e, Point(-1, -1), 1);
    erode(dst, currentframe, e_kernel_e, Point(-1, -1), 1);
}

// Dialte Custom Implementation
void dilate_custom(Mat &src, Mat &dst, Mat &d_kernel)
{
    dst = src.clone();
    int row_mid = d_kernel.rows / 2;
    int col_mid = d_kernel.cols / 2;

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            uchar max_val = 0;
            for (int ki = 0; ki < row_mid; ki++)
            {
                for (int kj = 0; kj < col_mid; kj++)
                {
                    int x = i + ki - row_mid;
                    int y = j + kj - col_mid;
                    if (x >= 0 && x < src.rows && y >= 0 && y < src.cols)
                    {
                        max_val = max(max_val, src.at<uchar>(x, y));
                    }
                }
            }
            dst.at<uchar>(i, j) = max_val;
        }
    }
}

// Erosion Custom Implementation
void erode_custom(Mat &src, Mat &dst, Mat &e_kernel)
{
    dst = src.clone();
    int row_mid = e_kernel.rows / 2;
    int col_mid = e_kernel.cols / 2;

    // Iterate source image
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            bool erode_pixel = true;
            // Iterate kernel
            for (int ki = 0; ki < row_mid; ki++)
            {
                for (int kj = 0; kj < col_mid; kj++)
                {
                    // Calculate target position
                    int x = i + ki - row_mid;
                    int y = j + kj - col_mid;

                    // Check for erosion condition
                    if (x >= 0 && x < src.rows && y >= 0 && y < src.cols && src.at<uchar>(x, y) == 0)
                    {
                        erode_pixel = false;
                        break;
                    }
                }
                if (!erode_pixel)
                {
                    break;
                }
            }
            // Apply erosion condition
            dst.at<uchar>(i, j) = erode_pixel ? 255 : 0;
        }
    }
}

//
void cleanup_custom(Mat frame, Mat &currentframe)
{
    Mat d_kernel = Mat::ones(Size(8, 8), CV_8U);
    Mat e_kernel = Mat::ones(Size(12, 12), CV_8U);
    Mat dst;
    dilate_custom(frame, dst, d_kernel);
    erode_custom(dst, currentframe, e_kernel);
}
/*
// Given a Binary Cleaned up Image, do the segmentation.
// In the segmented Image only show the regions that are greater than the user specified value
void segment_image(Mat cleanedup, Mat &segmented_output,const int min_area)
{

    // Use  these two lines to further continue for task 4 in main program itself, I did the visual display
    Mat img_labels, img_stats, centroids;
    Mat dst = Mat::zeros(cleanedup.size(),CV_8UC3);
    int label_count = connectedComponentsWithStats(cleanedup, img_labels, img_stats, centroids);
    vector<Vec3b> colors(label_count);
    colors[0] = Vec3b(0,0,0); // Background Color
    for(int label =1; label<label_count;label++ ){
        colors[label] = Vec3b((rand()&255),(rand()&255),(rand()&255));
    }
    for(int r=0;r<img_labels.rows;r++){
        for(int c=0;c<img_labels.cols;c++){
            int label = img_labels.at<int>(r,c);
            int* stat = img_stats.ptr<int>(label);
            int area = stat[ConnectedComponentsTypes::CC_STAT_AREA];
            if(area>=min_area){
                dst.at<Vec3b>(r,c) = colors[label];
            }
        }
    }
    segmented_output = dst.clone();
}
*/

// fixed predefined set of colors generate once per program
void create_color_vector(vector<Vec3b> &color_components)
{
    // Ensure the vector is empty before adding new colors
    color_components.clear();

    // Fixed list of 256 colors at start of the program
    for (int i = 0; i < 256; i++)
    {
        color_components.push_back(cv::Vec3b(rand() % 256, rand() % 256, rand() % 256));
    }
    // Background color
    color_components[0] = Vec3b(0, 0, 0);
}

// Image Segmentation
// Suppose we get the main regions then do the necessary
int segment_image(Mat frame, Mat &img_labels, const vector<Vec3b> &color_components, Mat &segment_output, int top_n, vector<int> &major_regions)
{
    // Ensure the output image has the same dimensions as the input, but with 3 channels for color
    segment_output = Mat::zeros(frame.size(), CV_8UC3);
    Mat img_stats, centroids;
    int label_count = connectedComponentsWithStats(frame, img_labels, img_stats, centroids, 8, CV_32S);

    vector<pair<int, int>> area_map;
    // Biggest Region in frame

    for (int i = 1; i < label_count; i++)
    { // Start from 1 to skip background
        int curr_area = img_stats.at<int>(i, CC_STAT_AREA);
        area_map.push_back(make_pair(curr_area, i));
    }
    sort(area_map.begin(), area_map.end(), greater<>());

    // Get the top n labels within range
    top_n = min(top_n, label_count - 1);

    // Keep the top n values in the area map
    area_map.resize(top_n);

    // Use set for faster retrieval of labels
    unordered_set<int> valid_labels;

    for (const auto &area_label_pair : area_map)
    {
        valid_labels.insert(area_label_pair.second);
    }

    // Color valid segments in one pass
    for (int x = 0; x < img_labels.rows; x++)
    {
        for (int y = 0; y < img_labels.cols; y++)
        {
            int curr_label = img_labels.at<int>(x, y);
            if (valid_labels.find(curr_label) != valid_labels.end())
            {
                segment_output.at<Vec3b>(x, y) = color_components[curr_label % color_components.size()];
            }
        }
    }
    return area_map[0].second;
}

// Feature Vector Generation for the Major Region
vector<float> computeFeatures(const Mat &regionMap, int regionId, const Mat &segmented_img)
{
    vector<float> features;

    // Find pixels belonging to the region
    Mat region = regionMap == regionId;

    // Calculate moments
    Moments m = moments(region, true);

    // Calculate area (m00) and centroid (m10/m00, m01/m00)
    double area = m.m00;
    double centroidX = m.m10 / area;
    double centroidY = m.m01 / area;

    // Calculate the orientation and axis of least central moment
    double a = m.mu20 / area;
    double b = 2 * m.mu11 / area;
    double c = m.mu02 / area;
    double theta = 0.5 * atan2(b, a - c);
    double eccentricity = sqrt(1 - (c / a));

    // Calculate bounding box
    vector<Point> regionPoints;
    findNonZero(region, regionPoints);
    RotatedRect boundingBox = minAreaRect(regionPoints);

    // Calculate percent filled and bounding box ratio
    float percentFilled = area / (boundingBox.size.width * boundingBox.size.height);
    float bboxRatio = max(boundingBox.size.height, boundingBox.size.width) / min(boundingBox.size.width, boundingBox.size.height);

    // HuMoments
    vector<double> humoment(7);
    HuMoments(m, humoment);
    vector<float> scaledhumoment(7);
    for (int j = 0; j < 7; j++)
    {
        // Check for the absolute value to avoid log of zero or negative numbers
        double absValue = abs(humoment[j]);
        if (absValue > numeric_limits<double>::epsilon())
        { // Check if absValue is not too close to zero
            scaledhumoment[j] = -1 * copysign(1.0, humoment[j]) * log10(absValue);
        }
        else
        {
            scaledhumoment[j] = 0; // Assign 0 if the Hu Moment is too close to zero to avoid -inf from log10
        }
        features.push_back(scaledhumoment[j]);
    }

    // Add features to the vector
    features.push_back(percentFilled);
    features.push_back(bboxRatio);
    features.push_back(eccentricity); // Example additional feature

    // bounding box and axis could be added here
    Point2f vertices[4];
    boundingBox.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        line(segmented_img, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2); // Green box with a thickness of 2
    }

    // Draw the axis of least moment central axis
    double length = max(boundingBox.size.width, boundingBox.size.height);                                                    // Length of the line representing the axis
    Point2f p1(centroidX, centroidY);                                                                                        // Centroid
    Point2f p2 = p1 + Point2f(0.5 * static_cast<float>(cos(theta) * length), 0.5 * static_cast<float>(sin(theta) * length)); // Forward direction
    Point2f p3 = p1 - Point2f(0.5 * static_cast<float>(cos(theta) * length), 0.5 * static_cast<float>(sin(theta) * length)); // Backward direction
    line(segmented_img, p1, p2, Scalar(0, 0, 255), 2);                                                                       // Red Line for the axis
    line(segmented_img, p1, p3, Scalar(0, 0, 255), 2);

    // Text Hu Moment
    string text = std::to_string(scaledhumoment[0]);
    putText(segmented_img, text, p1, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0), 1); // displaying the huMoment feature in real time for each contour
    return features;
}

// Scaled euclidean distance
pair<float, bool> compute_euclidean(vector<float> fvec_1, vector<float> fvec_2, float known_threshold, float stdev)
{

    pair<float, bool> result_pair;
    if (fvec_1.size() != fvec_2.size() || fvec_1.empty())
    {
        cout << "Size mismatch between feature vectors" << endl;
        return result_pair;
    }
    float score = 0;
    for (int i = 0; i < fvec_1.size(); i++)
    {
        score += pow((fvec_1[i] - fvec_2[i] / stdev), 2);
    }
    score = sqrt(score);
    result_pair.first = score;
    result_pair.second = score >= known_threshold ? true : false;
    return result_pair;
}

// cosine similarity
pair<float, bool> compute_similarity(vector<float> fvec_1, vector<float> fvec_2, float known_threshold)
{
    pair<float, bool> result_pair;
    if (fvec_1.size() != fvec_2.size() || fvec_1.empty())
    {
        cout << "Size mismatch between feature vectors" << endl;
        return result_pair;
    }

    // Convert vectors to OpenCV Mat
    cv::Mat mat_v1(fvec_1);
    cv::Mat mat_v2(fvec_2);

    // Normalize vectors
    cv::normalize(mat_v1, mat_v1);
    cv::normalize(mat_v2, mat_v2);

    float similarity = mat_v1.dot(mat_v2);
    result_pair.first = similarity;
    result_pair.second = similarity >= known_threshold ? true : false;

    return result_pair;
}
