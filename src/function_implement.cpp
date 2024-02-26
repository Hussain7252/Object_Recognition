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

int segment_image(Mat frame, Mat &img_labels, const vector<Vec3b> &color_components, Mat &segment_output, const int min_area, vector<int> &major_regions)
{
    // Ensure the output image has the same dimensions as the input, but with 3 channels for color
    segment_output = Mat::zeros(frame.size(), CV_8UC3);
    Mat img_stats, centroids;
    int label_count = connectedComponentsWithStats(frame, img_labels, img_stats, centroids, 8, CV_32S);

    // Biggest Region in frame
    int biggest_region;
    int min_ar = INT_MIN;

    // Create a flag array to mark labels that meet the area requirement
    vector<bool> valid_label(label_count, false);
    for (int i = 1; i < label_count; i++)
    { // Start from 1 to skip background
        int curr_area = img_stats.at<int>(i, CC_STAT_AREA);
        if (curr_area > min_area)
        {
            valid_label[i] = true;
            major_regions.push_back(i);
            if (curr_area > min_ar)
            {
                biggest_region = i;
                min_ar = curr_area;
            }
        }
    }

    // Color valid segments in one pass
    for (int x = 0; x < img_labels.rows; x++)
    {
        for (int y = 0; y < img_labels.cols; y++)
        {
            int curr_label = img_labels.at<int>(x, y);
            if (valid_label[curr_label])
            {
                segment_output.at<Vec3b>(x, y) = color_components[curr_label % color_components.size()];
            }
        }
    }
    return biggest_region;
}
// Image Segmentation
// Suppose we get the main regions then do the necessary
/*
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
*/
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

    // Calculate bounding box
    vector<Point> regionPoints;
    findNonZero(region, regionPoints);
    RotatedRect boundingBox = minAreaRect(regionPoints);

    // Calculate percent filled and bounding box ratio
    float percentFilled = area / (boundingBox.size.width * boundingBox.size.height);
    float bboxRatio = min(boundingBox.size.height, boundingBox.size.width) / max(boundingBox.size.width, boundingBox.size.height);

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
            if((-1 * copysign(1.0, humoment[j]) * log10(absValue))>0){
                scaledhumoment[j] = -1 * copysign(1.0, humoment[j]) * log10(absValue);
            }else{
                scaledhumoment[j] = copysign(1.0, humoment[j]) * log10(absValue);
            }
        }
        else
        {
            scaledhumoment[j] = 0; // Assign 0 if the Hu Moment is too close to zero to avoid -inf from log10
        }
    }
    for(int k=0;k<5;k++){
        features.push_back(scaledhumoment[k]);
    }

    // Add features to the vector
    features.push_back(percentFilled);
    features.push_back(bboxRatio);

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

vector<float> calculateStandardDeviations(vector<vector<float>> database) {
    if (database.empty()) return {};
    
    size_t numFeatures = database[0].size();
    std::vector<float> means(numFeatures, 0.0f);
    std::vector<float> stdDevs(numFeatures, 0.0f);

    // Calculate means
    for (const auto& entry : database) {
        for (size_t i = 0; i < numFeatures; ++i) {
            means[i] += entry[i];
        }
    }
    for (auto& mean : means) {
        mean /= database.size();
    }

    // Calculate standard deviation
    for (const auto& entry : database) {
        for (size_t i = 0; i < numFeatures; ++i) {
            stdDevs[i] += (entry[i] - means[i])*(entry[i] - means[i]);
        }
    }
    for (auto& stdDev : stdDevs) {
        stdDev = std::sqrt(stdDev / database.size());
    }

    return stdDevs;
}

std::vector<float> distanceMetric(const std::vector<float>& feature, const std::vector<std::vector<float>>& database) {
    std::vector<float> stdDevs = calculateStandardDeviations(database);
    std::vector<float> distances;

    for (const auto& dbFeature : database) {
        float distance = 0.0f;
        for (size_t i = 0; i < feature.size(); ++i) {
            // Avoid division by zero in case of constant feature across all vectors
            float scale = stdDevs[i] != 0 ? stdDevs[i] : 1.0f;
            distance += ((feature[i] - dbFeature[i]) / scale)*((feature[i] - dbFeature[i]) / scale);
        }
        distances.push_back(std::sqrt(distance));
    }
    return distances;
}

// Cosine Distance
CosineDistance::CosineDistance(const std::vector<float>& target, const std::vector<float>& img)
: vecA(target), vecB(img) {}

double CosineDistance::dotProduct(const std::vector<float>& A, const std::vector<float>& B) const {
    double product = 0.0;
    for (size_t i = 0; i < A.size(); ++i) {
        product += (double)A[i] * B[i];
    }
    return product;
}

double CosineDistance::vecNorm(const std::vector<float>& V) const {
    double norm = 0.0;
    for (auto& val : V) {
        norm += (double)val * val;
    }
    return sqrt(norm);
}

double CosineDistance::calculate() const {
    double dot = this->dotProduct(vecA, vecB);
    double normA = this->vecNorm(vecA);
    double normB = this->vecNorm(vecB);
    double cosSim = dot / (normA * normB);
    return (float)1.0 - cosSim;
}

// cosine similarity
std::vector<float> compute_similarity(const std::vector<float>& fvec_1, const std::vector<std::vector<float>>& database) {
    std::vector<float> similarity(database.size());
    
    for(size_t i = 0; i < database.size(); ++i) {
        CosineDistance cosineDistance(fvec_1, database[i]);
        similarity[i] = cosineDistance.calculate(); // Compute cosine distance
    }
    
    return similarity;
}

// Confusion Matrix
// Updates the confusion matrix with the true label and the predicted label
void updateConfusionMatrix(const std::string& trueLabel, const std::string& predictedLabel, std::map<std::string, std::map<std::string, int>>& confusionMatrix,
                            std::map<std::string, int>& labelToIndex,std::vector<std::string>& indexToLabel) {
    // If this is a new label, add it to the mapping
    if (labelToIndex.find(trueLabel) == labelToIndex.end()) {
        int newIndex = labelToIndex.size();
        labelToIndex[trueLabel] = newIndex;
        indexToLabel.push_back(trueLabel);
    }
    if (labelToIndex.find(predictedLabel) == labelToIndex.end()) {
        int newIndex = labelToIndex.size();
        labelToIndex[predictedLabel] = newIndex;
        indexToLabel.push_back(predictedLabel);
    }
    // Increment the count in the confusion matrix
    confusionMatrix[trueLabel][predictedLabel]++;
}

// DNN Embedding Part gives the embedding
/*
  cv::Mat src        thresholded and cleaned up image in 8UC1 format
  cv::Mat ebmedding  holds the embedding vector after the function returns
  cv::Rect bbox      the axis-oriented bounding box around the region to be identified
  cv::dnn::Net net   the pre-trained network
  int debug          1: show the image given to the network and print the embedding, 0: don't show extra info
 */
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug)
{
    const int ORNet_size = 128;
    cv::Mat padImg;
    cv::Mat blob;

    cv::Mat roiImg = src(bbox);
    int top = bbox.height > 128 ? 10 : (128 - bbox.height) / 2 + 10;
    int left = bbox.width > 128 ? 10 : (128 - bbox.width) / 2 + 10;
    int bottom = top;
    int right = left;

    cv::copyMakeBorder(roiImg, padImg, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
    cv::resize(padImg, padImg, cv::Size(128, 128));

    cv::dnn::blobFromImage(src,                              // input image
                           blob,                             // output array
                           (1.0 / 255.0) / 0.5,              // scale factor
                           cv::Size(ORNet_size, ORNet_size), // resize the image to this
                           128,                              // subtract mean prior to scaling
                           false,                            // input is a single channel image
                           true,                             // center crop after scaling short side to size
                           CV_32F);                          // output depth/type

    net.setInput(blob);
    embedding = net.forward("/fc1/Gemm_output_0");

    if (debug)
    {
        cv::imshow("pad image", padImg);
        std::cout << embedding << std::endl;
        cv::waitKey(0);
    }

    return (0);
}
