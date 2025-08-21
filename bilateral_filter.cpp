#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// 将类似深度的图像加载为单通道 CV_32F 并返回原始深度类型
static bool loadDepthAsFloat(const String &path, Mat &depth32f, int &origDepthType)
{
    Mat src = imread(path, IMREAD_UNCHANGED);
    if (src.empty()) {
        cerr << "Cannot read depth file: " << path << endl;
        return false;
    }

    origDepthType = src.depth();

    Mat single;
    if (src.channels() == 1) {
        single = src;
    } else if (src.channels() == 3) {
        cvtColor(src, single, COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cvtColor(src, single, COLOR_BGRA2GRAY);
    } else {
        Mat ch0; extractChannel(src, ch0, 0); single = ch0;
    }

    switch (single.depth()) {
        case CV_8U:  single.convertTo(depth32f, CV_32F, 1.0); break;
        case CV_16U: single.convertTo(depth32f, CV_32F, 1.0); break;
        case CV_16S: single.convertTo(depth32f, CV_32F, 1.0); break;
        case CV_32S: single.convertTo(depth32f, CV_32F, 1.0); break;
        case CV_32F: depth32f = single;                       break;
        case CV_64F: single.convertTo(depth32f, CV_32F, 1.0); break;
        default:
            cerr << "Unsupported depth type: depth=" << single.depth() << endl;
            return false;
    }
    return true;
}

// Save float depth back to original-like type
static bool saveDepthLike(const Mat &depth32f, const String &outPath, int likeDepth)
{
    Mat out;
    switch (likeDepth) {
        case CV_8U:  depth32f.convertTo(out, CV_8U);  break;
        case CV_16U: depth32f.convertTo(out, CV_16U); break;
        case CV_16S: depth32f.convertTo(out, CV_16S); break;
        case CV_32S: depth32f.convertTo(out, CV_32S); break;
        case CV_32F: out = depth32f.clone();          break;
        case CV_64F: depth32f.convertTo(out, CV_64F); break;
        default:
            cerr << "Unsupported output type" << endl;
            return false;
    }
    return imwrite(outPath, out);
}

int main(int argc, char** argv)
{

    String depth_path = "../depth.png";
    String out_path   = "../depth_bilateral.png";


    int    diameter      = 9;     // 必须为奇数；如果小于等于0，则由OpenCV根据sigmaSpace自动计算
    double sigmaColor    = -1.0;  // 如果小于0，则设置为数据范围的10%
    double sigmaSpace    = 15.0;  // 空间sigma
    int    iterations    = 1;     // 如果需要，多次应用滤波器

    if (argc >= 2) depth_path = argv[1];
    if (argc >= 3) out_path   = argv[2];
    if (argc >= 4) diameter   = atoi(argv[3]);
    if (argc >= 5) sigmaColor = atof(argv[4]);
    if (argc >= 6) sigmaSpace = atof(argv[5]);
    if (argc >= 7) iterations = atoi(argv[6]);

    // 加载深度图
    Mat depth32f;
    int origDepth = -1;
    if (!loadDepthAsFloat(depth_path, depth32f, origDepth)) return -1;

    // 使直径为奇数且 >= 1
    if (diameter <= 0) diameter = 9;
    if (diameter % 2 == 0) diameter += 1;
    if (iterations < 1) iterations = 1;

    // 从数据范围计算合理的sigmaColor
    double minV = 0.0, maxV = 0.0;
    minMaxLoc(depth32f, &minV, &maxV);
    double range = maxV - minV;
    if (sigmaColor < 0.0) {
        // 10%作为默认的动态范围  
        sigmaColor = (range > 0.0) ? 0.10 * range : 1.0;
    }

    // 应用双边滤波器（如果需要则迭代应用）
    Mat current = depth32f, filtered;
    for (int i = 0; i < iterations; ++i) {
        bilateralFilter(current, filtered, diameter, sigmaColor, sigmaSpace);
        current = filtered;
    }

    // 保存为原始数值类型
    if (!saveDepthLike(filtered, out_path, origDepth)) {
        cerr << "Failed to save: " << out_path << endl;
        return -1;
    }
    cout << "Saved bilateral-filtered depth to: " << out_path << endl;
    cout << "Params: d=" << diameter << ", sigmaColor=" << sigmaColor
         << ", sigmaSpace=" << sigmaSpace << ", iterations=" << iterations << endl;
    return 0;
}


