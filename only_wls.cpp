#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

// Helper to convert any depth image to single-channel CV_32F
static bool loadDepthAsFloat(const String &path, Mat &depth32f) {
    Mat src = imread(path, IMREAD_UNCHANGED);
    if (src.empty()) {
        cerr << "Cannot read depth file: " << path << endl;
        return false;
    }
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
        case CV_8U:  single.convertTo(depth32f, CV_32F, 1.0);                 break;
        case CV_16U: single.convertTo(depth32f, CV_32F, 1.0);                 break;
        case CV_16S: single.convertTo(depth32f, CV_32F, 1.0);                 break;
        case CV_32S: single.convertTo(depth32f, CV_32F, 1.0);                 break;
        case CV_32F: depth32f = single;                                       break;
        case CV_64F: single.convertTo(depth32f, CV_32F, 1.0);                 break;
        default:
            cerr << "Unsupported depth type: depth=" << single.depth() << endl;
            return false;
    }
    return true;
}

// Save float depth back to original-like type
static bool saveDepthLike(const Mat &depth32f, const String &outPath, int likeDepth) {
    Mat out;
    switch (likeDepth) {
        case CV_8U:  depth32f.convertTo(out, CV_8U);  break;
        case CV_16U: depth32f.convertTo(out, CV_16U); break;
        case CV_16S: depth32f.convertTo(out, CV_16S); break;
        case CV_32S: depth32f.convertTo(out, CV_32S); break;
        case CV_32F: out = depth32f.clone();          break;
        case CV_64F: depth32f.convertTo(out, CV_64F); break;
        default:     cerr << "Unsupported output type" << endl; return false;
    }
    return imwrite(outPath, out);
}

int main(int argc, char** argv) {
    // Inputs
    String depth_path = "../depth.png";     
    String out_path   = "../depth_filtered.png"; // 输出滤波后深度图（类型同输入）

    // Parameters
    double lambda = 20000.0; // 平滑强度（越大越平滑）
    double sigma  = 0.5;    // 颜色/引导相似性（这里我们用深度自身归一化为引导）
    int wsize     = 15;     // 影响不连续半径

    if (argc >= 2) depth_path = argv[1];
    if (argc >= 3) out_path   = argv[2];

    // 将深度作为浮点数加载
    Mat depth32f;
    if (!loadDepthAsFloat(depth_path, depth32f)) return -1;
    int inputDepthType = imread(depth_path, IMREAD_UNCHANGED).depth();

    // 可选的预滤波，减少斑点
    medianBlur(depth32f, depth32f, 3);

    // 从深度自身构建引导图（归一化为8U）
    Mat guide8u;
    double minV=0.0, maxV=0.0; minMaxLoc(depth32f, &minV, &maxV);
    if (maxV - minV < 1e-6) {
        guide8u = Mat(depth32f.size(), CV_8U, Scalar(128));
    } else {
        Mat guideFloat;
        normalize(depth32f, guideFloat, 0, 255, NORM_MINMAX);
        guideFloat.convertTo(guide8u, CV_8U);
    }

    // 将深度映射到WLS期望的CV_16S域（使用固定比例保持行为稳定）
    const double s = 16.0; // mimic disparity fixed-point scale
    Mat depth16s; depth32f.convertTo(depth16s, CV_16S, s);

    // 应用WLS
    Ptr<DisparityWLSFilter> wls = createDisparityWLSFilterGeneric(false);
    wls->setLambda(lambda);
    wls->setSigmaColor(sigma);
    wls->setDepthDiscontinuityRadius((int)ceil(0.33 * wsize));

    Mat filtered16s;
    Rect ROI(0, 0, depth16s.cols, depth16s.rows);
    wls->filter(depth16s, guide8u, filtered16s, Mat(), ROI);

    // 转换回原始数值比例/类型
    Mat filtered32f; filtered16s.convertTo(filtered32f, CV_32F, 1.0/s);

    if (!saveDepthLike(filtered32f, out_path, inputDepthType)) {
        cerr << "Failed to save: " << out_path << endl;
        return -1;
    }
    cout << "Saved filtered depth to: " << out_path << endl;
    return 0;
}
