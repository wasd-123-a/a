#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

int main(int argc, char** argv)
{
    String left_img = "../l.jpg";    
    String disp_img = "../depth.png";   
    
    String dst_path = "../filtered_disparity.jpg";
    String dst_raw_path = "../original_disparity.jpg"; // 原始

    double filtering_time = (double)getTickCount();
    
    double lambda = 8000.0;              // 滤波强度参数（数值越大平滑越强）
    double sigma = 1.0;                  // 相似性阈值
    int wsize = 15;                      // 窗口大小 

    Mat left = imread(left_img, 0);
    if (left.empty())
    {
        cout << "left NOT FOUND " << left_img << endl;
        return -1;
    }
   
    Mat loaded_disp = imread(disp_img, IMREAD_UNCHANGED);
    if (loaded_disp.empty())
    {
        cout << "disp NOT FOUND " << disp_img << endl;
        return -1;
    }
 
    Mat disp_single;
    // switch (loaded_disp.channels()) 
    // {
    //     case 1: 
        disp_single = loaded_disp;
    //     break;

    //     case 3: cvtColor(loaded_disp, disp_single, COLOR_BGR2GRAY);  break;
    //     case 4: cvtColor(loaded_disp, disp_single, COLOR_BGRA2GRAY);  break;
    //     default:
    //         Mat ch0;
    //         extractChannel(loaded_disp, disp_single, 0);
    //         disp_single = ch0;
            
    //         break;
    // }
    
    // 转换为CV_16S格式
    Mat left_disp;
    switch (disp_single.depth())
    {
        case CV_8U: disp_single.convertTo(left_disp, CV_16S, 16.0);  break;
        case CV_16U: disp_single.convertTo(left_disp, CV_16S, 1.0);  break;
        case CV_16S: left_disp = disp_single;  break;
        case CV_32S: disp_single.convertTo(left_disp, CV_16S, 1.0);  break;
        case CV_32F: disp_single.convertTo(left_disp, CV_16S, 16.0);  break;
        case CV_64F: disp_single.convertTo(left_disp, CV_16S, 16.0);  break;
        default:
        cerr << "Unsupported depth type: depth=" << disp_single.depth() << endl;
            return -1;
    }
     
    // 创建WLS滤波器
    Ptr<DisparityWLSFilter> wls_filter = createDisparityWLSFilterGeneric(false);

    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    wls_filter->setDepthDiscontinuityRadius((int)ceil(0.33 * wsize));

    Rect ROI(0, 0, left_disp.cols, left_disp.rows);
    

    Mat filtered_disp;
    

    wls_filter->filter(left_disp, left, filtered_disp, Mat(), ROI);
    
    filtering_time = ((double)getTickCount() - filtering_time) / getTickFrequency();
    cout << filtering_time << endl;
    
    //imwrite(dst_raw_path, left_disp);

    //imwrite(dst_path, filtered_disp);


    Mat raw_disp_vis;
    getDisparityVis(left_disp, raw_disp_vis);
    left_disp.convertTo(raw_disp_vis, CV_8U, 0.3);
    imshow("原始深度图", raw_disp_vis);
    
    Mat filtered_disp_vis;
    getDisparityVis(filtered_disp, filtered_disp_vis);
    filtered_disp.convertTo(filtered_disp_vis, CV_8U, 1);
    imshow("WLS滤波后深度图", filtered_disp_vis);
    
    waitKey(0);

    return 0;
}
