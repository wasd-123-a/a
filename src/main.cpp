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

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance);

int main(int argc, char** argv)
{
    String left_img = "../l.jpg";
    String right_img = "../r.jpg";
    String dst_path = "../filter.jpg"; //保存生成的经过滤波的视差图
    String dst_raw_path = "../origin.jpg"; //保存原始视差图
    String dst_conf_path = "None"; //保存置信度图

    String algo = "bm"; //立体匹配方法 (bm or sgbm)
    String filter = "wls_no_conf"; //使用后滤波 (wls_conf or wls_no_conf)

    bool no_downscale = true; //强制使用全尺寸视图进行立体匹配以提高质量;
    int max_disp = 48; //立体匹配参数, numDisparities

    double lambda = 8000.0; //后滤波参数, wls_lambda
    double sigma  = 1.0; //后滤波参数, wls_sigma

    int wsize;
    if(algo=="sgbm")
        wsize = 3; //默认窗口大小为3
    else if(!no_downscale && algo=="bm" && filter=="wls_conf")
        wsize = 7; //默认窗口大小为7
    else
        wsize = 15; //默认窗口大小为15

    //! [load_views]
    Mat left  = imread(left_img, IMREAD_COLOR);
    if ( left.empty() )
    {
        cout << "Cannot read image file: " << left_img;
        return -1;
    }

    Mat right = imread(right_img, IMREAD_COLOR);
    if ( right.empty() )
    {
        cout << "Cannot read image file: " << right_img;
        return -1;
    }
    //! [load_views]

    Mat left_for_matcher, right_for_matcher;
    Mat left_disp,right_disp;
    Mat filtered_disp;
    Mat conf_map = Mat(left.rows, left.cols, CV_8U);
    conf_map = Scalar(255);
    Rect ROI;
    Ptr<DisparityWLSFilter> wls_filter;
    double matching_time, filtering_time;
    if(max_disp<=0 || max_disp%16!=0)
    {
        cout << "Incorrect max_disparity value: it should be positive and divisible by 16";
        return -1;
    }
    if(wsize<=0 || wsize%2!=1)
    {
        cout << "Incorrect window_size value: it should be positive and odd";
        return -1;
    }
    if(filter=="wls_conf") // 使用置信度进行滤波（比wls_no_conf质量更好）
    {
        if(!no_downscale)
        {
            // 缩小图像以加速匹配阶段，因为我们需要计算左右视图的置信度图
            //! [downscale]
            max_disp/=2;
            if(max_disp%16!=0)
                max_disp += 16-(max_disp%16);
            resize(left ,left_for_matcher ,Size(),0.5,0.5);
            resize(right,right_for_matcher,Size(),0.5,0.5);
            //! [downscale]
        }
        else
        {
            left_for_matcher  = left.clone();
            right_for_matcher = right.clone();
        }

        if(algo=="bm")
        {
            //! [matching]
            Ptr<StereoBM> left_matcher = StereoBM::create(max_disp,wsize);
            wls_filter = createDisparityWLSFilter(left_matcher);
            Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

            cvtColor(left_for_matcher, left_for_matcher,  COLOR_BGR2GRAY);
            cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);

            matching_time = (double)getTickCount();
            left_matcher-> compute(left_for_matcher, right_for_matcher, left_disp);
            right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
            matching_time = ((double)getTickCount() - matching_time) / getTickFrequency();
            //! [matching]
        }
        else if(algo=="sgbm")
        {
            Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,wsize);
            left_matcher->setP1(24*wsize*wsize);
            left_matcher->setP2(96*wsize*wsize);
            left_matcher->setPreFilterCap(63);
            left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
            wls_filter = createDisparityWLSFilter(left_matcher);
            Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

            matching_time = (double)getTickCount();
            left_matcher-> compute(left_for_matcher, right_for_matcher, left_disp);
            right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
            matching_time = ((double)getTickCount() - matching_time) / getTickFrequency();
        }
        else
        {
            cout<<"Unsupported algorithm";
            return -1;
        }

        //! [filtering]
        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        filtering_time = (double)getTickCount();
        wls_filter->filter(left_disp, left, filtered_disp, right_disp);
        filtering_time = ((double)getTickCount() - filtering_time) / getTickFrequency();
        //! [filtering]
        conf_map = wls_filter->getConfidenceMap();

        // Get the ROI that was used in the last filter call:
        ROI = wls_filter->getROI();
        if(!no_downscale)
        {
            // upscale raw disparity and ROI back for a proper comparison:
            resize(left_disp, left_disp, Size(), 2.0, 2.0);
            left_disp = left_disp * 2.0;
            ROI = Rect(ROI.x*2, ROI.y*2, ROI.width*2, ROI.height*2);
        }
    }
    else if(filter=="wls_no_conf")
    {
        /* 没有方便的函数来处理没有置信度的情况，所以我们需要手动设置ROI和匹配器参数 */

        left_for_matcher  = left.clone();
        right_for_matcher = right.clone();

        if(algo=="bm")
        {
            cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create();
            bm->setPreFilterType(cv::StereoBM::PREFILTER_NORMALIZED_RESPONSE);
            bm->setPreFilterSize(9);
            bm->setPreFilterCap(31);
            bm->setBlockSize(15);
            bm->setMinDisparity(0);
            bm->setNumDisparities(48);
            bm->setTextureThreshold(10);
            bm->setUniquenessRatio(15);
            bm->setSpeckleWindowSize(100);
            bm->setSpeckleRange(32);

            cvtColor(left_for_matcher, left_for_matcher, COLOR_BGR2GRAY);
            cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);

            ROI = computeROI(left_for_matcher.size(), bm);
            wls_filter = createDisparityWLSFilterGeneric(false);
            wls_filter->setDepthDiscontinuityRadius((int)ceil(0.33 * wsize));

            matching_time = (double)getTickCount();
            bm->compute(left_for_matcher, right_for_matcher, left_disp);
            matching_time = ((double)getTickCount() - matching_time) / getTickFrequency();
        }
        else if(algo=="sgbm")
        {
            cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(-16, 48, 15, 0, 0, 1, 31, 15, 100, 2, 0);

            ROI = computeROI(left_for_matcher.size(), sgbm);
            wls_filter = createDisparityWLSFilterGeneric(false);
            wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5 * wsize));

            matching_time = (double)getTickCount();
            sgbm->compute(left_for_matcher, right_for_matcher, left_disp);
            matching_time = ((double)getTickCount() - matching_time) / getTickFrequency();
        }
        else
        {
            cout << "Unsupported algorithm";
            return -1;
        }

        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        filtering_time = (double)getTickCount();
        wls_filter->filter(left_disp, left, filtered_disp, Mat(), ROI);
        filtering_time = ((double)getTickCount() - filtering_time) / getTickFrequency();
    }
    else
    {
        cout << "Unsupported filter";
        return -1;
    }

    // 收集并打印所有统计数据:
    cout.precision(3);
    cout << "Matching time:  " << matching_time<< "s" << endl;
    cout << "Filtering time: " << filtering_time<< "s" << endl;
    cout<<endl;

    if(dst_path != "None")
    {
        imwrite(dst_path, filtered_disp);
    }
    if(dst_raw_path != "None")
    {
        imwrite(dst_raw_path, left_disp);
    }
    if(dst_conf_path != "None")
    {
        imwrite(dst_conf_path, conf_map);
    }

   // imshow("left", left);
   // imshow("right", right);

    Mat raw_disp_vis;
    //getDisparityVis(left_disp, raw_disp_vis);
    left_disp.convertTo(raw_disp_vis, CV_8U, 0.3);
    imshow("raw disparity", raw_disp_vis);

    Mat filtered_disp_vis;
    //getDisparityVis(filtered_disp, filtered_disp_vis);
    filtered_disp.convertTo(filtered_disp_vis, CV_8U, 1);
    imshow("filtered disparity", filtered_disp_vis);
    waitKey(0);

    return 0;
}

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance)
{
    int min_disparity = matcher_instance->getMinDisparity();
    int num_disparities = matcher_instance->getNumDisparities();
    int block_size = matcher_instance->getBlockSize();

    int bs2 = block_size / 2;
    int minD = min_disparity;
    int maxD = min_disparity + num_disparities - 1;

    int xmin = maxD + bs2;
    int xmax = src_sz.width + minD - bs2;
    int ymin = bs2;
    int ymax = src_sz.height - bs2;

    Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
    return r;
}

