#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

static void showHelp() {
    cout << "Usage:\n"
            "  edge_detection [image] [method] [out] [params...]\n\n"
            "Methods and params (all optional with defaults):\n"
            "  sobel [ksize=3 dx=1 dy=1 scale=1.0 delta=0.0]\n"
            "  scharr [dx=1 dy=0]\n"
            "  laplacian [ksize=3 scale=1.0 delta=0.0]\n"
            "  canny [threshold1=100 threshold2=200 apertureSize=3 L2gradient=0]\n\n"
            "Examples:\n"
            "  ./edge_detection ../l.jpg sobel edges.png 3 1 1 1.0 0.0\n"
            "  ./edge_detection ../l.jpg scharr edges.png 1 0\n"
            "  ./edge_detection ../l.jpg laplacian edges.png 3 1.0 0.0\n"
            "  ./edge_detection ../l.jpg canny edges.png 50 150 3 1\n";
}

int main(int argc, char** argv)
{
    string inPath  = "../l.jpg";
    string method  = "sobel"; // default
    string outPath = "../edges.png";

    if (argc >= 2) method  = argv[1];
    if (argc >= 3) inPath  = argv[2];
    if (argc >= 4) outPath = argv[3];

    Mat src = imread(inPath, IMREAD_COLOR);
    if (src.empty()) {
        cerr << "Cannot read image: " << inPath << endl;
        showHelp();
        return -1;
    }

    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    Mat edges;

    if (method == "sobel") {
        int ksize = (argc >= 5) ? atoi(argv[4]) : 3; // 1,3,5,7
        int dx    = (argc >= 6) ? atoi(argv[5]) : 1;
        int dy    = (argc >= 7) ? atoi(argv[6]) : 1;
        double scale = (argc >= 8) ? atof(argv[7]) : 1.0;
        double delta = (argc >= 9) ? atof(argv[8]) : 0.0;
        if (ksize % 2 == 0) ksize += 1;
        if (ksize <= 0) ksize = 3;

        Mat grad_x, grad_y, abs_x, abs_y;
        Sobel(gray, grad_x, CV_16S, dx, 0, ksize, scale, delta);
        Sobel(gray, grad_y, CV_16S, 0, dy, ksize, scale, delta);
        convertScaleAbs(grad_x, abs_x);
        convertScaleAbs(grad_y, abs_y);
        addWeighted(abs_x, 0.5, abs_y, 0.5, 0, edges);
    }
    else if (method == "scharr") {
        int dx = (argc >= 5) ? atoi(argv[4]) : 1; 
        int dy = (argc >= 6) ? atoi(argv[5]) : 1; 
        
        Mat grad_x, grad_y, abs_x, abs_y;
        Scharr(gray, grad_x, CV_16S, dx, 0);
        Scharr(gray, grad_y, CV_16S, 0, dy);
        convertScaleAbs(grad_x, abs_x);
        convertScaleAbs(grad_y, abs_y);
        addWeighted(abs_x, 0.5, abs_y, 0.5, 0, edges);
    }
    else if (method == "laplacian") {
        int ksize   = (argc >= 5) ? atoi(argv[4]) : 5; 
        double scale = (argc >= 6) ? atof(argv[5]) : 1.0;
        double delta = (argc >= 7) ? atof(argv[6]) : 0.0;
        if (ksize % 2 == 0) ksize += 1;
        Mat lap, abs_lap;
        Laplacian(gray, lap, CV_16S, ksize, scale, delta);
        convertScaleAbs(lap, edges);
    }
    else if (method == "canny") {
        Mat blur;
       // GaussianBlur(gray, blur, Size(3,3), 0);
        double th1 = (argc >= 5) ? atof(argv[4]) : 100.0;
        double th2 = (argc >= 6) ? atof(argv[5]) : 200.0;
        int aperture = (argc >= 7) ? atoi(argv[6]) : 5; 
        bool L2 = (argc >= 8) ? (atoi(argv[7]) != 0) : false;
        Canny(blur, edges, th1, th2, aperture, L2);
    }
    else {
        cerr << "Unknown method: " << method << endl;
        showHelp();
        return -1;
    }

    imwrite(outPath, edges);

    imshow("Edges - " + method, edges);
    waitKey(0);
    return 0;
}


