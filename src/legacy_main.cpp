#include <iostream>
#include <iomanip>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "Lane.hpp"
#include "Cluster.hpp"
#include "mode.hpp"

int main1()
{
    using namespace std;
    cv::VideoCapture cap("example2.mp4");
    if (!cap.isOpened())
    {
        std::cerr << "无法打开视频\n";
        return -1;
    }

    cv::Mat frame;
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double fontScale = 0.6;
    const int thickness = 2;
    const cv::Scalar color(0, 255, 0);

    int cnt = 0;           // 计数器：0~14，到15归零
    double lastMs = 0.0;   // 真正显示到画面上的耗时

    cv::Mat tinyImg;//交给巡线和锥桶识别的图像，原始帧frame保存，交由后续处理可能的调用
    cv::Mat processed_img;//巡线用的
    cv::Mat blob;
    vector<cv::Vec4i> lanes;
    std::vector<ClusterDescriptor> clusterDescriptors;
    HSV_Lane HSV_Data_Lane;//巡线用的hsv阈值结构体实例化
#ifdef WITH_IMSHOW
    Lane_init();
#endif
    for (;;)
    {
        cap >> frame;
        if (frame.empty()) {
            //重置视频读取指针到开始位置
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        //每帧都计时
        auto t0 = std::chrono::high_resolution_clock::now();

        //主逻辑从这里开始
        cv::resize(frame, tinyImg, cv::Size(240, 120), 0, 0, cv::INTER_AREA);
        processed_img = color_correction(tinyImg);
        lanes = detectLanes(processed_img, HSV_Data_Lane);
        clusterDescriptors = lanesCluster(lanes);
        drawClusterLines(frame, clusterDescriptors);

#ifdef WITH_IMSHOW
        imshow("frame", frame);
#endif



















        //主逻辑到此结束，剩下的是性能统计部分

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        //更新计数器
        ++cnt;
        if (cnt == 20)
        {
            lastMs = ms;   // 第15帧时才把这次耗时拿去显示
            cnt = 0;       // 归零
        }

        //把 lastMs 画到画面上
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << lastMs << " ms  quit==ESC";
        std::cout << oss.str() << std::endl;
        cv::putText(processed_img, oss.str(), cv::Point(10, 30),
            font, fontScale, color, thickness);
#ifdef WITH_IMSHOW
        cv::imshow("video", processed_img);

        if (cv::waitKey(1) == 27) break;
#endif
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}




