#include <iostream>
#include <iomanip>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "Lane.hpp"
#include "Cluster.hpp"
#include "mode.hpp"
#include "ThreadManager.hpp"
#include <thread>
#include <atomic>
#include <shared_mutex>
#include <cstdint>
#include "Retrans.hpp"
#include "World.hpp"

/// @brief 保证帧读取和分发双缓冲安全，读写锁
std::shared_mutex ReadFreamMutex;
/// @brief 保证图传双缓冲安全
std::mutex SendFreamMutex;
/// @brief 保证世界线程唤醒安全
std::mutex NotifyingMutex;
std::condition_variable LaneAvailableCondition;
volatile bool LaneReady = false;
World gWorld;
WorldSnapshot gSnap;
std::mutex gSnapMutex;

void frameReader()
{
    uint64_t frameId = 0;
#ifdef _WIN32
    cv::VideoCapture cap("img/example.mp4");
    if (!cap.isOpened())
    {
        std::cerr << "无法打开视频\n";
        return;
    }
#else
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        cap.open(1, cv::CAP_V4L2);
        if (!cap.isOpened()) {
            std::cerr << "ERROR: can not open camera\n";
            return;
        }
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

#endif
    cv::Mat frame, processedFrame, tinyImg;
    while (1)
    {
#ifdef DEBUG
        delay_ms(50);
#endif
        cap >> frame;
#ifdef _WIN32
        if (frame.empty()) {
            //重置视频读取指针到开始位置
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
#endif
        frameId++;
        //int startRow = frame.rows / 2;
        //cv::Mat halfImg = frame(cv::Rect(0, startRow, frame.cols, frame.rows - startRow));
        cv::resize(frame, tinyImg, cv::Size(240, 120), 0, 0, cv::INTER_AREA);
        processedFrame = color_correction(tinyImg);
        {
            std::unique_lock<std::shared_mutex> lock(ReadFreamMutex);
            CoreFrameData[!current_read_id].rawFrame = frame;
            CoreFrameData[!current_read_id].processedFrame = processedFrame;
            CoreFrameData[!current_read_id].frameId = frameId;
            current_read_id = 1 - current_read_id;
        }
    }
}

void laneTracker()
{
    uint64_t frameID = 0;
    cv::Mat processedFrame, rawFrame;
    std::vector<cv::Vec4i> lanes;
    std::vector<ClusterDescriptor> clusterDescriptors;
    HSV_Lane HSV_Data_Lane;//巡线用的hsv阈值结构体实例化
    while (1)
    {
        //计时开始
        auto t0 = std::chrono::high_resolution_clock::now();
        //获取帧
        if (CoreFrameData[current_read_id].frameId > frameID)
        {
            std::shared_lock<std::shared_mutex> lock(ReadFreamMutex);
            rawFrame = CoreFrameData[current_read_id].rawFrame;
            processedFrame = CoreFrameData[current_read_id].processedFrame;
            frameID = CoreFrameData[current_read_id].frameId;
        }
        else
        {
            delay_ms(10);
            continue;
        }

        //主逻辑
        lanes = detectLanes(processedFrame, HSV_Data_Lane);
        clusterDescriptors = lanesCluster(lanes);
        gWorld.updateLanes(std::move(clusterDescriptors));//更新数据
        
        //通知世界线程 
        {
            std::unique_lock<std::mutex> lock(NotifyingMutex);
            LaneReady = true;
            LaneAvailableCondition.notify_all();
        }
        std::cout << "noticed\n";
    }
}

void YOLO_Run()
{
    while (1)
    {
        delay_ms(1000);
    }
}

void sendMat()
{
    uint64_t sendID = 0;
    cv::Mat rawFrame;
    while (1)
    {
        if (SendMatData[current_send_id].frameId > sendID)
        {
            std::lock_guard<std::mutex> lock(SendFreamMutex);
            rawFrame = SendMatData[current_send_id].rawFrame;
            sendID = SendMatData[current_send_id].frameId;
        }
        else
        {
            delay_ms(10);
            continue;
        }
        imshow_self(rawFrame);
    }
}

void worldFuse()
{
    using Clock = std::chrono::steady_clock;
    using msDouble = std::chrono::duration<double, std::milli>;

    auto lastWake = Clock::now();
    int   cnt = 0;
    double displayFps = 0.0;          // 真正拿去画图的 fps
    uint64_t sendID = 0;
    cv::Mat rawFrame, processedFrame;

    while (true)
    {
        //等待生产者通知
        {
            std::unique_lock<std::mutex> lock(NotifyingMutex);
            LaneReady = false;
            LaneAvailableCondition.wait(lock, [] { return LaneReady; });
        }

        //计算唤醒间隔 -> fps
        auto now = Clock::now();
        double intervalMs = std::chrono::duration<double, std::milli>(now - lastWake).count();
        lastWake = now;
        double fps = (intervalMs > 0) ? 1000.0 / intervalMs : 0.0;

        //每 20 次求一次平均，避免画面乱跳
        static double fpsAccum = 0.0;
        fpsAccum += fps;
        if (++cnt == 20)
        {
            displayFps = fpsAccum / 20;   // 真正拿去显示的 fps
            fpsAccum = 0.0;
            cnt = 0;
        }

        //拿到融合后的新世界模型快照并全局发布
        WorldSnapshot fresh = gWorld.dataSync();
        {
            std::lock_guard<std::mutex> lk(gSnapMutex);
            gSnap = std::move(fresh);   // 整包移动
        }                               

        //拿帧。lane线程已经帮忙检查过更新了，直接拿即可
        {
            std::shared_lock<std::shared_mutex> lock(ReadFreamMutex);
            rawFrame = CoreFrameData[current_read_id].rawFrame.clone();
        }
        //绘制，带锁保证安全
        {
            std::lock_guard<std::mutex> lk(gSnapMutex);
            drawClusterLines(rawFrame, gSnap.lanes);
			drawMiddleLines(rawFrame, gSnap.dAngle, gSnap.dX);
        }
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << displayFps << " fps";
        cv::putText(rawFrame, oss.str(), cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        std::cout << displayFps << "\n";

        //劫持推流
        {
            std::lock_guard<std::mutex> lock(SendFreamMutex);
            sendID++;
            SendMatData[!current_send_id].rawFrame = rawFrame.clone();
            SendMatData[!current_send_id].frameId = sendID;
            current_send_id = 1 - current_send_id;
        }
    }
}

void controlThread()
{
    while (1)
    {
        delay_ms(1000);
    }
}

void MPUThread()
{
    while (1)
    {
        delay_ms(1000);
    }
}


/// @brief 开始调度，永不退出
/// @return 
int main()
{
#ifdef WITH_IMSHOW
    Lane_init();
#endif
    std::thread t1(frameReader);
    std::thread t2(laneTracker);
    std::thread t3(YOLO_Run);
    std::thread t4(sendMat);
    std::thread t5(worldFuse);
    std::thread t6(controlThread);
    std::thread t7(MPUThread);
    setThreadAffinity(t1, { 2 });
    setThreadAffinity(t2, { 3 });
    setThreadAffinity(t3, { 1 });
    setThreadAffinity(t4, { 0 });
    setThreadAffinity(t5, { 1 });
    setThreadAffinity(t6, { 1 });
    setThreadAffinity(t7, { 1 });
    for (;;)
    {
        delay_ms(10000);
    }
    cv::destroyAllWindows();
    return 0;
}


