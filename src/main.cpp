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
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif

/// @brief 保证帧读取和分发双缓冲安全，读写锁
std::shared_mutex ReadFreamMutex;
/// @brief 保证图传双缓冲安全
std::mutex SendFreamMutex;
/// @brief 保证世界线程唤醒安全
std::mutex NotifyingMutex;
std::condition_variable LaneAvailableCondition;
std::atomic<uint64_t> gNewestFrameId{ 0 };   // 只放最新帧号
World gWorld;
WorldSnapshot gSnap;
std::mutex gSnapMutex;

/// @brief 读取摄像头线程，负责读取摄像头帧并进行预处理
void frameReader()
{
    uint64_t frameId = 0;
#ifdef _WIN32
    // Windows环境下从图片路径读取图片
    std::string imgPath = "E:/img/imgs/img6/";  // 修改为正确的路径分隔符
    int imgIndex = 1;

    // 检查路径是否存在
    DWORD fileAttr = GetFileAttributesA(imgPath.c_str());
    if (fileAttr == INVALID_FILE_ATTRIBUTES || !(fileAttr & FILE_ATTRIBUTE_DIRECTORY))
    {
        std::cerr << "无法找到图片路径: " << imgPath << "\n";
        return;
    }
#else
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        cap.open(1, cv::CAP_V4L2);
        if (!cap.isOpened())
        {
            std::cerr << "ERROR: can not open camera\n";
            return;
        }
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 2);
#endif

    cv::Mat frame, processedFrame, tinyImg;
    while (1)
    {
#ifdef DEBUG
        delay_ms(100);
#endif

#ifdef _WIN32
        std::string imgFile = imgPath + std::to_string(imgIndex) + ".jpeg";
        frame = cv::imread(imgFile);
        // 如果图片不存在，重置到第一张
        if (frame.empty())
        {
            imgIndex = 1;
            imgFile = imgPath + std::to_string(imgIndex) + ".jpeg";
            frame = cv::imread(imgFile);
        }
        // 递增图片索引，准备读取下一张
        imgIndex++;
        // 如果下一张图片不存在，重置索引（循环播放）
        std::string nextImgFile = imgPath + std::to_string(imgIndex) + ".jpeg";
        if (!cv::imread(nextImgFile).data)
        {
            imgIndex = 1;
        }
#else
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "无法从摄像头读取帧\n";
            continue;
        }
#endif
        frameId++;
        // 保持原有的图像处理逻辑不变
        int startRow = frame.rows * 2 / 5;  // 上1/2的起始位置
        int endRow = frame.rows * 5 / 6;  // 下1/6的起始位置
        cv::Mat halfImg = frame(cv::Rect(0, startRow, frame.cols, endRow - startRow));
        cv::resize(halfImg, tinyImg, cv::Size(240, 120), 0, 0, cv::INTER_AREA);
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

/// @brief 巡线线程，负责车道线检测和聚类，上传原始描述子
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
            uint64_t id = CoreFrameData[current_read_id].frameId;  // 拿到当前帧号
            gNewestFrameId.store(id, std::memory_order_release);
            LaneAvailableCondition.notify_one();
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

/// @brief 图传线程，负责将处理后的帧推送到虚拟摄像头
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

/// @brief 融合传感器数据，更新世界模型，输出控制信息
/// 主线程
void worldFuse()
{
    using Clock = std::chrono::steady_clock;
    using msDouble = std::chrono::duration<double, std::milli>;

    auto lastWake = Clock::now();
    int   cnt = 0;
    double displayFps = 0.0;          // 真正拿去画图的 fps
    uint64_t sendID = 0;
    cv::Mat rawFrame, processedFrame;
    static uint64_t lastFrameId = 0;   // 上次已处理的帧号
    static uint64_t lastId = 0;
    while (true)
    {
        //等待生产者通知
        uint64_t nowId;
        {
            std::unique_lock<std::mutex> lock(NotifyingMutex);
            LaneAvailableCondition.wait(lock, [] {
                return gNewestFrameId.load(std::memory_order_acquire) > lastFrameId;
                });
            nowId = gNewestFrameId.load(std::memory_order_relaxed);
        }
        lastFrameId = nowId;   // 锁外更新
        std::cout << "weaken\n";

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
            displayFps = fpsAccum / 20;
            fpsAccum = 0.0;
            cnt = 0;
        }

        //拿到融合后的新世界模型快照并全局发布
        WorldSnapshot fresh = gWorld.dataSync();
        {
            std::lock_guard<std::mutex> lk(gSnapMutex);
            gSnap = std::move(fresh);   // 整包移动
        }

        //调度决策协程，真正的控制逻辑入口
        if (decision_alarm_ready())
            decision_resume();

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

/// @brief 唤醒周期100Hz，100Hz读取imu数据，50Hz平滑更新目标舵机角度和电机输出
void imuAndControl()
{
    using clock = std::chrono::steady_clock;
    const auto period = std::chrono::milliseconds(10); // 100 Hz
    auto next = clock::now() + period;                 // 第一次的到期点
    uint8_t cnt = 0;

    while (true)
    {
        delay_ms(5);    // 占位，在这里读取imu数据

        //50Hz唤醒
        if (cnt == 0)
        {
            //读取当前的目标舵机角度和电机输出，然后限制jerk平滑输出
        }
        cnt = 1 - cnt;

        std::this_thread::sleep_until(next);
        next += period;
    }
}


/// @brief 开始调度，永不退出
/// @return 
int main()
{
#ifdef WITH_IMSHOW
    Lane_init();
#endif
    decision_start();
    std::thread t1(frameReader);
    std::thread t2(laneTracker);
    std::thread t3(YOLO_Run);
    std::thread t4(sendMat);
    std::thread t5(worldFuse);
    std::thread t6(imuAndControl);
    setThreadAffinity(t1, { 2 });
    setThreadAffinity(t2, { 3 });
    setThreadAffinity(t3, { 1 });
    setThreadAffinity(t4, { 0 });
    setThreadAffinity(t5, { 1 });
    setThreadAffinity(t6, { 1 });
    for (;;)
    {
        delay_ms(10000);
    }
    cv::destroyAllWindows();
    return 0;
}


