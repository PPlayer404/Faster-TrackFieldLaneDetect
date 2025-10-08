#include "ThreadManager.hpp"
#include "opencv2/opencv.hpp"
#include <chrono>
#include <thread>
#include <cstdint>

#ifdef _WIN32
#include<Windows.h>
#else
#include <pthread.h>
#endif

/// @brief 分发读取到的摄像头帧
FrameData CoreFrameData[2]
{
    cv::Mat(),
    cv::Mat(),
    0
};

/// @brief 分发要推流的帧
FrameData SendMatData[2]
{
    cv::Mat(),
    cv::Mat(),
    0
};

//双缓冲标号
std::atomic<std::uint_fast8_t> current_read_id(0);
std::atomic<std::uint_fast8_t> current_send_id(0);

/// @brief 通用延时函数
/// @param milliseconds 毫秒
void delay_ms(unsigned int milliseconds)
{
#ifdef _WIN32

    //Windows
    Sleep(milliseconds);
#else
    //Linux
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
#endif
}

/// @brief 线程绑核
/// @param t 线程
/// @param coreId 物理核心
void setThreadAffinity(std::thread& t, const std::vector<int>& coreIds)
{
#ifdef _WIN32 //win
    HANDLE threadHandle = static_cast<HANDLE>(t.native_handle());
    DWORD_PTR affinityMask = 0;
    for (int coreId : coreIds) {
        affinityMask |= 1ULL << coreId;
    }
    SetThreadAffinityMask(threadHandle, affinityMask);
#else //linux平台下，反正除了win就是linux
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int coreId : coreIds) {
        CPU_SET(coreId, &cpuset);
    }
    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
#endif
}


