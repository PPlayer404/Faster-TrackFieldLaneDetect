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

// 协程相关，kimi写的，一点看不懂，轻易别动
std::atomic<int64_t> g_decision_wake_ms{ 0 };   // 0 = 没设闹钟

// 给协程用的“sleep”会写这个值
void decision_set_alarm(int ms_from_now) {
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    g_decision_wake_ms.store(now_ms + ms_from_now, std::memory_order_relaxed);
}

// 在 dataSync() 里调用的超短函数
bool decision_alarm_ready() {
    int64_t wake = g_decision_wake_ms.load(std::memory_order_relaxed);
    if (wake == 0) return false;          // 没设闹钟
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now_ms >= wake) {
        g_decision_wake_ms.store(0, std::memory_order_relaxed); // 响铃一次
        return true;
    }
    return false;
}

// 前端声明（本体写在用户文件）
Task decision_coroutine();                      // 用户实现

void decision_start() {
    Task t = decision_coroutine();   // 先拿到 Task
    g_coro_handle = t.h;             // 再取句柄
}

void decision_resume() {
    if (g_coro_handle && !g_coro_handle.done()) {
        g_coro_handle.resume();                 // 跑 1 条语句到下一个 sleep
    }
}
