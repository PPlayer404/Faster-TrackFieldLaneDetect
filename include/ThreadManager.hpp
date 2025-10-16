#ifndef THREAD_MANAGER_HPP
#define THREAD_MANAGER_HPP
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <cstdint>
#include <coroutine>
#include <chrono>

extern std::atomic<std::uint_fast8_t> current_read_id;
extern std::atomic<std::uint_fast8_t> current_send_id;

//全局图像帧数据结构
struct FrameData {
    cv::Mat rawFrame;           //原始帧，给模型的
    cv::Mat processedFrame;     //预处理帧
    uint64_t frameId;           //帧编号
};

extern FrameData CoreFrameData[2];
extern FrameData SendMatData[2];
void delay_ms(unsigned int milliseconds);
void setThreadAffinity(std::thread& t, const std::vector<int>& coreIds);

extern std::atomic<int64_t> g_decision_wake_ms;
bool decision_alarm_ready();
void decision_set_alarm(int ms_from_now);

struct Task {
    std::coroutine_handle<> h;   // 保存句柄

    struct promise_type {
        Task get_return_object() {
            return { std::coroutine_handle<promise_type>::from_promise(*this) };
        }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };
};
static std::coroutine_handle<> g_coro_handle;

void decision_start();
void decision_resume();

#endif
