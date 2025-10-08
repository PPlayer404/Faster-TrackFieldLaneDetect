#ifndef THREAD_MANAGER_HPP
#define THREAD_MANAGER_HPP
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <cstdint>

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

#endif
