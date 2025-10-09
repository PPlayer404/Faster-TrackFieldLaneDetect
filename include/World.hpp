#ifndef WORLD_HPP
#define WORLD_HPP
#include <vector>
#include <mutex>
#include <chrono>
#include "Cluster.hpp"
#include <cstdint>

/// @brief 状态机状态定义
namespace state 
{
    inline constexpr int TEST = 0;
    inline constexpr int WAIT = 1;
    inline constexpr int BEFORE_CONE = 2;
    inline constexpr int AVOIDING = 3;
    inline constexpr int AFTER_CONE = 4;
    inline constexpr int MEET_ZEBRA = 5;
    inline constexpr int STOPPED = 6;
    inline constexpr int TURN = 7;
    inline constexpr int GO_STRAIGHT = 8;
    inline constexpr int RETURN = 9;
    inline constexpr int STRAIGHT_AGAIN = 10;
    inline constexpr int FORWARD = 11;
    inline constexpr int TURN2 = 12;
    inline constexpr int FINAL_STOP = 13;

    inline constexpr int LEFT_CONE = 0;
    inline constexpr int RIGHT_CONE = 1;
    inline constexpr int LEFT = 0;
    inline constexpr int RIGHT = 1;
    inline constexpr int A = 0;
    inline constexpr int B = 1;
}

/// @brief yolo的这两个数据类型定义以后要移出去
enum class YoloType : uint8_t { coneBlue, coneYellow, left, right, signA, signB, zebra };
struct YoloBox { YoloType type; float x, y, w, h; };

// 2. 双缓冲内部块 
struct WorldBuf {
    std::vector<ClusterDescriptor> lanes;
    std::vector<YoloBox>           yolos;
    float                          currentSpeed;
    float                          mcuAngle;
};

/* 3. 对外快照（只读，POD 或 vector）*/
struct WorldSnapshot {
    std::vector<ClusterDescriptor> lanes;
    std::vector<YoloBox>           yolos;
    float                          currentSpeed;
    int                            State = state::TEST;
    int                            dX;
    float                          dAngle;
    float                          mcuAngle = 0;
    uint64_t                       frameId = 0;
};

/* 4. World 类 */
class World {
public:
    World();
    /* 4.1 生产者接口：直接写前端，无锁 */
    void updateLanes(std::vector<ClusterDescriptor> v);
    void updateYolos(std::vector<YoloBox> v);
    void updateSpeed(float v);
    void updateMCU(float angle);

    /* 4.2 world 线程入口：返回一份计算好的快照 */
    WorldSnapshot dataSync();

private:
    /* 4.3 双缓冲 + 一把锁 */
    WorldBuf bufFront_;          // 前端（生产者写）
    WorldBuf bufBack_;           // 后端（world 用）
    std::mutex mtx_;             // 只保护 swap
    uint64_t frameId_ = 0;
};

void drawMiddleLines(cv::Mat& frame, float dAngle, int dX);

#endif
