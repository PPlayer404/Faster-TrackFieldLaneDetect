#ifndef WORLD_HPP
#define WORLD_HPP
#include <vector>
#include <mutex>
#include <chrono>
#include "Cluster.hpp"

#define TEST 0
#define WAIT 1
#define BEFORE_CONE 2
#define AVOIDING 3
#define AFTER_CONE 4
#define MEET_ZEBRA 5
#define STOPPED 6
#define TURN 7
#define GO_STRAIGHT 8
#define RETURN 9
#define STRAIGHT_AGAIN 10
#define FORWARD 11
#define TURN2 12
#define FINAL_STOP 13

#define LEFT_CONE 0 
#define RIGHT_CONE 1
#define LEFT 0
#define RIGHT 1
#define A 0
#define B 1

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
    int                            State = TEST;
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

#endif
