#include "World.hpp"
#include "Cluster.hpp"

/* 生产者：只碰前端，无锁 */
void World::updateLanes(std::vector<ClusterDescriptor> v)
{
    bufFront_.lanes = std::move(v);
}

void World::updateYolos(std::vector<YoloBox> v)
{
    bufFront_.yolos = std::move(v);
}

void World::updateSpeed(float v)
{
    bufFront_.currentSpeed = v;
}

void World::updateMCU(float angle)
{
    bufFront_.mcuAngle = angle;
}

//world 线程：双缓冲 + 计算 + 返回快照
WorldSnapshot World::dataSync()
{
    //锁内指针交换
    {
        std::lock_guard<std::mutex> lk(mtx_);
        std::swap(bufFront_, bufBack_);
        ++frameId_;
    }  // 锁已释放

    //无锁，用的数据是 bufBack_ 
    WorldSnapshot snap
    {
        bufBack_.lanes,//lanes
        bufBack_.yolos,//yolos
        bufBack_.currentSpeed,//currentSpeed
        TEST,//state
        0,//dx
        0,//dAngle
        0,//mcuAngle
        frameId_//frameId
    };

    //融合/滤波，snap 的字段上算，算完 return //
    // fusion(snap.lanes, snap.yolos, snap.state);

    return snap;  // NRVO/move，读者无锁
}

World::World()
{
    WorldBuf init;
    init.currentSpeed = 0.0f;          // 举例：默认车速 3 m/s
    init.mcuAngle = 0;
    // yolos 默认空，就不填了

    bufFront_ = init;
    bufBack_ = init;   // 两端保持一致，第一次 swap 后读者就能拿到
}
