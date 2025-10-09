此Readme与项目功能无关，纯粹是一个接口记录

世界模型双缓冲发布机制

生产者写入缓冲区A
世界线程唤醒后交换缓冲区AB
WorldSnapshot World::dataSync()内加锁提取缓冲区B
WorldSnapshot World::dataSync()内无锁计算滤波融合

sync返回融合好的数据，世界线程全局发布：
WorldSnapshot fresh = gWorld.dataSync();
{
    std::lock_guard<std::mutex> lk(gSnapMutex);
    gSnap = std::move(fresh);   // 整包移动   
}

其他线程带互斥锁读取发布的世界数据：
WorldSnapshot snap;
{
    std::lock_guard<std::mutex> lk(gSnapMutex);
    snap = gSnap;          // 拷贝（或移动）到本地
}



帧分发双缓冲机制

单个数据包构成：两帧图像加一个帧标号，实例化为结构体数组
extern std::atomic<std::uint_fast8_t> current_read_id;来切换双缓冲
struct FrameData {
    cv::Mat rawFrame;           //原始帧，给模型的
    cv::Mat processedFrame;     //预处理帧
    uint64_t frameId;           //帧编号
};

frameID：记录当前帧的标号，接收端根据ID确保自己接受新帧

发布：
{
    std::unique_lock<std::shared_mutex> lock(ReadFreamMutex);
    CoreFrameData[!current_read_id].rawFrame = frame;
    CoreFrameData[!current_read_id].processedFrame = processedFrame;
    CoreFrameData[!current_read_id].frameId = frameId;
    current_read_id = 1 - current_read_id;
}

接受：

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

