#ifndef HARDWARE_HPP
#define HARDWARE_HPP
#include <chrono>
#ifdef _LINUX_
#include <pigpio.h>
#endif

// 舵机驱动类
class ServoDriver {
public:
    int pin_;          // 舵机GPIO引脚
    double minAngle_;  // 最小角度
    double maxAngle_;  // 最大角度

    ServoDriver(int pin, double min, double max);
    void init();
    void setAngle(double ang);
};

// 电机驱动类
class MotorDriver {
public:
    int pin_;  // 电机PWM引脚

    MotorDriver(int pin);
    void init();
    void unlock();
    void setSpeed(int duty);
};

// Stanley控制器
class StanleyController {
public:
    double k_;          // 横向偏差增益
    double L_;          // 车辆轴距（米）
    double maxSteer_;   // 最大转向角（°）

    StanleyController(double k, double L, double max);
    double calcAngle(double e_y, double theta_e, double v);
};

// IMU驱动类（带卡尔曼融合）
class ImuDriver {
public:
    int serHdl_;               // 串口句柄
    double yaw_;               // 航向角（°，陀螺输出）
    double gpsSpd_;            // GPS地速（m/s，原始测量值）
    double accX_;              // X轴加速度（m/s2，前进方向）
    double accY_;              // Y轴加速度（m/s2，横向）
    double accZ_;              // Z轴加速度（m/s2，垂直）
    uint8_t frameBuf_[32];     // 协议帧缓冲区
    int bufIdx_;               // 缓冲区索引

    // 速度融合相关
    double fusedSpeed_;        // 最终融合速度（对外提供）
    double imuSpeed_;          // IMU积分速度（备用）
    double kalman_v_;          // 卡尔曼滤波最优速度
    double P_;                 // 卡尔曼协方差（不确定性）
    const double Q_ = 0.01;    // 过程噪声（IMU漂移）
    const double R_ = 0.5;     // 测量噪声（GPS波动）
    double lastValidGpsSpeed_; // 保存上一次有效的GPS速度
    bool hasValidGps_;         // 标记是否曾收到过有效GPS数据
    std::chrono::high_resolution_clock::time_point lastTime_; // 上次采样时间

    ImuDriver();
    void readData();
    double getSpeed();
    void parseFrame();
    void close();
};

// 全局硬件对象声明
extern ServoDriver* servo;
extern MotorDriver* motor;
extern StanleyController* stanley;
extern ImuDriver* imu;

// 硬件初始化函数声明
void hardWareInit(void);
void stanleyControl(void);
void cleanup(void);

#endif


