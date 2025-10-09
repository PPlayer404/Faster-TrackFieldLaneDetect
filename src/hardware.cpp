#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef _WIN32
#include <pigpio.h>
#endif
#include <string.h>
#include <stdint.h>
#include "ThreadManager.hpp"

#ifdef _LINUX_

// 声明全局变量，让所有函数都能访问
extern ServoDriver* servo;
extern MotorDriver* motor;
extern StanleyController* stanley;
extern ImuDriver* imu;

// 舵机驱动：控制引脚、角度限幅、PWM转换
class ServoDriver {
public:
    int pin_;          // 舵机GPIO引脚
    double minAngle_;  // 最小角度
    double maxAngle_;  // 最大角度

    // 初始化引脚和角度范围
    ServoDriver(int pin, double min, double max) : pin_(pin), minAngle_(min), maxAngle_(max) {}

    // 引脚设为输出，舵机归中（85°直行），等待2秒
    void init() {
        gpioSetMode(pin_, PI_OUTPUT);
        setAngle(85);
        delay_ms(2000);
        printf("舵机就绪\n");
    }

    // 角度限幅后转PWM（500~2500us对应0~180°）
    void setAngle(double ang) {
        ang = ang < minAngle_ ? minAngle_ : (ang > maxAngle_ ? maxAngle_ : ang);
        gpioServo(pin_, 500 + (int)(ang / 180.0 * 2000));
    }
};

// 电机驱动：PWM配置、电调解锁、速度限幅
class MotorDriver {
public:
    int pin_;  // 电机PWM引脚

    // 初始化电机引脚
    MotorDriver(int pin) : pin_(pin) {}

    // 配置PWM（200Hz/0~40000），解锁电调
    void init() {
        gpioSetMode(pin_, PI_OUTPUT);
        gpioSetPWMfrequency(pin_, 200);
        gpioSetPWMrange(pin_, 40000);
        unlock();  // 电调必须解锁才响应
        printf("电机就绪\n");
    }

    // 发送解锁信号（10000占空比，延时2秒）
    void unlock() {
        gpioPWM(pin_, 10000);
        delay_ms(2000);
    }

    // 速度限幅（9400=最低，10000=最高）
    void setSpeed(int duty) {
        duty = duty < 9400 ? 9400 : (duty > 10000 ? 10000 : duty);
        gpioPWM(pin_, duty);
    }
};

// Stanley控制器：计算转向角（无人车横向控制）
class StanleyController {
public:
    double k_;          // 横向偏差增益（调参）
    double L_;          // 车辆轴距（米，实测）
    double maxSteer_;   // 最大转向角（°，匹配舵机）

    // 初始化算法参数
    StanleyController(double k, double L, double max) : k_(k), L_(L), maxSteer_(max) {}

    // 输入：横向偏差e_y、航向偏差theta_e、车速v，输出转向角
    double calcAngle(double e_y, double theta_e, double v) {
        double theta_rad = theta_e * M_PI / 180.0;
        double delta;

        // 低速仅修正航向，高速结合横向偏差
        delta = (v < 0.1) ? theta_rad : (theta_rad - atan2(k_ * e_y, v));

        delta = delta * 180.0 / M_PI;  // 弧度转角度
        return delta < -maxSteer_ ? -maxSteer_ : (delta > maxSteer_ ? maxSteer_ : delta);
    }
};

#include <chrono>

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

    // 构造函数
    ImuDriver()
        : yaw_(0), gpsSpd_(0), accX_(0), accY_(0), accZ_(0), bufIdx_(0),
        fusedSpeed_(0), imuSpeed_(0), kalman_v_(0), P_(1.0),
        lastValidGpsSpeed_(0.0), hasValidGps_(false)
    {
        serHdl_ = serialOpen("/dev/ttyUSB0", 115200);
        if (serHdl_ < 0) {
            printf("IMU串口失败！查接线/波特率\n");
            exit(-1);
        }
        lastTime_ = std::chrono::high_resolution_clock::now();
    }

    void readData() {
        // 读取串口数据到缓冲区，解析协议帧
        while (serialDataAvail(serHdl_) > 0 && bufIdx_ < 31) {
            frameBuf_[bufIdx_++] = (uint8_t)serialReadByte(serHdl_);
        }
        parseFrame();

        // 计算真实采样间隔 dt（秒）
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - lastTime_).count();
        lastTime_ = now;

        // IMU加速度积分（备用速度，不直接对外输出）
        imuSpeed_ += accX_ * dt;

        // 卡尔曼滤波（预测+更新，核心逻辑）
        double currentGps = gpsSpd_; // 当前GPS测量值

        // 步骤1：预测
        kalman_v_ += accX_ * dt;
        P_ += Q_;

        // 步骤2：更新
        if (currentGps > 0) {
            lastValidGpsSpeed_ = currentGps;
            hasValidGps_ = true;

            double K = P_ / (P_ + R_);
            kalman_v_ = kalman_v_ + K * (currentGps - kalman_v_);
            P_ = (1 - K) * P_;
        }
        else if (hasValidGps_) {
            double K = P_ / (P_ + R_);
            kalman_v_ = kalman_v_ + K * (lastValidGpsSpeed_ - kalman_v_);
            P_ = (1 - K) * P_;
        }

        fusedSpeed_ = kalman_v_;
    }

    double getSpeed() {
        return fusedSpeed_;
    }

    void parseFrame() {
        // 定位双帧头（0x55+0xAA）
        for (int i = 0; i < bufIdx_ - 1; i++) {
            if (frameBuf_[i] == 0x55 && frameBuf_[i + 1] == 0xAA) {
                memmove(frameBuf_, frameBuf_ + i, bufIdx_ - i);
                bufIdx_ -= i;
                break;
            }
        }

        if (bufIdx_ < 4) return;
        uint8_t frameAddr = frameBuf_[2];
        uint8_t dataLen = frameBuf_[3];
        uint8_t totalLen = 4 + dataLen + 1;

        if (bufIdx_ < totalLen) return;

        uint8_t calcChk = 0;
        for (int j = 2; j < 4 + dataLen; j++) calcChk += frameBuf_[j];
        if (calcChk != frameBuf_[4 + dataLen]) {
            memmove(frameBuf_, frameBuf_ + totalLen, bufIdx_ - totalLen);
            bufIdx_ -= totalLen;
            return;
        }

        switch (frameAddr) {
        case 0x30:
            memcpy(&yaw_, frameBuf_ + 12, 4);
            memcpy(&accX_, frameBuf_ + 16, 4);
            memcpy(&accY_, frameBuf_ + 20, 4);
            memcpy(&accZ_, frameBuf_ + 24, 4);
            break;
        case 0x31:
            memcpy(&gpsSpd_, frameBuf_ + 20, 4);
            break;
        }

        memmove(frameBuf_, frameBuf_ + totalLen, bufIdx_ - totalLen);
        bufIdx_ -= totalLen;
    }

    void close() {
        if (serHdl_ >= 0) {
            serialClose(serHdl_);
            serHdl_ = -1;
        }
    }
};

// 全局变量定义
ServoDriver* servo = nullptr;
MotorDriver* motor = nullptr;
StanleyController* stanley = nullptr;
ImuDriver* imu = nullptr;

/// @brief 主函数开始的时候调用，完成硬件初始化
void hardWareInit(void)
{
    // 初始化pigpio
    if (gpioInitialise() < 0) {
        printf("pigpio失败！先执行：sudo pigpiod\n");
        return;
    }

    // 创建对象实例
    servo = new ServoDriver(12, 40, 130);
    motor = new MotorDriver(13);
    stanley = new StanleyController(0.5, 0.3, 25);
    imu = new ImuDriver();

    // 初始化硬件
    servo->init();
    motor->init();
    motor->setSpeed(9600);
    printf("硬件就绪\n");
}

/// @brief 如果史坦纳发起进攻，一切都会好起来的 ————元首
void stanleyControl(void)
{
    // 检查对象是否已初始化
    if (!imu || !servo || !stanley) {
        printf("错误：硬件未初始化！\n");
        return;
    }

    // 读IMU数据（陀螺+GPS+加速度）
    imu->readData();

    // 控制参数（e_y=横向偏差模拟值，theta_e=航向偏差暂设0）
    double e_y = 0.1;
    double theta_e = 0.0;
    double v = imu->gpsSpd_ > 0 ? imu->gpsSpd_ : 1.5;  // GPS地速无效时用1.5m/s

    // 算转向角+控舵机
    double steer = stanley->calcAngle(e_y, theta_e, v);
    servo->setAngle(85 + steer);  // 85°为直行，加转向角

    // 打印核心数据（调试用）
    printf("航向=%.1f° | 地速=%.2f m/s | 加速度(X=%.2f,Y=%.2f,Z=%.2f) | 舵角=%.1f°\n",
        imu->yaw_, imu->gpsSpd_,
        imu->accX_, imu->accY_, imu->accZ_,
        85 + steer);
}

/// @brief 清理资源
void cleanup(void)
{
    if (imu) {
        imu->close();
        delete imu;
        imu = nullptr;
    }
    if (servo) {
        delete servo;
        servo = nullptr;
    }
    if (motor) {
        delete motor;
        motor = nullptr;
    }
    if (stanley) {
        delete stanley;
        stanley = nullptr;
    }

    gpioTerminate();
}

#else

// Windows端空接口实现
class ServoDriver {
public:
    int pin_;
    double minAngle_;
    double maxAngle_;

    ServoDriver(int pin, double min, double max) : pin_(pin), minAngle_(min), maxAngle_(max) {}

    void init() {
        printf("[模拟] 舵机初始化 引脚:%d 角度范围:%.1f~%.1f\n", pin_, minAngle_, maxAngle_);
        setAngle(85);
        delay_ms(2000);
        printf("舵机就绪\n");
    }

    void setAngle(double ang) {
        ang = ang < minAngle_ ? minAngle_ : (ang > maxAngle_ ? maxAngle_ : ang);
        printf("[模拟] 设置舵机角度: %.1f°\n", ang);
    }
};

class MotorDriver {
public:
    int pin_;

    MotorDriver(int pin) : pin_(pin) {}

    void init() {
        printf("[模拟] 电机初始化 引脚:%d\n", pin_);
        unlock();
        printf("电机就绪\n");
    }

    void unlock() {
        printf("[模拟] 电调解锁\n");
        delay_ms(2000);
    }

    void setSpeed(int duty) {
        duty = duty < 9400 ? 9400 : (duty > 10000 ? 10000 : duty);
        printf("[模拟] 设置电机速度: %d\n", duty);
    }
};

class StanleyController {
public:
    double k_;
    double L_;
    double maxSteer_;

    StanleyController(double k, double L, double max) : k_(k), L_(L), maxSteer_(max) {}

    double calcAngle(double e_y, double theta_e, double v) {
        printf("[模拟] Stanley控制计算: e_y=%.2f, theta_e=%.1f°, v=%.2f m/s\n", e_y, theta_e, v);

        double theta_rad = theta_e * M_PI / 180.0;
        double delta;

        delta = (v < 0.1) ? theta_rad : (theta_rad - atan2(k_ * e_y, v));
        delta = delta * 180.0 / M_PI;

        double result = delta < -maxSteer_ ? -maxSteer_ : (delta > maxSteer_ ? maxSteer_ : delta);
        printf("[模拟] 计算转向角: %.1f°\n", result);
        return result;
    }
};

#include <chrono>
#include <random>

class ImuDriver {
public:
    int serHdl_;
    double yaw_;
    double gpsSpd_;
    double accX_;
    double accY_;
    double accZ_;
    uint8_t frameBuf_[32];
    int bufIdx_;

    double fusedSpeed_;
    double imuSpeed_;
    double kalman_v_;
    double P_;
    const double Q_ = 0.01;
    const double R_ = 0.5;
    double lastValidGpsSpeed_;
    bool hasValidGps_;
    std::chrono::high_resolution_clock::time_point lastTime_;

    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    ImuDriver()
        : serHdl_(-1),                    // 显式初始化
        yaw_(0), gpsSpd_(0), accX_(0), accY_(0), accZ_(0),
        bufIdx_(0),
        frameBuf_{ 0 },                   // 数组清零初始化
        fusedSpeed_(0), imuSpeed_(0), kalman_v_(0), P_(1.0),
        lastValidGpsSpeed_(0.0), hasValidGps_(false),
        distribution(0.0, 0.1)
    {
        printf("[模拟] IMU初始化\n");
        lastTime_ = std::chrono::high_resolution_clock::now();

        // 初始化随机值
        yaw_ = 45.0;
        gpsSpd_ = 1.5;
        accX_ = 0.1;
        accY_ = 0.05;
        accZ_ = 9.8;
    }

    void readData() {
        // 模拟数据更新
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - lastTime_).count();
        lastTime_ = now;

        // 添加一些随机变化模拟真实传感器
        yaw_ += distribution(generator);
        gpsSpd_ += distribution(generator) * 0.5;
        accX_ += distribution(generator) * 0.1;
        accY_ += distribution(generator) * 0.1;

        // 简单卡尔曼滤波模拟
        kalman_v_ = gpsSpd_;
        fusedSpeed_ = kalman_v_;

        printf("[模拟] IMU数据更新: 航向=%.1f°, 地速=%.2f m/s\n", yaw_, gpsSpd_);
    }

    double getSpeed() {
        return fusedSpeed_;
    }

    void parseFrame() {
        // 空实现 - Windows端不需要解析真实串口数据
    }

    void close() {
        printf("[模拟] IMU关闭\n");
    }
};

// Windows端全局变量定义
ServoDriver* servo = nullptr;
MotorDriver* motor = nullptr;
StanleyController* stanley = nullptr;
ImuDriver* imu = nullptr;

/// @brief 主函数开始的时候调用，完成硬件初始化
void hardWareInit(void)
{
    printf("[模拟] 硬件初始化开始\n");

    // 创建对象实例
    servo = new ServoDriver(12, 40, 130);
    motor = new MotorDriver(13);
    stanley = new StanleyController(0.5, 0.3, 25);
    imu = new ImuDriver();

    // 初始化硬件
    servo->init();
    motor->init();
    motor->setSpeed(9600);
    printf("硬件就绪\n");
}

/// @brief 如果史坦纳发起进攻，一切都会好起来的 ————元首
void stanleyControl(void)
{
    if (!imu || !servo || !stanley) {
        printf("错误：硬件未初始化！\n");
        return;
    }

    imu->readData();

    double e_y = 0.1;
    double theta_e = 0.0;
    double v = imu->gpsSpd_ > 0 ? imu->gpsSpd_ : 1.5;

    double steer = stanley->calcAngle(e_y, theta_e, v);
    servo->setAngle(85 + steer);

    printf("航向=%.1f° | 地速=%.2f m/s | 加速度(X=%.2f,Y=%.2f,Z=%.2f) | 舵角=%.1f°\n",
        imu->yaw_, imu->gpsSpd_,
        imu->accX_, imu->accY_, imu->accZ_,
        85 + steer);
}

/// @brief 清理资源
void cleanup(void)
{
    printf("[模拟] 清理资源\n");

    if (imu) {
        imu->close();
        delete imu;
        imu = nullptr;
    }
    if (servo) {
        delete servo;
        servo = nullptr;
    }
    if (motor) {
        delete motor;
        motor = nullptr;
    }
    if (stanley) {
        delete stanley;
        stanley = nullptr;
    }
}

#endif