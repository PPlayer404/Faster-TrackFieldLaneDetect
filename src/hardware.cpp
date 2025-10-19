#define _USE_MATH_DEFINES
#include "hardware.hpp"
#include <pigpio.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstdint>

/* ==========================  ServoDriver  ========================== */
ServoDriver::ServoDriver(int pin, double min, double max, double mid)
    : pin_(pin), minAngle_(min), maxAngle_(max), midAngle_(mid)
{
    if (midAngle_ < minAngle_) midAngle_ = minAngle_;
    if (midAngle_ > maxAngle_) midAngle_ = maxAngle_;
}

void ServoDriver::init()
{
    gpioSetMode(pin_, PI_OUTPUT);
    setAngle(midAngle_);
    gpioDelay(500000);               // 500 ms 确保到位
    printf("舵机(GPIO%d)就绪，中位角度: %.1f°\n", pin_, midAngle_);
}

void ServoDriver::setAngle(double ang)
{
    ang = ang < minAngle_ ? minAngle_ : (ang > maxAngle_ ? maxAngle_ : ang);
    gpioServo(pin_, 500 + static_cast<int>(ang / 180.0 * 2000));
}

double ServoDriver::getMidAngle()
{
    return midAngle_;
}

/* ==========================  MotorDriver  ========================== */
MotorDriver::MotorDriver(int pin) : pin_(pin) {}

void MotorDriver::init()
{
    gpioSetMode(pin_, PI_OUTPUT);
    gpioSetPWMfrequency(pin_, 200);
    gpioSetPWMrange(pin_, 40000);
    unlock();
    printf("电机就绪\n");
}

void MotorDriver::unlock()
{
    gpioPWM(pin_, 10000);
    gpioDelay(2000000);              // 2 s 解锁
}

void MotorDriver::setSpeed(int duty)
{
    duty = duty < 9400 ? 9400 : (duty > 14000 ? 14000 : duty);
    gpioPWM(pin_, duty);
}

/* ==========================  StanleyController  ========================== */
StanleyController::StanleyController(double k, double L, double max)
    : k_(k), L_(L), maxSteer_(max) {}

double StanleyController::calcAngle(double e_y, double theta_e, double v)
{
    double theta_rad = theta_e * M_PI / 180.0;
    double delta = (v < 0.1) ? theta_rad : (theta_rad - std::atan2(k_ * e_y, v));
    delta *= 180.0 / M_PI;
    return delta < -maxSteer_ ? -maxSteer_ : (delta > maxSteer_ ? maxSteer_ : delta);
}

/* ==========================  ImuDriver  ========================== */
ImuDriver::ImuDriver()
    : serHdl_(-1), yaw_(0), pitch_(0), roll_(0), gpsSpd_(0),
      accX_(0), accY_(0), accZ_(0), bufIdx_(0),
      fusedSpeed_(0), imuSpeed_(0), kalman_v_(0), P_(1.0),
      lastValidGpsSpeed_(0.0), hasValidGps_(false),
      Q_(0.1), R_(1.0)  
{
    serHdl_ = serOpen(const_cast<char*>("/dev/ttyUSB0"), 115200, 0);
    if (serHdl_ < 0) {
        printf("IMU串口失败！查接线/波特率\n");
        std::exit(-1);
    }
    lastTime_ = std::chrono::high_resolution_clock::now();
}

void ImuDriver::readData()
{
    // 读取串口数据到缓冲区（大小66字节）
    while (serDataAvailable(serHdl_) > 0 && bufIdx_ < 66)
        frameBuf_[bufIdx_++] = static_cast<uint8_t>(serReadByte(serHdl_));

    parseFrame();

    auto now = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(now - lastTime_).count();
    lastTime_ = now;

    imuSpeed_ += accX_ * dt;

    double currentGps = gpsSpd_;

    kalman_v_ += accX_ * dt;
    P_ += Q_;

    if (currentGps > 0) {
        lastValidGpsSpeed_ = currentGps;
        hasValidGps_ = true;
        double K = P_ / (P_ + R_);
        kalman_v_ += K * (currentGps - kalman_v_);
        P_ = (1 - K) * P_;
    } else if (hasValidGps_) {
        double K = P_ / (P_ + R_);
        kalman_v_ += K * (lastValidGpsSpeed_ - kalman_v_);
        P_ = (1 - K) * P_;
    }
    fusedSpeed_ = kalman_v_;
}

double ImuDriver::getSpeed()
{
    return fusedSpeed_;
}

void ImuDriver::parseFrame()
{
    
    int last51 = -1, last53 = -1, last58 = -1;

   
    for (int i = bufIdx_ - 11; i >= 0; --i) {
        if (frameBuf_[i] != 0x55) continue; // 跳过非帧头

        uint8_t type = frameBuf_[i + 1];
        if (type != 0x51 && type != 0x53 && type != 0x58) continue;

        // 校验和检查
        uint8_t calcChk = 0;
        for (int j = 0; j < 10; ++j) {
            calcChk += frameBuf_[i + j];
        }
        if (calcChk != frameBuf_[i + 10]) continue;

        // 记录该类型的最新有效帧
        if (type == 0x51 && last51 == -1) last51 = i;
        else if (type == 0x53 && last53 == -1) last53 = i;
        else if (type == 0x58 && last58 == -1) last58 = i;

        // 三种类型都找到则提前退出
        if (last51 != -1 && last53 != -1 && last58 != -1) break;
    }

    // 解析找到的每种类型的最新帧
    auto parseType = [&](int startIdx) {
        if (startIdx == -1) return;

        uint8_t type = frameBuf_[startIdx + 1];
        switch (type) {
        case 0x51: { // 加速度
            int16_t ax = static_cast<int16_t>(frameBuf_[startIdx + 2] | (frameBuf_[startIdx + 3] << 8));
            int16_t ay = static_cast<int16_t>(frameBuf_[startIdx + 4] | (frameBuf_[startIdx + 5] << 8));
            int16_t az = static_cast<int16_t>(frameBuf_[startIdx + 6] | (frameBuf_[startIdx + 7] << 8));
            double accScale = 16.0 * 9.8 / 32768.0;
            accX_ = ax * accScale;
            accY_ = ay * accScale;
            accZ_ = az * accScale;
            break;
        }
        case 0x53: { // 姿态
            int16_t rollRaw  = static_cast<int16_t>(frameBuf_[startIdx + 4] | (frameBuf_[startIdx + 5] << 8));
            int16_t pitchRaw = static_cast<int16_t>(frameBuf_[startIdx + 2] | (frameBuf_[startIdx + 3] << 8));
            int16_t yawRaw   = static_cast<int16_t>(frameBuf_[startIdx + 6] | (frameBuf_[startIdx + 7] << 8));
            roll_  = rollRaw  / 32768.0 * 180.0;
            pitch_ = pitchRaw / 32768.0 * 180.0;
            yaw_   = yawRaw   / 32768.0 * 180.0;
            break;
        }
        case 0x58: { // GPS速度
            uint32_t gpsSpdRaw = static_cast<uint32_t>(frameBuf_[startIdx + 6]) |
                                 (static_cast<uint32_t>(frameBuf_[startIdx + 7]) << 8)  |
                                 (static_cast<uint32_t>(frameBuf_[startIdx + 8]) << 16) |
                                 (static_cast<uint32_t>(frameBuf_[startIdx + 9]) << 24);
            gpsSpd_ = gpsSpdRaw / 1000.0 / 3.6;
            break;
        }
        }
    };


    parseType(last51);
    parseType(last53);
    parseType(last58);

    bufIdx_ = 0;
}

void ImuDriver::close()
{
    if (serHdl_ >= 0) {
        serClose(serHdl_);
        serHdl_ = -1;
    }
}