#define _USE_MATH_DEFINES
#ifndef SERVO_MOTOR_CONTROL_HPP
#define SERVO_MOTOR_CONTROL_HPP

#include <pigpio.h>
#include <chrono>
#include <cstdint>
#include <cmath>

/* ===============  ServoDriver  =============== */
class ServoDriver {
public:
    ServoDriver(int pin, double min, double max, double mid);
    void init();
    void setAngle(double ang);
    double getMidAngle();

private:
    int pin_;
    double minAngle_, maxAngle_, midAngle_;
};

/* ===============  MotorDriver  =============== */
class MotorDriver {
public:
    MotorDriver(int pin);
    void init();
    void unlock();
    void setSpeed(int duty);

private:
    int pin_;
};

/* ===============  StanleyController  =============== */
class StanleyController {
public:
    StanleyController(double k, double L, double max);
    double calcAngle(double e_y, double theta_e, double v);

private:
    double k_, L_, maxSteer_;
};

/* ===============  ImuDriver  =============== */
class ImuDriver {
public:
    ImuDriver();
    void readData();
    double getSpeed();
    void parseFrame();
    void close();
    double getPitch() const { return pitch_; }
    double yaw_, pitch_, roll_, gpsSpd_;
    double accX_, accY_, accZ_;
    double fusedSpeed_, imuSpeed_, kalman_v_, P_;
private:
    int serHdl_;

    uint8_t frameBuf_[66];
    int bufIdx_;

    const double Q_ = 0.01;
    const double R_ = 0.5;
    double lastValidGpsSpeed_;
    bool hasValidGps_;
    std::chrono::high_resolution_clock::time_point lastTime_;
};

#endif // SERVO_MOTOR_CONTROL_HPP