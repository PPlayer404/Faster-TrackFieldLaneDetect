#include "World.hpp"
#include "Cluster.hpp"
#include "mode.hpp"
#include <climits>
#include "ThreadManager.hpp"
#include <chrono>

#define DEFAULT 0
#define RECHECK 1
#define RESET 2

/// @brief 在协程中实现延时，单位毫秒，非阻塞，原理是设置一个定时器，然后协程挂起，等定时器到期后再恢复
/// 呃，大概吧，我也不是很懂协程，反正能用就行
#define Coroutine_delay(ms) decision_set_alarm(ms);co_await std::suspend_always{}

struct LaneDescriptor
{
    int peakX;          // 起始生长点（x坐标）
    double mainAngle;   // 主导角度（弧度）
    bool leftLoss;
    bool rightLoss;
};

// --- 修改开始 ---
struct LaneKalmanFilter
{
    cv::KalmanFilter kf_left;
    cv::KalmanFilter kf_right;
    bool left_initialized;
    bool right_initialized;
    double last_left_angle;
    double last_right_angle;
    int last_left_x;
    int last_right_x;

    // 新增：用于内部计时的成员
    std::chrono::steady_clock::time_point last_update_time;
    // 新增：用于存储基础过程噪声矩阵，方便按dt缩放
    cv::Mat processNoiseCov_base;

    LaneKalmanFilter() : left_initialized(false), right_initialized(false),
        last_left_angle(0), last_right_angle(0),
        last_left_x(0), last_right_x(0),
        last_update_time{} { // 初始化时间戳为“零”

        // --- 参数重调 ---
        // 注意：这些参数现在是基于“每秒”的物理单位
        const float R = 3.0f;          // 测量方差 (pixel^2)，与时间无关，保持不变
        const float Qp_base = 10.0f;   // 位置过程噪声 (pixel^2/s)
        const float Qv_base = 100.0f;  // 速度过程噪声 (pixel^2/s^3)
        const float P0_pos = 5.0f;     // 初始位置不确定度 (pixel^2)
        const float P0_vel = 100.0f;   // 初始速度不确定度 (pixel^2/s^2)

        // 存储基础过程噪声矩阵
        processNoiseCov_base = cv::Mat::eye(2, 2, CV_32F); // 创建一个2x2的float类型单位矩阵
        processNoiseCov_base.at<float>(0, 0) = Qp_base;
        processNoiseCov_base.at<float>(1, 1) = Qv_base;

        for (auto kf : { &kf_left, &kf_right }) {
            kf->init(2, 1, 0);
            // 状态转移矩阵F将动态设置，这里先初始化为单位矩阵
            kf->transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1);
            kf->measurementMatrix = (cv::Mat_<float>(1, 2) << 1, 0);

            // 初始过程噪声为0，将在每次预测前根据dt更新
            cv::setIdentity(kf->processNoiseCov, cv::Scalar(0.0f));

            cv::setIdentity(kf->measurementNoiseCov, cv::Scalar(R));

            // 设置初始协方差矩阵P0
            cv::setIdentity(kf->errorCovPost, cv::Scalar(0.0f));
            kf->errorCovPost.at<float>(0, 0) = P0_pos;
            kf->errorCovPost.at<float>(1, 1) = P0_vel;
        }
    }
};

LaneKalmanFilter tracker;

/// @brief 使用卡尔曼滤波进行车道线跟踪和预测 (基于角度的左右识别)
/// @param lanes 输入的车道线聚类描述子集合
/// @param tracker 车道线卡尔曼滤波器跟踪器
/// @param left_mode 左线工作模式
/// @param right_mode 右线工作模式
/// @return 返回跟踪后的车道线描述子（包含左右车道线，索引左0右1）
std::vector<LaneDescriptor> kalmanLanes(std::vector<ClusterDescriptor>& lanes, LaneKalmanFilter& tracker, int left_mode, int right_mode)
{
    std::vector<LaneDescriptor> result;
    static const int IMAGE_CENTER = 120;
    // --- 修改开始 ---
    // 新增：基于时间的速度衰减常数
    static const float VELOCITY_DECAY_LAMBDA = 2.0f; // 衰减时间常数约为 1/lambda = 0.5秒
    // --- 修改结束 ---
    static const double ANGLE_THRESHOLD = 0.5;
    static const double POSITION_WEIGHT = 0.7;
    static const double ANGLE_WEIGHT = 0.3;
    static const double MIN_SCORE_THRESHOLD = 0.5;
    static const int MATCH_THRESHOLD = 45;

    // --- 修改开始 ---
    // 1. 计算时间增量 dt
    auto now = std::chrono::steady_clock::now();
    double dt = 1.0 / 30.0; // 默认dt，假设30fps
    if (tracker.last_update_time != std::chrono::steady_clock::time_point{}) {
        dt = std::chrono::duration<double>(now - tracker.last_update_time).count();
        // 将dt限制在合理范围内，防止因帧率骤降导致预测发散
        dt = std::max(0.001, std::min(dt, 0.2)); // dt在1ms到200ms之间
    }
    // --- 修改结束 ---

    std::vector<ClusterDescriptor> left_candidates;
    std::vector<ClusterDescriptor> right_candidates;
    if (!lanes.empty()) {
        for (const auto& lane : lanes) {
            if (std::abs(lane.mainAngle) < 1e-4) continue;
            if (lane.mainAngle < 0) left_candidates.push_back(lane);
            else if (lane.mainAngle > 0) right_candidates.push_back(lane);
        }
        std::sort(left_candidates.begin(), left_candidates.end(), [](const ClusterDescriptor& a, const ClusterDescriptor& b) { return a.peakX > b.peakX; });
        std::sort(right_candidates.begin(), right_candidates.end(), [](const ClusterDescriptor& a, const ClusterDescriptor& b) { return a.peakX < b.peakX; });
    }

    bool reset_left = false;
    bool reset_right = false;
    if (left_mode == RECHECK) {
        if (!lanes.empty() && tracker.left_initialized && !left_candidates.empty()) {
            float current_left_x = tracker.kf_left.statePost.at<float>(0);
            float candidate_left_x = left_candidates[0].peakX;
            if (std::abs(candidate_left_x - current_left_x) > 40) reset_left = true;
        }
    }
    else if (left_mode == RESET) { reset_left = true; }
    if (right_mode == RECHECK) {
        if (!lanes.empty() && tracker.right_initialized && !right_candidates.empty()) {
            float current_right_x = tracker.kf_right.statePost.at<float>(0);
            float candidate_right_x = right_candidates[0].peakX;
            if (std::abs(candidate_right_x - current_right_x) > 40) reset_right = true;
        }
    }
    else if (right_mode == RESET) { reset_right = true; }

    if (reset_left) tracker.left_initialized = false;
    if (reset_right) tracker.right_initialized = false;
    // 注意：我们不在这里重置 last_update_time，因为计时器应该持续运行

    if (lanes.empty()) {
        LaneDescriptor left_lane, right_lane;
        if (tracker.left_initialized) {
            // --- 修改开始 ---
            // 2. 在预测前更新卡尔曼滤波器矩阵
            tracker.kf_left.transitionMatrix.at<float>(0, 1) = static_cast<float>(dt);
            tracker.kf_left.processNoiseCov = tracker.processNoiseCov_base * static_cast<float>(dt);
            // --- 修改结束 ---

            tracker.kf_left.predict();
            // --- 修改开始 ---
            // 3. 使用基于时间的指数衰减
            tracker.kf_left.statePost.at<float>(1) *= std::exp(-VELOCITY_DECAY_LAMBDA * dt);
            // --- 修改结束 ---
            cv::Mat left_state = tracker.kf_left.statePost;
            left_lane.peakX = left_state.at<float>(0);
            left_lane.mainAngle = tracker.last_left_angle;
            left_lane.leftLoss = true;
        }
        else { left_lane.leftLoss = true; }

        if (tracker.right_initialized) {
            // --- 修改开始 ---
            // 2. 在预测前更新卡尔曼滤波器矩阵
            tracker.kf_right.transitionMatrix.at<float>(0, 1) = static_cast<float>(dt);
            tracker.kf_right.processNoiseCov = tracker.processNoiseCov_base * static_cast<float>(dt);
            // --- 修改结束 ---

            tracker.kf_right.predict();
            // --- 修改开始 ---
            // 3. 使用基于时间的指数衰减
            tracker.kf_right.statePost.at<float>(1) *= std::exp(-VELOCITY_DECAY_LAMBDA * dt);
            // --- 修改结束 ---
            cv::Mat right_state = tracker.kf_right.statePost;
            right_lane.peakX = right_state.at<float>(0);
            right_lane.mainAngle = tracker.last_right_angle;
            right_lane.rightLoss = true;
        }
        else { right_lane.rightLoss = true; }

        result.push_back(left_lane);
        result.push_back(right_lane);

        // --- 修改开始 ---
        // 4. 在函数返回前更新时间戳
        tracker.last_update_time = now;
        // --- 修改结束 ---
        return result;
    }

    int left_index = -1, right_index = -1;
    LaneDescriptor left_lane, right_lane;

    if (tracker.left_initialized) {
        float best_score = -1;
        for (size_t i = 0; i < left_candidates.size(); ++i) {
            float dist = std::abs(left_candidates[i].peakX - tracker.last_left_x);
            if (dist > MATCH_THRESHOLD) continue;
            float dist_score = 1.0f - dist / MATCH_THRESHOLD;
            float angle_diff = std::abs(left_candidates[i].mainAngle - tracker.last_left_angle);
            float angle_score = (angle_diff < ANGLE_THRESHOLD) ? (1.0f - angle_diff / ANGLE_THRESHOLD) : 0;
            float total_score = POSITION_WEIGHT * dist_score + ANGLE_WEIGHT * angle_score;
            if (total_score > best_score && total_score >= MIN_SCORE_THRESHOLD) {
                best_score = total_score;
                left_index = i;
            }
        }
    }
    else { if (!left_candidates.empty()) left_index = 0; }

    if (tracker.right_initialized) {
        float best_score = -1;
        for (size_t i = 0; i < right_candidates.size(); ++i) {
            float dist = std::abs(right_candidates[i].peakX - tracker.last_right_x);
            if (dist > MATCH_THRESHOLD) continue;
            float dist_score = 1.0f - dist / MATCH_THRESHOLD;
            float angle_diff = std::abs(right_candidates[i].mainAngle - tracker.last_right_angle);
            float angle_score = (angle_diff < ANGLE_THRESHOLD) ? (1.0f - angle_diff / ANGLE_THRESHOLD) : 0;
            float total_score = POSITION_WEIGHT * dist_score + ANGLE_WEIGHT * angle_score;
            if (total_score > best_score && total_score >= MIN_SCORE_THRESHOLD) {
                best_score = total_score;
                right_index = i;
            }
        }
    }
    else { if (!right_candidates.empty()) right_index = 0; }

    if (left_index != -1) {
        cv::Mat measurement = (cv::Mat_<float>(1, 1) << left_candidates[left_index].peakX);
        if (!tracker.left_initialized) {
            tracker.kf_left.statePre.at<float>(0) = left_candidates[left_index].peakX;
            tracker.kf_left.statePre.at<float>(1) = 0; // 初始速度为0 (像素/秒)
            tracker.kf_left.statePost = tracker.kf_left.statePre.clone();
            tracker.left_initialized = true;
            tracker.last_left_x = left_candidates[left_index].peakX;
        }
        else {
            // --- 修改开始 ---
            // 2. 在预测前更新卡尔曼滤波器矩阵
            tracker.kf_left.transitionMatrix.at<float>(0, 1) = static_cast<float>(dt);
            tracker.kf_left.processNoiseCov = tracker.processNoiseCov_base * static_cast<float>(dt);
            // --- 修改结束 ---

            tracker.kf_left.predict();
            tracker.kf_left.correct(measurement);
            tracker.last_left_x = left_candidates[left_index].peakX;
        }
        cv::Mat left_state = tracker.kf_left.statePost;
        left_lane.peakX = left_state.at<float>(0);
        left_lane.mainAngle = left_candidates[left_index].mainAngle;
        tracker.last_left_angle = left_candidates[left_index].mainAngle;
        left_lane.leftLoss = false;
    }
    else if (tracker.left_initialized) {
        // --- 修改开始 ---
        // 2. 在预测前更新卡尔曼滤波器矩阵
        tracker.kf_left.transitionMatrix.at<float>(0, 1) = static_cast<float>(dt);
        tracker.kf_left.processNoiseCov = tracker.processNoiseCov_base * static_cast<float>(dt);
        // --- 修改结束 ---

        tracker.kf_left.predict();
        // --- 修改开始 ---
        // 3. 使用基于时间的指数衰减
        tracker.kf_left.statePost.at<float>(1) *= std::exp(-VELOCITY_DECAY_LAMBDA * dt);
        // --- 修改结束 ---
        cv::Mat left_state = tracker.kf_left.statePost;
        left_lane.peakX = left_state.at<float>(0);
        left_lane.mainAngle = tracker.last_left_angle;
        left_lane.leftLoss = true;
    }
    else { left_lane.leftLoss = true; }

    if (right_index != -1) {
        cv::Mat measurement = (cv::Mat_<float>(1, 1) << right_candidates[right_index].peakX);
        if (!tracker.right_initialized) {
            tracker.kf_right.statePre.at<float>(0) = right_candidates[right_index].peakX;
            tracker.kf_right.statePre.at<float>(1) = 0; // 初始速度为0 (像素/秒)
            tracker.kf_right.statePost = tracker.kf_right.statePre.clone();
            tracker.right_initialized = true;
            tracker.last_right_x = right_candidates[right_index].peakX;
        }
        else {
            // --- 修改开始 ---
            // 2. 在预测前更新卡尔曼滤波器矩阵
            tracker.kf_right.transitionMatrix.at<float>(0, 1) = static_cast<float>(dt);
            tracker.kf_right.processNoiseCov = tracker.processNoiseCov_base * static_cast<float>(dt);
            // --- 修改结束 ---

            tracker.kf_right.predict();
            tracker.kf_right.correct(measurement);
            tracker.last_right_x = right_candidates[right_index].peakX;
        }
        cv::Mat right_state = tracker.kf_right.statePost;
        right_lane.peakX = right_state.at<float>(0);
        right_lane.mainAngle = right_candidates[right_index].mainAngle;
        tracker.last_right_angle = right_candidates[right_index].mainAngle;
        right_lane.rightLoss = false;
    }
    else if (tracker.right_initialized) {
        // --- 修改开始 ---
        // 2. 在预测前更新卡尔曼滤波器矩阵
        tracker.kf_right.transitionMatrix.at<float>(0, 1) = static_cast<float>(dt);
        tracker.kf_right.processNoiseCov = tracker.processNoiseCov_base * static_cast<float>(dt);
        // --- 修改结束 ---

        tracker.kf_right.predict();
        // --- 修改开始 ---
        // 3. 使用基于时间的指数衰减
        tracker.kf_right.statePost.at<float>(1) *= std::exp(-VELOCITY_DECAY_LAMBDA * dt);
        // --- 修改结束 ---
        cv::Mat right_state = tracker.kf_right.statePost;
        right_lane.peakX = right_state.at<float>(0);
        right_lane.mainAngle = tracker.last_right_angle;
        right_lane.rightLoss = true;
    }
    else { right_lane.rightLoss = true; }

    result.push_back(left_lane);
    result.push_back(right_lane);

#ifdef KALMAN_DEBUG
    // Debug visualization code remains the same
    cv::Mat vis = cv::Mat::zeros(120, 240, CV_8UC3);
    for (const auto& lane : lanes) {
        cv::Scalar color = (lane.mainAngle < 0) ? cv::Scalar(100, 100, 255) : cv::Scalar(100, 255, 100);
        cv::circle(vis, cv::Point(lane.peakX, 60), 3, color, -1);
    }
    if (tracker.left_initialized) {
        cv::circle(vis, cv::Point(left_lane.peakX, 60), 5, cv::Scalar(255, 0, 0), -1);
        cv::putText(vis, left_lane.leftLoss ? "L:pred" : "L:det", cv::Point(left_lane.peakX - 20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 0));
    }
    if (tracker.right_initialized) {
        cv::circle(vis, cv::Point(right_lane.peakX, 60), 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(vis, right_lane.rightLoss ? "R:pred" : "R:det", cv::Point(right_lane.peakX - 20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255));
    }
    cv::imshow("Lane Tracking", vis);
    cv::waitKey(1);
#endif

    // --- 修改开始 ---
    // 4. 在函数返回前更新时间戳
    tracker.last_update_time = now;
    // --- 修改结束 ---

    return result;
}

/// @brief dx方向：中线在图像中右偏为正，左偏为负； dAngle方向：车道线向左倾斜为正，右倾斜为负，0°垂直
/// 内部带有针对不同情况的处理算法，当两侧车道线均丢失时，返回上次的中线位置，
/// 单侧丢线时，基于存活车道线，车道线宽度和正反逆透视估计中线位置，透视矩阵为静态变量，开启WITH_IMSHOW之后可以自行在Lane.cpp中矫正逆透视矩阵再复制回来
/// @param lanes 车道线描述子集合，索引左0右1
/// @return 中线描述子
LaneDescriptor getMiddleLane(std::vector<LaneDescriptor>& lanes)
{
    constexpr int IMG_W = 240;
    constexpr int IMG_H = 120;
    constexpr int REF_Y = 60;
    constexpr int CTR_X = IMG_W / 2;
    constexpr int BOTTOM_Y = IMG_H - 1;
    constexpr double OFFSET_DIST = 92.0;

    static LaneDescriptor lastMiddleLane = { 0, 0 };
    static cv::Mat ipm_mat_inv;
    static bool initialized = false;

    if (!initialized) {
        ipm_mat_inv = ipm_mat.inv();
        initialized = true;
    }

    if (lanes.size() < 2) return lastMiddleLane;

    bool leftLoss = lanes[0].leftLoss;
    bool rightLoss = lanes[1].rightLoss;

    if (leftLoss && rightLoss) {
        return lastMiddleLane;
    }

    if (!leftLoss && !rightLoss) {
        auto& lc = lanes[0];
        auto& rc = lanes[1];

        auto safeTan = [](double a) {
            double t = std::tan(a);
            return std::abs(t) < 1e-4 ? (t >= 0 ? 1e-4 : -1e-4) : t;
            };

        double tanL = safeTan(lc.mainAngle);
        double tanR = safeTan(rc.mainAngle);

        double dyBottom = BOTTOM_Y - REF_Y;
        double xlBottom = lc.peakX + dyBottom / tanL;
        double xrBottom = rc.peakX + dyBottom / tanR;

        double midXBottom = (xlBottom + xrBottom) * 0.5;
        double midXRef = (lc.peakX + rc.peakX) * 0.5;

        double dx = midXRef - midXBottom;
        double dy = REF_Y - BOTTOM_Y;
        double angle = std::atan2(dx, dy);

        lastMiddleLane = { int(std::round(midXBottom - CTR_X)), angle };
        return lastMiddleLane;
    }

    LaneDescriptor validLane = leftLoss ? lanes[1] : lanes[0];
    double offsetSign = leftLoss ? 1.0 : -1.0;

    double tanValid = std::tan(validLane.mainAngle);
    if (std::abs(tanValid) < 1e-4) {
        tanValid = validLane.mainAngle >= 0 ? 1e-4 : -1e-4;
    }

    cv::Point2d p1_img(validLane.peakX, REF_Y);
    cv::Point2d p2_img(validLane.peakX + (BOTTOM_Y - REF_Y) / tanValid, BOTTOM_Y);

    std::vector<cv::Point2d> img_points = { p1_img, p2_img };
    std::vector<cv::Point2d> bev_points;
    cv::perspectiveTransform(img_points, bev_points, ipm_mat);

    cv::Point2d dir_bev = bev_points[1] - bev_points[0];
    double length = std::sqrt(dir_bev.x * dir_bev.x + dir_bev.y * dir_bev.y);
    if (length > 1e-4) {
        dir_bev.x /= length;
        dir_bev.y /= length;
    }

    cv::Point2d normal_bev(-dir_bev.y, dir_bev.x);

    cv::Point2d mid_p1_bev = bev_points[0] + normal_bev * offsetSign * OFFSET_DIST;
    cv::Point2d mid_p2_bev = bev_points[1] + normal_bev * offsetSign * OFFSET_DIST;

    std::vector<cv::Point2d> mid_bev_points = { mid_p1_bev, mid_p2_bev };
    std::vector<cv::Point2d> mid_img_points;
    cv::perspectiveTransform(mid_bev_points, mid_img_points, ipm_mat_inv);

    cv::Point2d mid_p1_img = mid_img_points[0];
    cv::Point2d mid_p2_img = mid_img_points[1];

    double dx_img = mid_p2_img.x - mid_p1_img.x;
    double dy_img = mid_p2_img.y - mid_p1_img.y;

    double midXBottom = mid_p1_img.x + (BOTTOM_Y - mid_p1_img.y) * dx_img / dy_img;
    double midXRef = mid_p1_img.x + (REF_Y - mid_p1_img.y) * dx_img / dy_img;

    double dx = midXRef - midXBottom;
    double dy = REF_Y - BOTTOM_Y;
    double angle = std::atan2(dx, dy);

    lastMiddleLane = { int(std::round(midXBottom - CTR_X)), angle };
    return lastMiddleLane;
}

/// @brief  在输入图像上绘制表示车道中线的红色线段，用于可视化调试
/// @param  frame 目标画布（任意分辨率，函数内部按比例缩放）
/// @param  angle 车道透视角（rad，左正右负，0 为垂直）
/// @param  dX    中线底边相对图像中心的横向偏移（像素，右正左负）
void drawMiddleLines(cv::Mat& frame, float angle, int dX)
{
    int h = frame.rows;
    int w = frame.cols;

    // 裁剪参数
    const double cropStartRatio = 2.0 / 5.0;
    const double cropEndRatio = 5.0 / 6.0;
    int cropStartY = cvRound(h * cropStartRatio);
    int cropHeight = cvRound(h * cropEndRatio) - cropStartY;

    // 缩放比例
    float scaleX = w / 240.0f;
    float scaleY = cropHeight / 120.0f;

    // 小图底部中心映射到原图
    int bottomCenterX = w / 2;
    int bottomY = cropStartY + cropHeight - 1;

    // 应用偏移量dX（小图底部的x偏移）
    bottomCenterX += dX * scaleX;

    // 计算线段长度
    int lineLength = h / 5*2;

    // 计算终点坐标（调整斜率计算）
    int topX = bottomCenterX - lineLength * std::tan(angle) * (scaleX / scaleY);
    int topY = bottomY - lineLength;

    // 绘制线段
    cv::line(frame,
        cv::Point(bottomCenterX, bottomY),
        cv::Point(topX, topY),
        cv::Scalar(0, 0, 255),
        8);
}

/// @brief 使用逆透视矩阵对偏航角进行校正和滤波,车道中线向左倾斜为正，右倾斜为负，这个函数应当在控制器输入之前调用
/// @param peakX 输入的车道中线peakX（int类型）
/// @param angle 输入的车道中线角度（float类型，弧度制）
/// @param reset 是否重置滤波器内部状态
/// @return 校正之后的车道中线描述子
LaneDescriptor middleLaneFilter(int peakX, float angle, bool reset = false)
{
    constexpr int IMG_W = 240;
    constexpr int IMG_H = 120;
    constexpr int REF_Y = 60;
    constexpr int CTR_X = IMG_W / 2;
    constexpr int BOTTOM_Y = IMG_H - 1;

    // 滤波器参数 - 基于25fps帧率和7Hz截止频率
    constexpr double FS = 25.0;          // 采样频率 (Hz)
    constexpr double FC_PEAKX = 12;      // peakX的截止频率 (Hz) - 稍微轻一点
    constexpr double FC_ANGLE = 5.5;     // 角度的截止频率 (Hz) - 稍微重一点
    constexpr double DT = 1.0 / FS;      // 采样时间 (s)

    // 计算滤波器系数
    constexpr double ALPHA_PEAKX = 2.0 * CV_PI * FC_PEAKX * DT / (1.0 + 2.0 * CV_PI * FC_PEAKX * DT);
    constexpr double ALPHA_ANGLE = 2.0 * CV_PI * FC_ANGLE * DT / (1.0 + 2.0 * CV_PI * FC_ANGLE * DT);
    static cv::Mat ipm_mat_inv;
    static bool initialized = false;

    // 滤波器状态
    static double filtered_peakX = 0.0;
    static double filtered_angle = 0.0;
    static bool first_run = true;

    if (!initialized)
    {
        ipm_mat_inv = ipm_mat.inv();
        initialized = true;
    }

    // 重置滤波器状态
    if (reset || first_run)
    {
        filtered_peakX = peakX;  // 使用输入的peakX初始化
        filtered_angle = angle;  // 使用输入的angle初始化
        first_run = false;

        // 即使重置，也进行逆透视变换
        double midXBottom = peakX + CTR_X;  // 替换为输入的peakX
        double midXRef = midXBottom + (REF_Y - BOTTOM_Y) * std::tan(angle);  // 替换为输入的angle

        cv::Point2d bottom_img(midXBottom, BOTTOM_Y);
        cv::Point2d ref_img(midXRef, REF_Y);

        std::vector<cv::Point2d> img_points = { bottom_img, ref_img };
        std::vector<cv::Point2d> bev_points;
        cv::perspectiveTransform(img_points, bev_points, ipm_mat);

        cv::Point2d dir_bev = bev_points[1] - bev_points[0];
        double bev_angle = -std::atan2(dir_bev.x, dir_bev.y);

        filtered_angle = bev_angle;
        return LaneDescriptor{ (int)filtered_peakX, filtered_angle };
    }

    // 计算图像坐标系中的中线底部点和参考点（使用输入的peakX和angle）
    double midXBottom = peakX + CTR_X;  // 替换为输入的peakX
    double midXRef = midXBottom + (REF_Y - BOTTOM_Y) * std::tan(angle);  // 替换为输入的angle

    // 定义图像坐标系中的两个点
    cv::Point2d bottom_img(midXBottom, BOTTOM_Y);
    cv::Point2d ref_img(midXRef, REF_Y);

    // 变换到BEV坐标系
    std::vector<cv::Point2d> img_points = { bottom_img, ref_img };
    std::vector<cv::Point2d> bev_points;
    cv::perspectiveTransform(img_points, bev_points, ipm_mat);

    // 计算BEV坐标系中的中线方向向量
    cv::Point2d dir_bev = bev_points[1] - bev_points[0];
    double bev_angle = -std::atan2(dir_bev.x, dir_bev.y);

    // 角度周期性校正（核心修复逻辑保留）
    double angle_diff = bev_angle - filtered_angle;
    angle_diff = std::atan2(std::sin(angle_diff), std::cos(angle_diff));
    double corrected_bev_angle = filtered_angle + angle_diff;

    // 应用一阶低通滤波器（使用输入的peakX和校正后的角度）
    filtered_peakX = ALPHA_PEAKX * peakX + (1.0 - ALPHA_PEAKX) * filtered_peakX;  // 替换为输入的peakX
    filtered_angle = ALPHA_ANGLE * corrected_bev_angle + (1.0 - ALPHA_ANGLE) * filtered_angle;

    return LaneDescriptor{ (int)filtered_peakX, filtered_angle };
}

//生产者，前端无锁
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

void World::updateIMU(float angle)
{
    bufFront_.imuAngle = angle;
}

/// @brief 构造函数，给一个初始值
World::World()
{
    WorldBuf init;
    init.currentSpeed = 0.0f;
    init.imuAngle = 0;
    // yolos 默认空，就不填了

    bufFront_ = init;
    bufBack_ = init;
}

/// @brief 更新维护世界模型，双缓冲 + 计算 + 返回快照，这个只融合传感器数据和滤波，决策逻辑在协程中
WorldSnapshot World::dataSync()
{
    static int left_loss_count = 0;
    static int right_loss_count = 0;
    const int MAX_LOSS_COUNT = 3; // 最大丢线帧数阈值
    //锁内指针交换
    {
        std::lock_guard<std::mutex> lk(mtx_);
        std::swap(bufFront_, bufBack_);
        ++frameId_;
    }  // 锁已释放

    //无锁，用的数据是bufBack_,先实例化一个快照
    WorldSnapshot snap
    {
        bufBack_.lanes,//lanes
        bufBack_.yolos,//yolos
        bufBack_.currentSpeed,//currentSpeed
        state::TEST,//state
        0,//dx
        0,//dAngle
        bufBack_.imuAngle,//imuAngle
        0,//motorOutput
        0,//servoOutput
        frameId_//frameId
    };

    //融合/滤波，snap 的字段上算，算完 return,这一块负责传感器融合和滤波，由外部事件驱动
    filterLanes(snap.lanes, 60.0);

    // 修改：根据丢线计数决定是否重置滤波器
    int left_mode = DEFAULT;
    int right_mode = DEFAULT;

    // 如果左车道线丢线计数达到阈值，则持续尝试重置
    if (left_loss_count >= MAX_LOSS_COUNT) {
        left_mode = RESET;
    }

    // 如果右车道线丢线计数达到阈值，则持续尝试重置
    if (right_loss_count >= MAX_LOSS_COUNT) {
        right_mode = RESET;
    }

    std::vector<LaneDescriptor> tracked_lanes = kalmanLanes(snap.lanes, tracker, left_mode, right_mode);

    // 修改：更新丢线计数器逻辑
    if (tracked_lanes.size() >= 2) {
        // 处理左车道线
        if (tracked_lanes[0].leftLoss) {
            left_loss_count++;
            // 修改：不再在达到阈值时重置计数器，而是保持重置状态
        }
        else {
            // 只有当成功检测到车道线时才重置计数器
            left_loss_count = 0;
        }

        // 处理右车道线
        if (tracked_lanes[1].rightLoss) {
            right_loss_count++;
            // 修改：不再在达到阈值时重置计数器，而是保持重置状态
        }
        else {
            // 只有当成功检测到车道线时才重置计数器
            right_loss_count = 0;
        }
    }

#ifdef DEBUG
    if (left_loss_count > 0) {
        std::cout << "[DEBUG] Left lane loss count: " << left_loss_count << std::endl;
    }
    if (right_loss_count > 0) {
        std::cout << "[DEBUG] Right lane loss count: " << right_loss_count << std::endl;
    }
    if (left_mode == RESET) {
        std::cout << "[DEBUG] Resetting left lane tracker" << std::endl;
    }
    if (right_mode == RESET) {
        std::cout << "[DEBUG] Resetting right lane tracker" << std::endl;
    }
#endif
    LaneDescriptor middleDescriptor = getMiddleLane(tracked_lanes);
    snap.dX = (int)(middleDescriptor.peakX);
    snap.dAngle = (float)(middleDescriptor.mainAngle);

    return snap;  // NRVO/move，读者无锁
}

/// @brief 决策协程，内部有决策逻辑，通过Coroutine_delay(ms)来延时，这一部分由world线程调度,控制器输入的时候应当使用filter做一次逆透视校正和滤波
/// 负责所有的决策和控制输出，逻辑是被调度之后会一直运行，直到遇到Coroutine_delay(ms)才会挂起，到时间再被调度器唤醒
/// 非常麻烦，最好不要动外部定义，这个全是kimi弄的，我没能力维护，只改内部逻辑就好了，必须支持cpp20
/// @return 
Task decision_coroutine() {
    while (1)
    {
        std::cout << "[CORO] Speed = 0\n";
        Coroutine_delay(1000);
		//调用示例（需要自己写好stanley和电控部分的实现和gcommandData输出控制结构体）：
        /*LaneDescriptor filteredMiddleLane = middleLaneFilter(gSnap.dX, gSnap.dAngle);
        gcommandData.setServoOutput(stanley.calcAngle(filteredMiddleLane.mainAngle * 0.53 / 240 - SHIFT, filteredMiddleLane.peakX, 0.9));*/

        std::cout << "[CORO] Speed = 5\n";
        Coroutine_delay(500);

        std::cout << "[CORO] Speed = 3\n";
        Coroutine_delay(500);

    returnLoop:
        std::cout << "[CORO] Finished Or Returned\n";
        Coroutine_delay(0);
    }
}

