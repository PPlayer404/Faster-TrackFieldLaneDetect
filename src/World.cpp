#include "World.hpp"
#include "Cluster.hpp"
#include "mode.hpp"
#include <climits>

#define DEFAULT 0
#define RECHECK 1
#define RESET 2

struct LaneDescriptor
{
    int peakX;          // 起始生长点（x坐标）
    double mainAngle;   // 主导角度（弧度）
    bool leftLoss;
    bool rightLoss;
};

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

    LaneKalmanFilter() : left_initialized(false), right_initialized(false),
        last_left_angle(0), last_right_angle(0),
        last_left_x(0), last_right_x(0) {
        const float R = 3.0f;          // 测量方差 ≈ 1.7 pixel σ
        const float Qp = 1.0f;          // 位置过程噪声
        const float Qv = 40.0f;         // 速度过程噪声
        const float P0 = 5.0f;          // 初始不确定度

        for (auto kf : { &kf_left, &kf_right }) {
            kf->init(2, 1, 0);
            kf->transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 1, 0, 1);
            kf->measurementMatrix = (cv::Mat_<float>(1, 2) << 1, 0);
            cv::setIdentity(kf->processNoiseCov, cv::Scalar(0.0f));
            kf->processNoiseCov.at<float>(0, 0) = Qp;
            kf->processNoiseCov.at<float>(1, 1) = Qv;
            cv::setIdentity(kf->measurementNoiseCov, cv::Scalar(R));
            cv::setIdentity(kf->errorCovPost, cv::Scalar(P0));
        }
    }
};

LaneKalmanFilter tracker;

/// @brief 使用卡尔曼滤波进行车道线跟踪和预测
/// @param lanes 输入的车道线聚类描述子集合
/// @param tracker 车道线卡尔曼滤波器跟踪器
/// @param mode 工作模式：DEFAULT(0)-默认模式, RECHECK(1)-重新检查模式, RESET(2)-重置模式
/// @return 返回跟踪后的车道线描述子（包含左右车道线，索引左0右1）
std::vector<LaneDescriptor> kalmanLanes(std::vector<ClusterDescriptor>& lanes, LaneKalmanFilter& tracker, int left_mode, int right_mode)
{
    std::vector<LaneDescriptor> result;
    static const int IMAGE_CENTER = 120;
    static const int MAX_LANE_WIDTH = 230;
    static const int MIN_LANE_WIDTH = 150;
    static const float VELOCITY_DECAY = 0.2f;
    static const double ANGLE_THRESHOLD = 0.5;
    static const double POSITION_WEIGHT = 0.7;
    static const double ANGLE_WEIGHT = 0.3;
    static const double MIN_SCORE_THRESHOLD = 0.5;
    static const int MATCH_THRESHOLD = 45;

    // Precompute candidates if lanes are not empty
    std::vector<ClusterDescriptor> left_candidates;
    std::vector<ClusterDescriptor> right_candidates;
    if (!lanes.empty()) {
        for (const auto& lane : lanes) {
            if (lane.peakX < IMAGE_CENTER) {
                left_candidates.push_back(lane);
            }
            else {
                right_candidates.push_back(lane);
            }
        }
        std::sort(left_candidates.begin(), left_candidates.end(),
            [](const ClusterDescriptor& a, const ClusterDescriptor& b) {
                return a.peakX > b.peakX;
            });
        std::sort(right_candidates.begin(), right_candidates.end(),
            [](const ClusterDescriptor& a, const ClusterDescriptor& b) {
                return a.peakX < b.peakX;
            });
    }

    // Handle modes independently for left and right
    bool reset_left = false;
    bool reset_right = false;

    // Left lane mode handling
    if (left_mode == RECHECK) {
        if (!lanes.empty() && tracker.left_initialized && !left_candidates.empty()) {
            float current_left_x = tracker.kf_left.statePost.at<float>(0);
            float candidate_left_x = left_candidates[0].peakX;
            if (std::abs(candidate_left_x - current_left_x) > 40) {
                reset_left = true;
            }
        }
    }
    else if (left_mode == RESET) {
        reset_left = true;
    }

    // Right lane mode handling  
    if (right_mode == RECHECK) {
        if (!lanes.empty() && tracker.right_initialized && !right_candidates.empty()) {
            float current_right_x = tracker.kf_right.statePost.at<float>(0);
            float candidate_right_x = right_candidates[0].peakX;
            if (std::abs(candidate_right_x - current_right_x) > 40) {
                reset_right = true;
            }
        }
    }
    else if (right_mode == RESET) {
        reset_right = true;
    }

    if (reset_left) {
        tracker.left_initialized = false;
    }
    if (reset_right) {
        tracker.right_initialized = false;
    }

    // 其余代码保持不变...
    // [原函数中从 "Original logic" 开始到函数结束的所有代码保持不变]
    // 这里为了简洁省略重复代码，实际使用时需要将原函数的剩余部分完整复制过来

    // 复制原函数中从 "Original logic" 开始的所有代码...
    if (lanes.empty()) {
        LaneDescriptor left_lane, right_lane;
        if (tracker.left_initialized) {
            tracker.kf_left.predict();
            tracker.kf_left.statePost.at<float>(1) *= VELOCITY_DECAY;
            cv::Mat left_state = tracker.kf_left.statePost;
            left_lane.peakX = left_state.at<float>(0);
            left_lane.mainAngle = tracker.last_left_angle;
            left_lane.leftLoss = true;
            left_lane.rightLoss = false;
        }
        else {
            left_lane.leftLoss = true;
            left_lane.rightLoss = false;
        }
        if (tracker.right_initialized) {
            tracker.kf_right.predict();
            tracker.kf_right.statePost.at<float>(1) *= VELOCITY_DECAY;
            cv::Mat right_state = tracker.kf_right.statePost;
            right_lane.peakX = right_state.at<float>(0);
            right_lane.mainAngle = tracker.last_right_angle;
            right_lane.leftLoss = false;
            right_lane.rightLoss = true;
        }
        else {
            right_lane.leftLoss = false;
            right_lane.rightLoss = true;
        }
        result.push_back(left_lane);
        result.push_back(right_lane);

        cv::Mat vis = cv::Mat::zeros(120, 240, CV_8UC3);
        if (tracker.left_initialized) {
            cv::circle(vis, cv::Point(left_lane.peakX, 60), 5, cv::Scalar(255, 0, 0), -1);
        }
        if (tracker.right_initialized) {
            cv::circle(vis, cv::Point(right_lane.peakX, 60), 5, cv::Scalar(0, 0, 255), -1);
        }
        cv::imshow("Lane Tracking", vis);
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
    else {
        if (!left_candidates.empty()) {
            left_index = 0;
        }
    }

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
    else {
        if (!right_candidates.empty()) {
            right_index = 0;
        }
    }

    if (left_index != -1 && right_index != -1) {
        int lane_width = right_candidates[right_index].peakX - left_candidates[left_index].peakX;
        if (lane_width < MIN_LANE_WIDTH || lane_width > MAX_LANE_WIDTH) {
            if (tracker.right_initialized) {
                left_index = -1;
            }
            else if (tracker.left_initialized) {
                right_index = -1;
            }
        }
    }

    if (left_index != -1) {
        cv::Mat measurement = (cv::Mat_<float>(1, 1) << left_candidates[left_index].peakX);
        if (!tracker.left_initialized) {
            tracker.kf_left.statePre.at<float>(0) = left_candidates[left_index].peakX;
            tracker.kf_left.statePre.at<float>(1) = 0;
            tracker.kf_left.statePost = tracker.kf_left.statePre.clone();
            tracker.left_initialized = true;
            tracker.last_left_x = left_candidates[left_index].peakX;
        }
        else {
            tracker.kf_left.predict();
            tracker.kf_left.correct(measurement);
            tracker.last_left_x = left_candidates[left_index].peakX;
        }
        cv::Mat left_state = tracker.kf_left.statePost;
        left_lane.peakX = left_state.at<float>(0);
        left_lane.mainAngle = left_candidates[left_index].mainAngle;
        tracker.last_left_angle = left_candidates[left_index].mainAngle;
        left_lane.leftLoss = false;
        left_lane.rightLoss = false;
    }
    else if (tracker.left_initialized) {
        tracker.kf_left.predict();
        tracker.kf_left.statePost.at<float>(1) *= VELOCITY_DECAY;
        cv::Mat left_state = tracker.kf_left.statePost;
        left_lane.peakX = left_state.at<float>(0);
        left_lane.mainAngle = tracker.last_left_angle;
        left_lane.leftLoss = true;
        left_lane.rightLoss = false;
    }
    else {
        left_lane.leftLoss = true;
        left_lane.rightLoss = false;
    }

    if (right_index != -1) {
        cv::Mat measurement = (cv::Mat_<float>(1, 1) << right_candidates[right_index].peakX);
        if (!tracker.right_initialized) {
            tracker.kf_right.statePre.at<float>(0) = right_candidates[right_index].peakX;
            tracker.kf_right.statePre.at<float>(1) = 0;
            tracker.kf_right.statePost = tracker.kf_right.statePre.clone();
            tracker.right_initialized = true;
            tracker.last_right_x = right_candidates[right_index].peakX;
        }
        else {
            tracker.kf_right.predict();
            tracker.kf_right.correct(measurement);
            tracker.last_right_x = right_candidates[right_index].peakX;
        }
        cv::Mat right_state = tracker.kf_right.statePost;
        right_lane.peakX = right_state.at<float>(0);
        right_lane.mainAngle = right_candidates[right_index].mainAngle;
        tracker.last_right_angle = right_candidates[right_index].mainAngle;
        right_lane.leftLoss = false;
        right_lane.rightLoss = false;
    }
    else if (tracker.right_initialized) {
        tracker.kf_right.predict();
        tracker.kf_right.statePost.at<float>(1) *= VELOCITY_DECAY;
        cv::Mat right_state = tracker.kf_right.statePost;
        right_lane.peakX = right_state.at<float>(0);
        right_lane.mainAngle = tracker.last_right_angle;
        right_lane.leftLoss = false;
        right_lane.rightLoss = true;
    }
    else {
        right_lane.leftLoss = false;
        right_lane.rightLoss = true;
    }

    result.push_back(left_lane);
    result.push_back(right_lane);

#ifdef KALMAN_DEBUG
    cv::Mat vis = cv::Mat::zeros(120, 240, CV_8UC3);
    for (const auto& lane : lanes) {
        cv::circle(vis, cv::Point(lane.peakX, 60), 3, cv::Scalar(255, 255, 255), -1);
    }
    if (tracker.left_initialized) {
        cv::circle(vis, cv::Point(left_lane.peakX, 60), 5, cv::Scalar(255, 0, 0), -1);
        cv::putText(vis, left_lane.leftLoss ? "L:pred" : "L:det",
            cv::Point(left_lane.peakX - 20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 0));
    }
    if (tracker.right_initialized) {
        cv::circle(vis, cv::Point(right_lane.peakX, 60), 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(vis, right_lane.rightLoss ? "R:pred" : "R:det",
            cv::Point(right_lane.peakX - 20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255));
    }
    cv::imshow("Lane Tracking", vis);
    cv::waitKey(1);
#endif

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
    constexpr double OFFSET_DIST = 60.0;

    static LaneDescriptor lastMiddleLane = { 0, 0 };

    static cv::Mat ipm_mat = (cv::Mat_<double>(3, 3) <<
        4.000000, 10.266667, -312.000000,
        0.000000, 41.833333, -120.000000,
        0.000000, 0.088889, 1.000000
        );
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

/// @brief 更新维护世界模型，双缓冲 + 计算 + 返回快照
WorldSnapshot World::dataSync()
{
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

    //融合/滤波，snap 的字段上算，算完 return
	filterLanes(snap.lanes, 60.0);
    std::vector<LaneDescriptor> tracked_lanes = kalmanLanes(snap.lanes, tracker, DEFAULT, DEFAULT);
	LaneDescriptor middleDescriptor = getMiddleLane(tracked_lanes);
	snap.dX = (int)(middleDescriptor.peakX);
	snap.dAngle = (float)(middleDescriptor.mainAngle);
    //stanley();

    return snap;  // NRVO/move，读者无锁
}

