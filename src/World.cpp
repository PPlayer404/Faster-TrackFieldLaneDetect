#include "World.hpp"
#include "Cluster.hpp"
#include "mode.hpp"
#include <climits>

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
        last_left_angle(0), last_right_angle(0), last_left_x(0), last_right_x(0) {
        kf_left.init(2, 1, 0);
        kf_left.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 1, 0, 1);
        kf_left.measurementMatrix = (cv::Mat_<float>(1, 2) << 1, 0);
        cv::setIdentity(kf_left.processNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(kf_left.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(kf_left.errorCovPost, cv::Scalar::all(1));

        kf_right.init(2, 1, 0);
        kf_right.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 1, 0, 1);
        kf_right.measurementMatrix = (cv::Mat_<float>(1, 2) << 1, 0);
        cv::setIdentity(kf_right.processNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(kf_right.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(kf_right.errorCovPost, cv::Scalar::all(1));
    }
};

LaneKalmanFilter tracker;

std::vector<LaneDescriptor> kalmanLanes(std::vector<ClusterDescriptor>& lanes, LaneKalmanFilter& tracker)
{
    std::vector<LaneDescriptor> result;
    static const int IMAGE_CENTER = 120;
    static const int MAX_LANE_WIDTH = 100;
    static const int MIN_LANE_WIDTH = 40;
    static const float VELOCITY_DECAY = 0.6f;
    static const double ANGLE_THRESHOLD = 0.5;
    static const double POSITION_WEIGHT = 0.7;
    static const double ANGLE_WEIGHT = 0.3;
    static const double MIN_SCORE_THRESHOLD = 0.5;

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

    std::vector<ClusterDescriptor> left_candidates;
    std::vector<ClusterDescriptor> right_candidates;

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

    int left_index = -1, right_index = -1;
    LaneDescriptor left_lane, right_lane;

    const int MATCH_THRESHOLD = 30;

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
/// @param lanes 
/// @return 
LaneDescriptor getMiddleLane(std::vector<LaneDescriptor>& lanes)
{
    constexpr int IMG_W = 240;
    constexpr int IMG_H = 120;
    constexpr int REF_Y = 60;
    constexpr int CTR_X = IMG_W / 2;
    constexpr int BOTTOM_Y = IMG_H - 1;
    constexpr double OFFSET_DIST = 36.0;

    static LaneDescriptor lastMiddleLane = { 0, 0 };

    static cv::Mat ipm_mat = (cv::Mat_<double>(3, 3) <<
        4.000000, 4.200000, -312.000000,
        0.000000, 5.600000, -0.000000,
        0.000000, 0.038333, 1.000000
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

    // 计算缩放比例
    float scaleX = w / 240.0f;
    float scaleY = h / 120.0f;

    // 底部中心点（相对于原240x120图像的坐标系）
    int bottomCenterX = w / 2 + dX * scaleX;
    int bottomY = h - 1;

    // 计算线段长度（从底部到顶部1/4）
    int lineLength = h / 4 * 3;

    // 计算终点坐标
    int topX = bottomCenterX - lineLength * std::tan(angle) * (scaleX / scaleY);
    int topY = bottomY - lineLength;

    // 绘制线段
    cv::line(frame,
        cv::Point(bottomCenterX, bottomY),
        cv::Point(topX, topY),
        cv::Scalar(0, 0, 255),  // 绿色
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

void World::updateMCU(float angle)
{
    bufFront_.mcuAngle = angle;
}

/// @brief 构造函数，给一个初始值
World::World()
{
    WorldBuf init;
    init.currentSpeed = 0.0f;
    init.mcuAngle = 0;
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

    //无锁，用的数据是 bufBack_ 
    WorldSnapshot snap
    {
        bufBack_.lanes,//lanes
        bufBack_.yolos,//yolos
        bufBack_.currentSpeed,//currentSpeed
        state::TEST,//state
        0,//dx
        0,//dAngle
        bufBack_.mcuAngle,//mcuAngle
        frameId_//frameId
    };

    //融合/滤波，snap 的字段上算，算完 return
    std::vector<LaneDescriptor> tracked_lanes = kalmanLanes(snap.lanes, tracker);

	LaneDescriptor middleDescriptor = getMiddleLane(tracked_lanes);
	snap.dX = (int)(middleDescriptor.peakX);
	snap.dAngle = (float)(middleDescriptor.mainAngle);

    return snap;  // NRVO/move，读者无锁
}

