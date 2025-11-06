#include <opencv2/opencv.hpp>
#include "Lane.hpp"
#include "mode.hpp"
#include "guidedFilter.hpp"

namespace LaneTB
{
    int rho = 2;            // 霍夫累加器距离分辨率
    int theta = 100;         // 霍夫累加器角度分辨率（实际用 theta/100.0*CV_PI/180）
    int thresh = 110;        // 累加器阈值
    int minLen = 40;        // 最小线段长度
    int maxGap = 5;         // 最大允许断裂
    const int maxCount = 100;
    std::string winName = "Lane Detect Params";
}

/// @brief 双阈值实现
/// @param input 边缘强度图
/// @param highRatio 高阈值
/// @param lowRatio 低阈值
void optimizedDoubleThreshold(cv::Mat& input, double highRatio = 0.7, double lowRatio = 0.3)
{
    CV_Assert(!input.empty() && (input.channels() == 1));

    double maxVal;
    cv::minMaxLoc(input, nullptr, &maxVal);
    if (maxVal < std::numeric_limits<double>::epsilon()) {
        input.setTo(0);
        return;
    }
    double highThreshold = maxVal * highRatio;
    double lowThreshold = maxVal * lowRatio;
    cv::Mat allPotentialEdges = (input >= lowThreshold);
    cv::Mat strongMask = (input >= highThreshold);
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(allPotentialEdges, labels, stats, centroids, 8, CV_32S);
    cv::Mat finalMask = cv::Mat::zeros(input.size(), CV_8U);

    for (int i = 1; i < numLabels; ++i) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        if (width <= 0 || height <= 0) continue;

        cv::Rect roi(x, y, width, height);
        cv::Mat labelsROI = labels(roi);
        cv::Mat strongMaskROI = strongMask(roi);

        cv::Mat componentContainsStrong = (labelsROI == i) & strongMaskROI;
        if (cv::countNonZero(componentContainsStrong) > 0) {
            finalMask.setTo(255, labels == i);
        }
    }

    input.setTo(0);
    input.setTo(255, finalMask);
}

/// @brief Fused_Directional_Edge_Filter 算子融合定向边缘检测器
/// @param gray 要提取边缘的灰度图
/// @return 专有结构体
FDEF_Result FDEF(const cv::Mat& gray)
{
    static const cv::Mat kernel_45 = (cv::Mat_<float>(5, 5) <<
        0, 1, 1, 1, 1,
        -1, 0, 3, 2, 1,
        -1, -3, 0, 3, 1,
        -1, -2, -3, 0, 1,
        -1, -1, -1, -1, 0);

    static const cv::Mat kernel_135 = (cv::Mat_<float>(5, 5) <<
        -1, -1, -1, -1, 0,
        -1, -2, -3, 0, 1,
        -1, -3, 0, 3, 1,
        -1, 0, 3, 2, 1,
        0, 1, 1, 1, 1);

    cv::Mat s_vert0;
    cv::Sobel(gray, s_vert0, CV_32F, 1, 0, 5); // x方向导数，5x5核

    cv::Mat s45, s135;
    cv::filter2D(gray, s45, CV_32F, kernel_45, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(gray, s135, CV_32F, kernel_135, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    cv::Mat s_vert1(s_vert0.size(), CV_32F), s_vert2(s_vert0.size(), CV_32F);
    cv::max(s_vert0, 0, s_vert1);
    cv::min(s_vert0, 0, s_vert2);
    cv::multiply(s_vert2, -1.0f, s_vert2);

    cv::max(s45, 0, s45);
    cv::max(s135, 0, s135);

    const float alpha = 1.0f / 12.0f;
    cv::Mat s_vert1_u8, s_vert2_u8, s45_u8, s135_u8;
    s_vert1.convertTo(s_vert1_u8, CV_8U, alpha);
    s_vert2.convertTo(s_vert2_u8, CV_8U, alpha);
    s45.convertTo(s45_u8, CV_8U, alpha);
    s135.convertTo(s135_u8, CV_8U, alpha);

    cv::Mat shifted = cv::Mat::zeros(s_vert2_u8.size(), s_vert2_u8.type());
    if (s_vert2_u8.cols > 5) {
        cv::Rect srcRect(5, 0, s_vert2_u8.cols - 5, s_vert2_u8.rows);
        cv::Rect dstRect(0, 0, s_vert2_u8.cols - 5, s_vert2_u8.rows);
        s_vert2_u8(srcRect).copyTo(shifted(dstRect));
    }
    s_vert2_u8 = shifted;

    int mid = gray.cols / 3;
    if (mid > 0) {
        s45_u8.colRange(0, mid) = cv::Scalar(0);
    }
    mid *= 2;
    if (mid < gray.cols) {
        s135_u8.colRange(mid, gray.cols) = cv::Scalar(0);
    }

    cv::Mat s_diag, s_vert;
    cv::max(s45_u8, s135_u8, s_diag);
    cv::max(s_vert1_u8, s_vert2_u8, s_vert);

    FDEF_Result result;
    result.sDiag = s_diag;
    result.sVert = s_vert;

    return result;
}

/// @brief 车道线检测，输入处理后的图像和hsv阈值结构体，返回原始的，未加任何滤波的直线组
/// @param processed_img 预处理之后的图像，预处理放主线程，实在不行再调整
/// @param HSV HSV阈值结构体，按要求填写
/// @return 原始直线组结果
std::vector<cv::Vec4i> detectLanes(cv::Mat& processed_img, HSV_Lane HSV)
{
    cv::Mat blurred, gray, blurred_1;
    cv::transform(processed_img, gray, cv::Matx13f(0.114f / 0.701f, 0.587f / 0.701f, 0.0f));
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.7, cv::Size(4, 2));
    clahe->apply(gray, gray);
    ShadowLift(gray, 170, 140);
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
    FDEF_Result edges = FDEF(blurred);
    // 双阈值法
    optimizedDoubleThreshold(edges.sVert, 0.6, 0.4);
    optimizedDoubleThreshold(edges.sDiag, 0.4, 0.2);

    double r = LaneTB::rho;
    double t = LaneTB::theta / 100.0 * CV_PI / 180.0;
    int   th = LaneTB::thresh;
    int   mlen = LaneTB::minLen;
    int   mgap = LaneTB::maxGap;

    std::vector<cv::Vec4i> linesY, lines45;
    cv::HoughLinesP(edges.sVert, linesY, r, t, th, mlen, mgap);
    cv::HoughLinesP(edges.sDiag, lines45, r, t, th, mlen, mgap);

    //合并所有直线
    std::vector<cv::Vec4i> Lanes;
    Lanes.insert(Lanes.end(), linesY.begin(), linesY.end());
    Lanes.insert(Lanes.end(), lines45.begin(), lines45.end());

#ifdef WITH_IMSHOW
    // 1. 画车道线
    cv::Mat img_with_lanes = processed_img.clone();
    for (const auto& l : Lanes)
        cv::line(img_with_lanes, cv::Point(l[0], l[1]),
            cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    // 2. 原始逆透视矩阵
    cv::Mat H = (cv::Mat_<double>(3, 3) <<
        -0.350648f, -2.041805f, 461.86929f,
        -0.147207f, -3.328071f, 749.042284f,
        -0.000292f, -0.005138f, 1.0f
        );

    // 3. 构造“画布居中”平移矩阵
    int canvas_size = 640;
    int roi_w = 240, roi_h = 240;               // 想要保留的 ROI 尺寸
    int offset_x = (canvas_size - roi_w) / 2;   // ROI 在画布上的左上角
    int offset_y = (canvas_size - roi_h) / 2;
    cv::Mat T = (cv::Mat_<double>(3, 3) <<
        1, 0, offset_x,
        0, 1, offset_y,
        0, 0, 1);

    // 4. 新的单应：先平移再执行原 H
    cv::Mat H_new = T * H;

    // 5. 一次 warp 到 640×640 画布
    cv::Mat canvas;
    cv::warpPerspective(img_with_lanes, canvas, H_new,
        cv::Size(canvas_size, canvas_size),
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    // 6. 显示
    cv::imshow("Bird-Eye View", canvas);
    cv::imshow("blurred", blurred);
    cv::imshow("raw", img_with_lanes);
    cv::imshow("edge", edges.sDiag);
    cv::imshow("edgeY", edges.sVert);
    cv::imshow("gray", gray);
    cv::waitKey(1);
#endif
    return Lanes;
}

/// @brief 零开销拷贝，const强制要求不得修改传入的图像指针
/// @param frame 输入图像，最好240*120
/// @return 校正之后的图片，仅校正亮度
cv::Mat color_correction(const cv::Mat& frame)
{
    cv::Mat lab;
    cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_planes;
    cv::split(lab, lab_planes); // lab_planes[0]=L, [1]=a, [2]=b

    //仅对L通道应用CLAHE
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(4, 2));
    clahe->apply(lab_planes[0], lab_planes[0]);
    cv::Mat lab_corrected;
    cv::merge(lab_planes, lab_corrected);

    //LAB -> BGR
    cv::Mat result;
    cv::cvtColor(lab_corrected, result, cv::COLOR_Lab2BGR);
    return result;
}


/// @brief 对灰度图进行暗部提升
/// @param gray 输入灰度图
/// @param threshold 截断阈值
/// @param lift_to 提升后的最低值
void ShadowLift(cv::Mat& gray, int threshold, int lift_to) {
    //参数检查
    CV_Assert(threshold > lift_to && lift_to >= 0 && threshold < 256);
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();

    //计算二次多项式系数
    float a = (float)lift_to / (threshold * threshold);
    float b = 1.0f - 2.0f * lift_to / threshold;
    //填充映射
    for (int i = 0; i < threshold; ++i) {
        float x = (float)i;
        p[i] = cv::saturate_cast<uchar>(a * x * x + b * x + lift_to);
    }
    for (int i = threshold; i < 256; ++i) {
        p[i] = i;
    }

    //查表完成变换
    cv::LUT(gray, lookUpTable, gray);
}

/// @brief 进度条初始化
/// @param void
void Lane_init(void)
{
    cv::namedWindow(LaneTB::winName, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("rho", LaneTB::winName, &LaneTB::rho, 10);
    cv::createTrackbar("theta(*PI/18000)", LaneTB::winName, &LaneTB::theta, 300);
    cv::createTrackbar("threshold", LaneTB::winName, &LaneTB::thresh, 200);
    cv::createTrackbar("minLineLength", LaneTB::winName, &LaneTB::minLen, 200);
    cv::createTrackbar("maxLineGap", LaneTB::winName, &LaneTB::maxGap, 100);
    return;
}

//长度为6的数组：{L_min, L_max, a_min, a_max, b_min, b_max}

/// @brief 该函数不能使用，已被弃用
/// @param bgr 
/// @param th 
/// @return 
cv::Mat labGetBlob(const cv::Mat& bgr, const int th[6])
{
    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);   // L,a,b 均为 8U

    // 创建三通道掩码，每个通道满足条件
    cv::Mat mask_low = (lab >= cv::Scalar(th[0], th[2], th[4]));
    cv::Mat mask_high = (lab <= cv::Scalar(th[1], th[3], th[5]));
    cv::Mat mask = mask_low & mask_high; // 三通道掩码

    // 将三通道掩码转换为单通道掩码：所有通道都为255时才为255
    std::vector<cv::Mat> channels;
    cv::split(mask, channels);
    cv::Mat singleMask = channels[0] & channels[1] & channels[2];

    return singleMask;
}

/// @brief 快速导向滤波 - 来自知乎大哥的回答，不知道好用不好用，以下注释都是ai写的
/// @param srcImage 输入图像（单通道）
/// @param guidedImage 引导图像（单通道）
/// @param outputImage 输出图像
/// @param filterSize 滤波半径（在原图尺度）
/// @param eps 正则化参数，控制平滑程度（值越小边缘保持越强）
/// @param samplingRate 下采样率，用于加速计算（1=不加速，2=长宽各减半）
void FastGuidedFilter(cv::Mat& srcImage, cv::Mat& guidedImage, cv::Mat& outputImage, int filterSize, double eps, int samplingRate)
{
    try
    {
        if (srcImage.empty() || guidedImage.empty() || filterSize <= 0 || eps < 0 ||
            srcImage.channels() != 1 || guidedImage.channels() != 1 || samplingRate < 1)
        {
            throw "params input error";
        }

        // 确保下采样后的filterSize至少为1
        int downsampledFilterSize = std::max(1, filterSize / samplingRate);

        cv::Mat srcImageP, srcImageSubI, srcImageI, meanP, meanI, meanIP, meanII, var, alfa, beta;

        // 下采样
        cv::resize(srcImage, srcImageP, cv::Size(srcImage.cols / samplingRate, srcImage.rows / samplingRate));
        cv::resize(guidedImage, srcImageSubI, cv::Size(srcImage.cols / samplingRate, srcImage.rows / samplingRate));

        // 转换为浮点并归一化到[0,1]
        srcImageP.convertTo(srcImageP, CV_32FC1, 1.0 / 255.0);
        guidedImage.convertTo(srcImageI, CV_32FC1, 1.0 / 255.0);
        srcImageSubI.convertTo(srcImageSubI, CV_32FC1, 1.0 / 255.0);

        // 导向滤波计算
        cv::boxFilter(srcImageP, meanP, CV_32FC1, cv::Size(downsampledFilterSize, downsampledFilterSize));
        cv::boxFilter(srcImageSubI, meanI, CV_32FC1, cv::Size(downsampledFilterSize, downsampledFilterSize));
        cv::boxFilter(srcImageSubI.mul(srcImageP), meanIP, CV_32FC1, cv::Size(downsampledFilterSize, downsampledFilterSize));
        cv::boxFilter(srcImageSubI.mul(srcImageSubI), meanII, CV_32FC1, cv::Size(downsampledFilterSize, downsampledFilterSize));

        var = meanII - meanI.mul(meanI);
        alfa = (meanIP - meanI.mul(meanP)) / (var + eps);
        beta = meanP - alfa.mul(meanI);

        cv::boxFilter(alfa, alfa, CV_32FC1, cv::Size(downsampledFilterSize, downsampledFilterSize));
        cv::boxFilter(beta, beta, CV_32FC1, cv::Size(downsampledFilterSize, downsampledFilterSize));

        cv::resize(alfa, alfa, srcImage.size());
        cv::resize(beta, beta, srcImage.size());

        outputImage = alfa.mul(srcImageI) + beta;

        // 关键修复：将输出转换回0-255的uchar类型
        outputImage.convertTo(outputImage, CV_8UC1, 255.0);
    }
    catch (cv::Exception& e)
    {
        throw e;
    }
    catch (std::exception& e)
    {
        throw e;
    }
}

/// @brief CSDN大哥的实现，具体参考上一个函数实现
/// opencv官方的接口需要contrib，win端的opencv没下这个包
/// @param I_org 输入图像
/// @param p_org 引导图像
/// @param r 滤波半径
/// @param eps 正则化参数
/// @param s 下采样率
/// @return 
cv::Mat fastGuidedFilter_2(cv::Mat I_org, cv::Mat p_org, int r, double eps, int s)
{
    cv::Mat I, _I;
    I_org.convertTo(_I, CV_64FC1, 1.0 / 255);
    resize(_I, I, cv::Size(), 1.0 / s, 1.0 / s, cv::INTER_LINEAR);

    cv::Mat p, _p;
    p_org.convertTo(_p, CV_64FC1, 1.0 / 255);
    resize(_p, p, cv::Size(), 1.0 / s, 1.0 / s, cv::INTER_LINEAR);

    // 调整半径
    r = (2 * r + 1) / s + 1;

    cv::Mat mean_I, mean_p, mean_Ip, mean_II;
    cv::boxFilter(I, mean_I, CV_64FC1, cv::Size(r, r));
    cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));
    cv::boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));
    cv::boxFilter(I.mul(I), mean_II, CV_64FC1, cv::Size(r, r));

    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    cv::Mat var_I = mean_II - mean_I.mul(mean_I);
    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);

    cv::Mat mean_a, mean_b;
    cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
    cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));

    cv::Mat rmean_a, rmean_b;
    resize(mean_a, rmean_a, I_org.size(), 0, 0, cv::INTER_LINEAR);
    resize(mean_b, rmean_b, I_org.size(), 0, 0, cv::INTER_LINEAR);

    cv::Mat q = rmean_a.mul(_I) + rmean_b;

    // 可选：转换回8UC1用于显示
    cv::Mat result;
    q.convertTo(result, CV_8UC1, 255.0);

    return result;  // 或者返回 q 保持浮点精度
}

/**
 * @brief Applies the Frangi filter to a 2D grayscale image to enhance line-like structures.
 *
 * @param src The input single-channel 8-bit image (CV_8UC1).
 * @param sigma_start The starting scale for the filter (e.g., 1.0).
 * @param sigma_end The ending scale for the filter (e.g., 5.0).
 * @param sigma_step The step size between scales (e.g., 0.5).
 * @param beta1 Parameter that controls the sensitivity to blob-like structures.
 * @param beta2 Parameter that controls the sensitivity to background noise.
 * @return The filtered image (CV_8UC1), where bright pixels indicate strong line-like responses.
 */
cv::Mat frangi_filter(const cv::Mat& src, float sigma_start, float sigma_end, float sigma_step, float beta1, float beta2) {
    cv::Mat src_float;
    src.convertTo(src_float, CV_32F);
    cv::Mat J = cv::Mat::zeros(src.size(), CV_32F);

    float beta1_sq = beta1 * beta1;
    float beta2_sq = beta2 * beta2;

    for (float sigma = sigma_start; sigma <= sigma_end; sigma += sigma_step) {
        cv::Mat blurred;
        cv::GaussianBlur(src_float, blurred, cv::Size(0, 0), sigma);

        cv::Mat Ixx, Ixy, Iyy;
        cv::Sobel(blurred, Ixx, CV_32F, 2, 0, 1);
        cv::Sobel(blurred, Ixy, CV_32F, 1, 1, 1);
        cv::Sobel(blurred, Iyy, CV_32F, 0, 2, 1);

        cv::Mat temp1, temp2, temp3;
        temp1 = (Ixx - Iyy) / 2.0f;
        temp2 = Ixy;
        cv::pow(temp1, 2, temp1);
        cv::pow(temp2, 2, temp2);
        cv::sqrt(temp1 + temp2, temp3); // sqrt((Ixx-Iyy)^2/4 + Ixy^2)

        cv::Mat lambda1 = (Ixx + Iyy) / 2.0f + temp3;
        cv::Mat lambda2 = (Ixx + Iyy) / 2.0f - temp3;

        lambda1 = cv::abs(lambda1);
        lambda2 = cv::abs(lambda2);

        // Ensure |lambda1| >= |lambda2|
        cv::Mat mask = lambda1 < lambda2;
        cv::Mat temp = lambda1.clone();
        lambda1.copyTo(temp, mask);
        lambda2.copyTo(lambda1, mask);
        temp.copyTo(lambda2, mask);

        cv::Mat Rb, S, F;
        cv::divide(lambda2, lambda1, Rb); // lambda2 / lambda1
        cv::pow(Rb, 2, Rb);
        cv::exp(-Rb / beta1_sq, Rb); // exp(-(lambda2/lambda1)^2 / beta1^2)

        cv::pow(lambda1, 2, S);
        cv::pow(lambda2, 2, temp1);
        S += temp1;
        cv::exp(-S / (2 * beta2_sq), S); // exp(-(lambda1^2 + lambda2^2) / (2 * beta2^2))

        F = Rb.mul(S);
        J = cv::max(J, F);
    }

    cv::Mat result;
    J.convertTo(result, CV_8U, 255.0);
    return result;
}
