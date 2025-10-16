#include <opencv2/opencv.hpp>
#include "Lane.hpp"
#include "mode.hpp"
#include "guidedFilter.hpp"

namespace LaneTB
{
    int rho = 2;            // 霍夫累加器距离分辨率
    int theta = 100;         // 霍夫累加器角度分辨率（实际用 theta/100.0*CV_PI/180）
    int thresh = 120;        // 累加器阈值
    int minLen = 50;        // 最小线段长度
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
    double maxVal;
    cv::minMaxLoc(input, nullptr, &maxVal);

    double highThreshold = maxVal * highRatio;
    double lowThreshold = maxVal * lowRatio;

    // 创建强边缘和弱边缘掩码
    cv::Mat strongMask = (input >= highThreshold);
    cv::Mat weakMask = (input >= lowThreshold) & (input < highThreshold);

    // 使用距离变换找到弱边缘到强边缘的连通性
    cv::Mat distTransform;
    cv::distanceTransform(~strongMask, distTransform, cv::DIST_C, 3);

    // 设置连通距离阈值（可根据图像大小调整）
    double connectivityThreshold = 1.0;
    cv::Mat connectedWeak = (distTransform <= connectivityThreshold) & weakMask;

    // 合并结果
    input.setTo(0);
    input.setTo(255, strongMask);
    input.setTo(255, connectedWeak);
}

/// @brief Fused_Directional_Edge_Filter 算子融合定向边缘检测器
/// @param gray 要提取边缘的灰度图
/// @param weight 已废弃参数
/// @return 专有结构体
FDEF_Result FDEF(const cv::Mat& gray, float weight = 7.f)
{
    float w = weight / 2.5f;
    float d = 0.95f;

    // 创建45度核
    cv::Mat kernel_45 = (cv::Mat_<float>(5, 5) <<
         0,  1,  1,  1,  1,
        -1,  0,  3,  2,  1,
        -1, -3,  0,  3,  1,
        -1, -2, -3,  0,  1,
        -1, -1, -1, -1,  0);

    // 创建135度核
    cv::Mat kernel_135 = (cv::Mat_<float>(5, 5) <<
        -1, -1, -1, -1,  0,
        -1, -2, -3,  0,  1,
        -1, -3,  0,  3,  1,
        -1,  0,  3,  2,  1,
         0,  1,  1,  1,  1);

    cv::Mat kernel_90 = (cv::Mat_<float>(5, 5) <<
        -1, -1,  0,  1,  1,
        -1, -2,  0,  2,  1,
        -1, -3,  0,  3,  1,
        -1, -2,  0,  2,  1,
        -1, -1,  0,  1,  1);

    // 应用滤波器
    cv::Mat s_vert0, s_vert1, s_vert2, s45, s135;
    cv::filter2D(gray, s_vert0, -1, kernel_90, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(gray, s45, -1, kernel_45, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(gray, s135, -1, kernel_135, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    // 第一步：直接截断0以下的梯度值（将负值设为0）
    cv::max(s_vert0, 0, s_vert1);  // 只保留非负值，负值直接截断为0
    cv::min(s_vert0, 0, s_vert2);
    s_vert2 = -s_vert2;//把垂直方向的正负梯度都弄出来
    cv::max(s45, 0, s45);
    cv::max(s135, 0, s135);

    // 第二步：仅对正梯度进行归一化拉伸（0到最大值映射到0到255）
    cv::normalize(s_vert1, s_vert1, 0, 255, cv::NORM_MINMAX);
    cv::normalize(s_vert2, s_vert2, 0, 255, cv::NORM_MINMAX);
    cv::normalize(s45, s45, 0, 255, cv::NORM_MINMAX);
    cv::normalize(s135, s135, 0, 255, cv::NORM_MINMAX);

    // 转换为8位无符号整数
    s_vert1.convertTo(s_vert1, CV_8U);
    s_vert2.convertTo(s_vert2, CV_8U);
    s45.convertTo(s45, CV_8U);
    s135.convertTo(s135, CV_8U);

    // 左移5个像素 - 使用参考程序中的方法
    cv::Mat shifted = cv::Mat::zeros(s_vert2.size(), s_vert2.type());
    s_vert2(cv::Rect(5, 0, s_vert2.cols - 5, s_vert2.rows)).copyTo(
        shifted(cv::Rect(0, 0, s_vert2.cols - 5, s_vert2.rows)));
    s_vert2 = shifted;

    // 处理对角线结果
    int mid = gray.cols / 3;
    s45(cv::Rect(0, 0, mid, gray.rows)) = 0;
    mid *= 2;
    s135(cv::Rect(mid, 0, gray.cols - mid, gray.rows)) = 0;
    cv::Mat s_diag = cv::max(s45, s135);

    // 处理垂直结果
    cv::Mat s_vert = cv::max(s_vert1, s_vert2);

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
    //cv::Mat gray_custom;
    //cv::transform(processed_img, gray_custom, cv::Matx13f(1, 0.0, 0.0)); // -R + G + B
    //cv::normalize(gray_custom, gray, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::cvtColor(processed_img, gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.7, cv::Size(4, 2));
    clahe->apply(gray, gray);
    cv::Mat mask = gray < 150;
    gray.setTo(150, mask);
    clahe->apply(gray, gray);
    cv::GaussianBlur(gray, blurred_1, cv::Size(3, 3), 1.0);
    fgf::structureTransference(gray, blurred_1, blurred, 12, 0.02, 2);
    /*blurred = fastGuidedFilter_2(gray, blurred_1, 12, 0.02, 2);*/
    //FastGuidedFilter(gray, gray, blurred, 12, 0.02, 2);
    //cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);  // 5x5核， sigma=1.5
    FDEF_Result edges = FDEF(blurred);
    // 双阈值法
    optimizedDoubleThreshold(edges.sVert, 0.8, 0.6);
    optimizedDoubleThreshold(edges.sDiag, 0.8, 0.6);

    /*-------------  进度条参数实时读取 -------------*/
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
        4.000000, 10.266667, -312.000000,
        0.000000, 41.833333, -120.000000,
        0.000000, 0.088889, 1.000000
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
/// @return 校正之后的图片
cv::Mat color_correction(const cv::Mat& frame)
{
    // BGR -> LAB
    cv::Mat lab;
    cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> lab_planes;
    cv::split(lab, lab_planes);          // lab_planes[0]=L, [1]=a, [2]=b

    // CLAHE on L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(4, 2));
    clahe->apply(lab_planes[0], lab_planes[0]);

    // a-channel: a' = clip(a - mean(a) + 128, 0, 255)
    cv::Scalar mean_a = cv::mean(lab_planes[1]);
    lab_planes[1] = lab_planes[1] - mean_a[0] + 128.0;
    cv::threshold(lab_planes[1], lab_planes[1], 255, 255, cv::THRESH_TRUNC);
    cv::threshold(lab_planes[1], lab_planes[1], 0, 0, cv::THRESH_TOZERO);

    // Merge channels back
    cv::Mat lab_corrected;
    cv::merge(lab_planes, lab_corrected);

    // LAB -> BGR
    cv::Mat result;
    cv::cvtColor(lab_corrected, result, cv::COLOR_Lab2BGR);
    return result;
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

