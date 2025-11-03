#ifndef CLUSTER_HPP
#define CLUSTER_HPP
#include "opencv2/opencv.hpp"

struct ClusterDescriptor
{
    int peakX;          //起始生长点（x坐标）
    double mainAngle;   //主导角度（弧度）
    double peakDensity; //峰值强度
};

extern cv::Mat ipm_mat;

std::vector<ClusterDescriptor> lanesCluster(std::vector<cv::Vec4i> lanes, double maxAngleThreshold = 45);
void drawClusterLines(cv::Mat& frame, const std::vector<ClusterDescriptor>& descriptors);
void filterLanes(std::vector<ClusterDescriptor>& lanes, double maxAngleThreshold);

#define REFERENCE_Y 60

#endif 
