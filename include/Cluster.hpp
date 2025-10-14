#ifndef CLUSTER_HPP
#define CLUSTER_HPP
#include "opencv2/opencv.hpp"

struct ClusterDescriptor
{
    int peakX;          //起始生长点（x坐标）
    double mainAngle;   //主导角度（弧度）
};

std::vector<ClusterDescriptor> lanesCluster(std::vector<cv::Vec4i> lanes);
void drawClusterLines(cv::Mat& frame, const std::vector<ClusterDescriptor>& descriptors);
void filterLanes(std::vector<ClusterDescriptor>& lanes, double maxAngleThreshold);

#define REFERENCE_Y 60

#endif 
