#ifndef LANE_HPP
#define LANE_HPP

#include <opencv2/opencv.hpp>

struct HSV_Lane
{
	int H_High = 100;
	int S_High = 100;
	int V_High = 100;
	int H_Low = 100;
	int S_Low = 100;
	int V_Low = 100;

};

typedef struct FDEF_Result
{
	cv::Mat sDiag;
	cv::Mat sVert;
} FDEF_Result;

std::vector<cv::Vec4i> detectLanes(cv::Mat& processed_img, HSV_Lane HSV);
cv::Mat color_correction(const cv::Mat& frame);
void Lane_init(void);
cv::Mat labGetBlob(const cv::Mat& bgr, const int th[6]);
void FastGuidedFilter(cv::Mat& srcImage, cv::Mat& guidedImage, cv::Mat& outputImage, int filterSize, double eps, int samplingRate);
cv::Mat fastGuidedFilter_2(cv::Mat I_org, cv::Mat p_org, int r, double eps, int s);

#endif
