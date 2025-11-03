#include <opencv2/opencv.hpp>
#include "Cluster.hpp"
#include <cmath>
#include "mode.hpp"

cv::Mat ipm_mat = (cv::Mat_<double>(3, 3) <<
    6.452682f, 41.380027f, -359.987889f,
    -0.63295f, 64.321531f, -635.834781f,
    -0.001352f, 0.103485f, 1.0f
    );

/// @brief 可视化聚类结果，工具函数
/// @param densityResult KDE曲线
/// @param crossPointX 基准交点
/// @param lineLengths 直线长度
/// @param clusters 聚类结果 (传入的是索引数组)
/// @param windowName 窗口名称
void visualizeKDEWithClusters(const std::vector<double>& densityResult,
	const std::vector<int>& crossPointX,
	const std::vector<int>& lineLengths,
	const std::vector<std::vector<int>>& clusters,
	const std::string& windowName = "KDE with Clusters")
{
	// --- MODIFICATION START ---
	// 定义与主函数一致的KDE范围参数
	const int KDE_MIN_X = -120;
	const int KDE_MAX_X = 360;
	const int KDE_RANGE_TOTAL = KDE_MAX_X - KDE_MIN_X; // 480
	const int KDE_MAX_INDEX = KDE_RANGE_TOTAL - 1;      // 479

	// 扩展画布宽度以适应新的范围
	static cv::Mat display(400, KDE_RANGE_TOTAL, CV_8UC3, cv::Scalar(0, 0, 0));
	display.setTo(cv::Scalar(0, 0, 0));
	// --- MODIFICATION END ---

	// 找到最大值用于归一化
	double maxDensity = *std::max_element(densityResult.begin(), densityResult.end());
	if (maxDensity <= 0) maxDensity = 1.0;

	// 绘制更新后的坐标轴
	cv::line(display, cv::Point(0, 350), cv::Point(KDE_MAX_INDEX, 350), cv::Scalar(100, 100, 100), 1);
	cv::line(display, cv::Point(0, 350), cv::Point(0, 50), cv::Scalar(100, 100, 100), 1);

	// --- MODIFICATION START ---
	// 添加中心线和范围标签，便于观察
	int midline_pixel = 120 - KDE_MIN_X; // 物理中心 120 对应的像素位置 120 - (-120) = 240
	cv::line(display, cv::Point(midline_pixel, 350), cv::Point(midline_pixel, 50), cv::Scalar(50, 50, 50), 1);
	cv::putText(display, "x=120", cv::Point(midline_pixel - 20, 365), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
	cv::putText(display, std::to_string(KDE_MIN_X), cv::Point(5, 365), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
	cv::putText(display, std::to_string(KDE_MAX_X), cv::Point(KDE_MAX_INDEX - 25, 365), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
	// --- MODIFICATION END ---


	// 绘制核密度曲线
	for (int i = 1; i < KDE_RANGE_TOTAL; ++i) { // 修改循环边界
		int y1 = 350 - static_cast<int>((densityResult[i - 1] / maxDensity) * 300);
		int y2 = 350 - static_cast<int>((densityResult[i] / maxDensity) * 300);
		y1 = std::max(50, std::min(350, y1));
		y2 = std::max(50, std::min(350, y2));

		cv::line(display, cv::Point(i - 1, y1), cv::Point(i, y2), cv::Scalar(0, 255, 0), 2);
	}

	// 显示所有交点
	if (!crossPointX.empty() && !lineLengths.empty()) {
		// 找到最大线段长度用于归一化点的大小
		int maxLength = *std::max_element(lineLengths.begin(), lineLengths.end());
		if (maxLength <= 0) maxLength = 1;

		for (size_t i = 0; i < crossPointX.size(); ++i) {
			int x_coord = crossPointX[i]; // x_coord是物理坐标
			// --- MODIFICATION START ---
			// 检查物理坐标是否在新的显示范围内
			if (x_coord >= KDE_MIN_X && x_coord < KDE_MAX_X) {
				// 将物理坐标转换为像素索引
				int pixelX = x_coord - KDE_MIN_X;
				// --- MODIFICATION END ---

				// 根据线段长度确定点的大小和颜色
				float sizeRatio = static_cast<float>(lineLengths[i]) / maxLength;
				int radius = cvRound(2 + 3 * sizeRatio); // 半径从2到5像素
				int colorIntensity = cvRound(200 + 55 * sizeRatio); // 颜色强度

				cv::circle(display, cv::Point(pixelX, 350), radius,
					cv::Scalar(colorIntensity, colorIntensity, 255), -1);
			}
		}
	}

	// 显示找到的簇
	// --- MODIFICATION START ---
	// 簇的颜色和绘制逻辑基本不变，因为它本身就是索引
	// --- MODIFICATION END ---
	std::vector<cv::Scalar> clusterColors = {
		cv::Scalar(255, 0, 0),   // 蓝色
		cv::Scalar(0, 255, 255), // 黄色
		cv::Scalar(255, 0, 255), // 粉色
		cv::Scalar(0, 165, 255)  // 橙色
	};

	for (size_t i = 0; i < clusters.size(); ++i) {
		const auto& cluster_indices = clusters[i]; // 重命名以明确是索引
		if (cluster_indices.empty()) continue;

		cv::Scalar color = clusterColors[i % clusterColors.size()];

		// 绘制簇区域 (cluster_indices中的值已经是像素索引)
		int clusterMinX_pixel = *std::min_element(cluster_indices.begin(), cluster_indices.end());
		int clusterMaxX_pixel = *std::max_element(cluster_indices.begin(), cluster_indices.end());

		cv::rectangle(display, cv::Point(clusterMinX_pixel, 50), cv::Point(clusterMaxX_pixel, 350),
			color, 1);

		// 标记簇中心
		int centerX_pixel = (clusterMinX_pixel + clusterMaxX_pixel) / 2;
		cv::line(display, cv::Point(centerX_pixel, 50), cv::Point(centerX_pixel, 350), color, 1);

		// 显示簇编号
		std::string clusterText = "C" + std::to_string(i + 1);
		cv::putText(display, clusterText, cv::Point(centerX_pixel - 5, 40),
			cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
	}

	// 显示统计信息
	std::stringstream info;
	info << "Points: " << crossPointX.size() << " Clusters: " << clusters.size();
	cv::putText(display, info.str(), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

	cv::imshow(windowName, display);
}

/// @brief 可视化工具，绘制直线
/// @param frame 要绘制的图像
/// @param descriptors 簇描述子
void drawClusterLines(cv::Mat& frame, const std::vector<ClusterDescriptor>& descriptors)
{
    int frameHeight = frame.rows;
    int frameWidth = frame.cols;

    // 裁剪参数
    const double cropStartRatio = 2.0 / 5.0;
    const double cropEndRatio = 5.0 / 6.0;
    int cropStartY = cvRound(frameHeight * cropStartRatio);
    int cropHeight = cvRound(frameHeight * cropEndRatio) - cropStartY;

    const int clusterWidth = 240;
    const int clusterHeight = 120;

    // 从小图到原图的缩放比例
    double scaleX = static_cast<double>(frameWidth) / clusterWidth;
    double scaleY = static_cast<double>(cropHeight) / clusterHeight;

    // 参考点映射（小图y=60对应裁剪区域中心）
    int referenceY = cvRound(60 * scaleY) + cropStartY;

    int startY = frameHeight - 1;
    int endY = frameHeight * 2 / 5;

    cv::Scalar greenColor(0, 255, 0);
    int thickness = 7;

    for (const auto& descriptor : descriptors) {
        double peakX_scaled = descriptor.peakX * scaleX;

        // 调整斜率计算，考虑y方向的缩放
        double slope = std::tan(descriptor.mainAngle) * (scaleY / scaleX);

        std::vector<cv::Point> linePoints;

        auto calculateX = [&](int y) -> double {
            return peakX_scaled + (y - referenceY) / slope;
            };

        auto calculateY = [&](int x) -> double {
            return referenceY + slope * (x - peakX_scaled);
            };

        double bottomX = calculateX(startY);
        if (bottomX >= 0 && bottomX <= frameWidth - 1) {
            linePoints.emplace_back(cvRound(bottomX), startY);
        }

        double topX = calculateX(endY);
        if (topX >= 0 && topX <= frameWidth - 1) {
            linePoints.emplace_back(cvRound(topX), endY);
        }

        double leftY = calculateY(0);
        if (leftY >= endY && leftY <= startY) {
            linePoints.emplace_back(0, cvRound(leftY));
        }

        double rightY = calculateY(frameWidth - 1);
        if (rightY >= endY && rightY <= startY) {
            linePoints.emplace_back(frameWidth - 1, cvRound(rightY));
        }

        if (linePoints.size() >= 2) {
            std::sort(linePoints.begin(), linePoints.end(),
                [](const cv::Point& a, const cv::Point& b) {
                    return a.y > b.y;
                });
            cv::Point bottomPoint = linePoints[0];
            cv::Point topPoint = linePoints.back();

            if (bottomPoint.y > topPoint.y) {
                cv::line(frame, bottomPoint, topPoint, greenColor, thickness);
                cv::circle(frame, cv::Point(cvRound(peakX_scaled), referenceY),
                    3, cv::Scalar(255, 0, 0), -1);
            }
        }
    }
}

/// @brief 输入原始的直线组，输出进行KED-DBSCAN处理后的簇描述子
/// @param lanes 原始霍夫变换融合结果
/// @param maxAngleThreshold 在鸟瞰图中，与水平方向的最大夹角阈值（单位：度）。小于此值的线段将被视为过于水平而提前过滤。默认45度。
/// @return 簇描述子
std::vector<ClusterDescriptor> lanesCluster(std::vector<cv::Vec4i> lanes, double maxAngleThreshold)
{
	// 如果需要过滤，则进行此步骤
	if (maxAngleThreshold > 0 && !lanes.empty()) {
		static const cv::Mat ipm_mat = (cv::Mat_<double>(3, 3) <<
			6.452682f, 41.380027f, -359.987889f,
			-0.63295f, 64.321531f, -635.834781f,
			-0.001352f, 0.103485f, 1.0f
			);
		double maxAngleRad = maxAngleThreshold * CV_PI / 180.0;
		std::vector<int> indicesToRemove;
		// 遍历所有原始线段
		for (size_t i = 0; i < lanes.size(); ++i) {
			const auto& line = lanes[i];
			// 将线段的两个端点从图像坐标转换到鸟瞰图坐标
			std::vector<cv::Point2d> img_points = { cv::Point2d(line[0], line[1]), cv::Point2d(line[2], line[3]) };
			std::vector<cv::Point2d> bev_points;
			cv::perspectiveTransform(img_points, bev_points, ipm_mat);
			// 计算鸟瞰图下的方向向量
			cv::Point2d dir_bev = bev_points[1] - bev_points[0];
			double length = std::sqrt(dir_bev.x * dir_bev.x + dir_bev.y * dir_bev.y);
			if (length < 1e-4) {
				// 如果线段长度在BEV下趋近于0，视为无效，也过滤掉
				indicesToRemove.push_back(i);
				continue;
			}
			// 计算与BEV坐标系中水平方向（X轴）的夹角
			double angle_with_horizontal = std::abs(std::atan2(dir_bev.y, dir_bev.x));
			// 如果这个角度小于阈值，说明线段在鸟瞰图下过于水平，需要过滤
			if (angle_with_horizontal < maxAngleRad) {
				indicesToRemove.push_back(i);
			}
		}
		// 按逆序移除，避免索引错乱
		for (auto it = indicesToRemove.rbegin(); it != indicesToRemove.rend(); ++it) {
			lanes.erase(lanes.begin() + *it);
		}
	}
	int referenceY = REFERENCE_Y; // 水平线交点参考线
	float kTreshold = 0.2f; // 水平线斜率阈值
	std::vector<int> crossPointX; // 存储每个线段的水平交点
	std::vector<int> lineLengths; // 存储每个线段的长度
	crossPointX.reserve(100);
	lineLengths.reserve(100);
	// 删掉太过水平的线段 (这是一个额外的、基于图像斜率的过滤)
	for (auto it = lanes.begin(); it != lanes.end();)
	{
		int x1 = (*it)[0];
		int y1 = (*it)[1];
		int x2 = (*it)[2];
		int y2 = (*it)[3];
		int dx = abs(x1 - x2);
		int dy = abs(y1 - y2);
		if (dx == 0)
		{
			(*it)[2] += 1;
			dx = 1;
		}
		if (1.0 * dy / dx < kTreshold)
			it = lanes.erase(it);
		else
		{
			// 计算线段长度
			double length = cvRound(sqrt(dx * dx + dy * dy));
			lineLengths.push_back(length);
			it++;
		}
	}
	// 计算得到与水平线的交点
	for (const auto& lane : lanes)
	{
		int x1 = lane[0];
		int y1 = lane[1];
		int x2 = lane[2];
		int y2 = lane[3];
		if (y1 == y2) continue; // 避免除以零，虽然前面已经过滤了水平线，但以防万一
		// 计算交点x坐标
		double t = (referenceY - y1) / static_cast<double>(y2 - y1);
		double x = x1 + t * (x2 - x1);
		int xInt = cvRound(x);
		crossPointX.push_back(xInt);
	}

	// --- MODIFICATION START ---
	// 定义KDE采样的新范围和参数
	const int KDE_MIN_X = -120;
	const int KDE_MAX_X = 360;
	const int KDE_RANGE_TOTAL = KDE_MAX_X - KDE_MIN_X; // 480
	const int KDE_MAX_INDEX = KDE_RANGE_TOTAL - 1;      // 479

	// 计算KDE核密度曲线，范围从-120到360，共480个采样点
	std::vector<double> densityResult(KDE_RANGE_TOTAL, 0.0);
	double bandwidth = 6.0; // 带宽保持不变
	double sqrt2pi = std::sqrt(2.0 * CV_PI); // 使用CV_PI
	double normalizer = 1.0 / (bandwidth * sqrt2pi);
	// 主循环：采样点
	for (int i = 0; i < KDE_RANGE_TOTAL; ++i) { // 修改循环边界
		double density = 0.0;
		// 将数组索引 i 映射到实际的物理 x 坐标
		double x = static_cast<double>(i) + KDE_MIN_X; // 修改坐标映射
		// 内循环：数据点
		for (size_t dataIdx = 0; dataIdx < crossPointX.size(); ++dataIdx) {
			double xi = static_cast<double>(crossPointX[dataIdx]);
			double weight = static_cast<double>(lineLengths[dataIdx]);
			// 高斯核计算
			double u = (x - xi) / bandwidth;
			density += std::exp(-0.5 * u * u) * normalizer * weight;
		}
		densityResult[i] = density;
	}
	// --- MODIFICATION END ---

	// 迭代寻找密度最高的簇
	std::vector<std::vector<int>> clusters; // 存储找到的x坐标簇(此处存储的仍是索引)
	std::vector<ClusterDescriptor> allDescriptors; // 存储所有找到的簇描述子
	std::vector<double> densityTemp = densityResult; // 临时密度副本用于迭代处理
	double initialMaxDensity = *std::max_element(densityResult.begin(), densityResult.end());
	double minDensityThreshold = initialMaxDensity * 0.2; // 最小密度阈值
	if (minDensityThreshold > 6) {
		minDensityThreshold = 6;
	}

	for (int clusterIdx = 0; clusterIdx < 6; ++clusterIdx) {
		// 找到当前密度最大的点
		auto maxIt = std::max_element(densityTemp.begin(), densityTemp.end());
		if (maxIt == densityTemp.end() || *maxIt < minDensityThreshold) {
			break;
		}
		int peakIndex = std::distance(densityTemp.begin(), maxIt); // peakIndex是数组索引
		double peakDensity = *maxIt;
		std::vector<int> currentClusterIndices; // 存储当前簇的索引
		std::vector<cv::Vec4i> currentLaneCluster; // 对应的直线簇
		std::queue<int> toVisit;
		std::vector<bool> visited(KDE_RANGE_TOTAL, false); // 修改visited大小
		toVisit.push(peakIndex);
		visited[peakIndex] = true;
		double growthThreshold = peakDensity * 0.3; // 生长阈值

		while (!toVisit.empty()) {
			int currentIndex = toVisit.front();
			toVisit.pop();
			currentClusterIndices.push_back(currentIndex);

			// 将交点在该x坐标的线段归入簇
			// 注意：这里的比较需要将索引转换为坐标
			int currentXCoord = currentIndex + KDE_MIN_X;
			for (size_t i = 0; i < crossPointX.size(); ++i) {
				if (crossPointX[i] == currentXCoord) {
					// 避免重复添加同一条线段
					if (std::find(currentLaneCluster.begin(), currentLaneCluster.end(), lanes[i]) == currentLaneCluster.end()) {
						currentLaneCluster.push_back(lanes[i]);
					}
				}
			}
			// 探索邻近点
			if (currentIndex > 0 && !visited[currentIndex - 1] && densityTemp[currentIndex - 1] >= growthThreshold) {
				toVisit.push(currentIndex - 1);
				visited[currentIndex - 1] = true;
			}
			if (currentIndex < KDE_MAX_INDEX && !visited[currentIndex + 1] && densityTemp[currentIndex + 1] >= growthThreshold) { // 修改边界检查
				toVisit.push(currentIndex + 1);
				visited[currentIndex + 1] = true;
			}
		}

		// 将找到的簇添加到结果中
		if (!currentClusterIndices.empty()) {
			clusters.push_back(currentClusterIndices);
			// 计算当前簇的主导角度
			if (!currentLaneCluster.empty()) {
				std::vector<double> angles;
				std::vector<double> angleWeights;
				for (const auto& lane : currentLaneCluster) {
					int x1 = lane[0], y1 = lane[1], x2 = lane[2], y2 = lane[3];
					double dx = x2 - x1;
					double dy = y2 - y1;
					double angle = std::atan2(dy, dx);
					double length = std::sqrt(dx * dx + dy * dy);
					angles.push_back(angle);
					angleWeights.push_back(length);
				}
				// 计算角度KDE (此部分逻辑未变)
				double angleBandwidth = 0.3;
				int angleSamples = 180;
				std::vector<double> angleDensity(angleSamples, 0.0);
				double angleNormalizer = 1.0 / (angleBandwidth * std::sqrt(2.0 * CV_PI)); // 使用CV_PI
				double angleMin = -CV_PI / 2.0;
				double angleMax = CV_PI / 2.0;
				double angleStep = (angleMax - angleMin) / angleSamples;
				for (int i = 0; i < angleSamples; ++i) {
					double sampleAngle = angleMin + i * angleStep;
					double density = 0.0;
					for (size_t j = 0; j < angles.size(); ++j) {
						double u = (sampleAngle - angles[j]) / angleBandwidth;
						density += std::exp(-0.5 * u * u) * angleNormalizer * angleWeights[j];
					}
					angleDensity[i] = density;
				}
				// 找到角度KDE的最大密度点
				auto maxAngleIt = std::max_element(angleDensity.begin(), angleDensity.end());
				int maxAngleIdx = std::distance(angleDensity.begin(), maxAngleIt);
				double mainAngle = angleMin + maxAngleIdx * angleStep;

				// 创建簇描述子
				ClusterDescriptor descriptor;
				// --- MODIFICATION START ---
				// peakX 存储的是物理坐标，而不是索引
				descriptor.peakX = peakIndex + KDE_MIN_X;
				// --- MODIFICATION END ---
				descriptor.mainAngle = mainAngle;
				descriptor.peakDensity = peakDensity; // 存储峰值密度以供后续筛选
				allDescriptors.push_back(descriptor);
			}
			// 将簇及其边缘区域置零
			int clusterMinIndex = *std::min_element(currentClusterIndices.begin(), currentClusterIndices.end());
			int clusterMaxIndex = *std::max_element(currentClusterIndices.begin(), currentClusterIndices.end());
			int margin = cvRound(bandwidth * 2);
			for (int i = std::max(0, clusterMinIndex - margin); i <= std::min(KDE_MAX_INDEX, clusterMaxIndex + margin); ++i) { // 修改边界
				densityTemp[i] = 0.0;
			}
		}
	}
	//可视化结果 (此部分逻辑未改变)
#ifdef WITH_IMSHOW
	// 注意：可视化函数可能也需要调整以适应新的范围，但根据要求，我们不更改它。
	// 如果需要可视化，您可能需要修改 `visualizeKDEWithClusters` 的实现。
	visualizeKDEWithClusters(densityResult, crossPointX, lineLengths, clusters);
	cv::waitKey(1);
#endif

	const int MIDLINE_X = 120; // 图像中线（物理坐标）
	std::vector<ClusterDescriptor> finalDescriptors;
	// 将找到的所有描述子分为左右两组
	std::vector<ClusterDescriptor> leftDescriptors;
	std::vector<ClusterDescriptor> rightDescriptors;
	for (const auto& desc : allDescriptors) {
		// 由于 desc.peakX 现在是物理坐标，此比较逻辑保持正确
		if (desc.peakX < MIDLINE_X) {
			leftDescriptors.push_back(desc);
		}
		else { // peakX >= MIDLINE_X 的归为右簇
			rightDescriptors.push_back(desc);
		}
	}
	// 在左组中寻找峰值密度最高的描述子
	if (!leftDescriptors.empty()) {
		auto max_it_left = std::max_element(leftDescriptors.begin(), leftDescriptors.end(),
			[](const ClusterDescriptor& a, const ClusterDescriptor& b) {
				return a.peakDensity < b.peakDensity;
			});
		finalDescriptors.push_back(*max_it_left);
	}
	// 在右组中寻找峰值密度最高的描述子
	if (!rightDescriptors.empty()) {
		auto max_it_right = std::max_element(rightDescriptors.begin(), rightDescriptors.end(),
			[](const ClusterDescriptor& a, const ClusterDescriptor& b) {
				return a.peakDensity < b.peakDensity;
			});
		finalDescriptors.push_back(*max_it_right);
	}
	// 返回最终的筛选结果
	return finalDescriptors;
}

/// @brief 利用逆透视结果进行角度过滤
/// @param lanes 要过滤的车道线簇描述子
/// @param maxAngleThreshold 与垂直方向的最大夹角阈值，单位为度，默认45度
void filterLanes(std::vector<ClusterDescriptor>& lanes, double maxAngleThreshold = 45.0)
{
    constexpr int IMG_H = 120;
    constexpr int REF_Y = 60;
    constexpr int BOTTOM_Y = IMG_H - 1;

    double maxAngleRad = maxAngleThreshold * CV_PI / 180.0;

    if (lanes.empty()) return;

    std::vector<int> indicesToRemove;

    for (size_t i = 0; i < lanes.size(); ++i) {
        auto& lane = lanes[i];

        double tanAngle = std::tan(lane.mainAngle);
        if (std::abs(tanAngle) < 1e-4) {
            tanAngle = (lane.mainAngle >= 0) ? 1e-4 : -1e-4;
        }

        cv::Point2d p1_img(lane.peakX, REF_Y);
        cv::Point2d p2_img(lane.peakX + (BOTTOM_Y - REF_Y) / tanAngle, BOTTOM_Y);

        std::vector<cv::Point2d> img_points = { p1_img, p2_img };
        std::vector<cv::Point2d> bev_points;
        cv::perspectiveTransform(img_points, bev_points, ipm_mat);

        cv::Point2d dir_bev = bev_points[1] - bev_points[0];
        double length = std::sqrt(dir_bev.x * dir_bev.x + dir_bev.y * dir_bev.y);

        if (length < 1e-4) {
            indicesToRemove.push_back(i);
            continue;
        }

        // 计算与水平方向的夹角
        double angle = std::abs(std::atan2(dir_bev.y, dir_bev.x));

        // 如果角度小于阈值，说明过于水平，需要过滤
        if (angle < maxAngleRad) {
            indicesToRemove.push_back(i);
        }
    }

    for (auto it = indicesToRemove.rbegin(); it != indicesToRemove.rend(); ++it) {
        if (*it < lanes.size()) {
            lanes.erase(lanes.begin() + *it);
        }
    }
}