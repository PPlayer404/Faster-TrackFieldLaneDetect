//#ifndef UTILS_HPP
//#define UTILS_HPP
//
//
//
//#include <ncnn/net.h>
//#include <opencv2/opencv.hpp>
//#include <vector>
//#include <string>
//#include <map>
//
//struct ModelConfig {
//    std::string model_name;
//    int epochs;
//    std::vector<int> steps;
//    int batch_size;
//    int subdivisions;
//    float learning_rate;
//    std::string pre_weights;
//    int classes;
//    int width;
//    int height;
//    int anchor_num;
//    std::vector<float> anchors;
//    std::string val;
//    std::string train;
//    std::string names;
//};
//
//// 数学函数
//float sigmoid(float x);
//std::vector<float> softmax(const std::vector<float>& x);
//
//// 坐标转换
//std::vector<float> xywh2xyxy(const std::vector<float>& box);
//
//// 网格生成
//std::vector<std::vector<std::vector<std::vector<float>>>> make_grid(int h, int w, int anchor_num);
//
//// 核心预测处理
//std::vector<std::vector<float>> handel_preds(
//    const std::vector<ncnn::Mat>& preds,
//    const ModelConfig& cfg);
//
//// NMS
//std::vector<std::vector<float>> multiclass_non_max_suppression(
//    const std::vector<std::vector<float>>& prediction,
//    float conf_thres = 0.5f,
//    float iou_thres = 0.4f);
//
//// 配置加载
//ModelConfig load_datafile(const std::string& data_path);
//
//// 辅助函数
//std::vector<std::string> load_label_names(const std::string& names_path);
//
//#endif