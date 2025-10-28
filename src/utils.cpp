//#include "utils.hpp"
//#include <cmath>
//#include <algorithm>
//#include <fstream>
//#include <sstream>
//#include <iostream>
//#include <numeric>
//
//// sigmoid 和 softmax 函数已经很优化，保持不变
//float sigmoid(float x) {
//    return 1.0f / (1.0f + std::exp(-x));
//}
//
//std::vector<float> softmax(const std::vector<float>& x) {
//    std::vector<float> result(x.size());
//    float max_val = *std::max_element(x.begin(), x.end());
//    float sum = 0.0f;
//
//    for (size_t i = 0; i < x.size(); i++) {
//        result[i] = std::exp(x[i] - max_val);
//        sum += result[i];
//    }
//
//    for (size_t i = 0; i < result.size(); i++) {
//        result[i] /= sum;
//    }
//
//    return result;
//}
//
//// xywh2xyxy 保持不变
//std::vector<float> xywh2xyxy(const std::vector<float>& box) {
//    std::vector<float> result(4);
//    result[0] = box[0] - box[2] / 2.0f;
//    result[1] = box[1] - box[3] / 2.0f;
//    result[2] = box[0] + box[2] / 2.0f;
//    result[3] = box[1] + box[3] / 2.0f;
//    return result;
//}
//
//// 优化 make_grid：预分配空间，避免重复计算
//std::vector<std::vector<std::vector<std::vector<float>>>> make_grid(int h, int w, int anchor_num) {
//    std::vector<std::vector<std::vector<std::vector<float>>>> grid(
//        h, std::vector<std::vector<std::vector<float>>>(
//            w, std::vector<std::vector<float>>(
//                anchor_num, std::vector<float>(2, 0.0f)
//            )
//        )
//    );
//
//    for (int i = 0; i < h; i++) {
//        for (int j = 0; j < w; j++) {
//            for (int k = 0; k < anchor_num; k++) {
//                grid[i][j][k][0] = static_cast<float>(j);
//                grid[i][j][k][1] = static_cast<float>(i);
//            }
//        }
//    }
//
//    return grid;
//}
//
//// 重点优化 handel_preds 函数
//std::vector<std::vector<float>> handel_preds(const std::vector<ncnn::Mat>& preds, const ModelConfig& cfg) {
//    // 解析anchors
//    int num_scales = preds.size() / 3;
//    std::vector<std::vector<std::vector<float>>> anchors_3d(num_scales);
//
//    for (int i = 0; i < num_scales; i++) {
//        anchors_3d[i].resize(cfg.anchor_num);
//        for (int j = 0; j < cfg.anchor_num; j++) {
//            anchors_3d[i][j] = {
//                cfg.anchors[i * cfg.anchor_num * 2 + j * 2],
//                cfg.anchors[i * cfg.anchor_num * 2 + j * 2 + 1]
//            };
//        }
//    }
//
//    std::vector<std::vector<float>> output;
//
//    for (int i = 0; i < num_scales; i++) {
//        const ncnn::Mat& reg_preds = preds[i * 3];
//        const ncnn::Mat& obj_preds = preds[i * 3 + 1];
//        const ncnn::Mat& cls_preds = preds[i * 3 + 2];
//
//        int h = reg_preds.h;
//        int w = reg_preds.w;
//        int num_classes = cfg.classes;
//        float stride = static_cast<float>(cfg.height) / h;
//
//        // 预分配空间，避免重复分配
//        size_t total_boxes = static_cast<size_t>(h * w * cfg.anchor_num);
//        std::vector<std::vector<float>> batch_bboxes;
//        batch_bboxes.reserve(total_boxes);
//
//        // 预计算网格坐标
//        auto grid = make_grid(h, w, cfg.anchor_num);
//
//        // 预计算sigmoid值缓存（可选，但会增加内存使用）
//        // 这里选择直接计算，因为sigmoid本身很快
//
//        // 优化主循环：减少嵌套，提前计算常用值
//        for (int y = 0; y < h; y++) {
//            for (int x = 0; x < w; x++) {
//                // 预读取当前x,y位置的所有类别预测
//                std::vector<float> cls_scores(num_classes);
//                for (int cls = 0; cls < num_classes; cls++) {
//                    cls_scores[cls] = cls_preds.channel(cls).row(y)[x];
//                }
//                std::vector<float> softmax_cls = softmax(cls_scores);
//
//                for (int a = 0; a < cfg.anchor_num; a++) {
//                    std::vector<float> anchor_box(5 + num_classes);
//
//                    // 读取坐标预测
//                    int reg_base_idx = a * 4;
//                    float reg0 = reg_preds.channel(reg_base_idx).row(y)[x];
//                    float reg1 = reg_preds.channel(reg_base_idx + 1).row(y)[x];
//                    float reg2 = reg_preds.channel(reg_base_idx + 2).row(y)[x];
//                    float reg3 = reg_preds.channel(reg_base_idx + 3).row(y)[x];
//
//                    // 读取目标性预测
//                    float obj_score = obj_preds.channel(a).row(y)[x];
//
//                    // 计算anchor box
//                    float cx = (sigmoid(reg0) * 2.0f - 0.5f) + grid[y][x][a][0];
//                    float cy = (sigmoid(reg1) * 2.0f - 0.5f) + grid[y][x][a][1];
//                    anchor_box[0] = cx * stride;
//                    anchor_box[1] = cy * stride;
//
//                    float bw = std::pow(sigmoid(reg2) * 2.0f, 2);
//                    float bh = std::pow(sigmoid(reg3) * 2.0f, 2);
//                    anchor_box[2] = bw * anchors_3d[i][a][0];
//                    anchor_box[3] = bh * anchors_3d[i][a][1];
//
//                    anchor_box[4] = sigmoid(obj_score);
//
//                    // 使用预计算的softmax结果
//                    for (int cls_idx = 0; cls_idx < num_classes; cls_idx++) {
//                        anchor_box[5 + cls_idx] = softmax_cls[cls_idx];
//                    }
//
//                    batch_bboxes.emplace_back(std::move(anchor_box));
//                }
//            }
//        }
//
//        // 合并到输出
//        output.insert(output.end(), batch_bboxes.begin(), batch_bboxes.end());
//    }
//
//    return output;
//}
//
//// 优化 multiclass_non_max_suppression 函数
//std::vector<std::vector<float>> multiclass_non_max_suppression(
//    const std::vector<std::vector<float>>& prediction,
//    float conf_thres, float iou_thres) {
//
//    if (prediction.empty()) {
//        return {};
//    }
//
//    int num_classes = prediction[0].size() - 5;
//
//    // 第一步：过滤低置信度预测，同时计算最终得分
//    std::vector<std::vector<float>> filtered_preds;
//    std::vector<float> scores;
//    std::vector<int> class_ids;
//
//    filtered_preds.reserve(prediction.size());
//    scores.reserve(prediction.size());
//    class_ids.reserve(prediction.size());
//
//    for (const auto& pred : prediction) {
//        float obj_score = pred[4];
//        if (obj_score <= conf_thres) continue;
//
//        // 找到最大类别得分
//        auto max_it = std::max_element(pred.begin() + 5, pred.end());
//        int class_id = static_cast<int>(std::distance(pred.begin() + 5, max_it));
//        float class_score = *max_it;
//        float final_score = obj_score * class_score;
//
//        if (final_score > conf_thres) {
//            filtered_preds.push_back(pred);
//            scores.push_back(final_score);
//            class_ids.push_back(class_id);
//        }
//    }
//
//    if (filtered_preds.empty()) {
//        return {};
//    }
//
//    // 转换为xyxy格式
//    std::vector<std::vector<float>> boxes_xyxy;
//    boxes_xyxy.reserve(filtered_preds.size());
//    for (const auto& pred : filtered_preds) {
//        boxes_xyxy.push_back(xywh2xyxy({ pred[0], pred[1], pred[2], pred[3] }));
//    }
//
//    // 获取唯一类别
//    std::vector<int> unique_classes = class_ids;
//    std::sort(unique_classes.begin(), unique_classes.end());
//    unique_classes.erase(std::unique(unique_classes.begin(), unique_classes.end()), unique_classes.end());
//
//    std::vector<std::vector<float>> nms_results;
//
//    // 对每个类别进行NMS
//    for (int cls : unique_classes) {
//        // 收集当前类别的索引
//        std::vector<int> cls_indices;
//        for (size_t i = 0; i < class_ids.size(); i++) {
//            if (class_ids[i] == cls) {
//                cls_indices.push_back(static_cast<int>(i));
//            }
//        }
//
//        if (cls_indices.empty()) continue;
//
//        // 按得分降序排序
//        std::sort(cls_indices.begin(), cls_indices.end(),
//            [&scores](int a, int b) { return scores[a] > scores[b]; });
//
//        // NMS主循环
//        std::vector<int> keep_indices;
//        while (!cls_indices.empty()) {
//            int current_idx = cls_indices[0];
//            keep_indices.push_back(current_idx);
//
//            if (cls_indices.size() == 1) break;
//
//            const std::vector<float>& current_box = boxes_xyxy[current_idx];
//            std::vector<int> next_indices;
//
//            // 计算IoU并过滤
//            for (size_t i = 1; i < cls_indices.size(); i++) {
//                int idx = cls_indices[i];
//                const std::vector<float>& other_box = boxes_xyxy[idx];
//
//                // 计算IoU
//                float x1 = std::max(current_box[0], other_box[0]);
//                float y1 = std::max(current_box[1], other_box[1]);
//                float x2 = std::min(current_box[2], other_box[2]);
//                float y2 = std::min(current_box[3], other_box[3]);
//
//                float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
//                float union_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1]) +
//                    (other_box[2] - other_box[0]) * (other_box[3] - other_box[1]) -
//                    intersection;
//
//                float iou = intersection / union_area;
//
//                if (iou < iou_thres) {
//                    next_indices.push_back(idx);
//                }
//            }
//
//            cls_indices = std::move(next_indices);
//        }
//
//        // 保留结果
//        for (int idx : keep_indices) {
//            nms_results.push_back(filtered_preds[idx]);
//        }
//    }
//
//    return nms_results;
//}
//
//// load_datafile 和 load_label_names 保持不变
//ModelConfig load_datafile(const std::string& data_path) {
//    ModelConfig cfg;
//
//    std::ifstream file(data_path);
//    if (!file.is_open()) {
//        throw std::runtime_error("无法打开配置文件: " + data_path);
//    }
//
//    std::string line;
//    while (std::getline(file, line)) {
//        if (line.empty() || line[0] == '[') {
//            continue;
//        }
//
//        size_t pos = line.find('=');
//        if (pos != std::string::npos) {
//            std::string key = line.substr(0, pos);
//            std::string value = line.substr(pos + 1);
//
//            key.erase(0, key.find_first_not_of(" \t"));
//            key.erase(key.find_last_not_of(" \t") + 1);
//            value.erase(0, value.find_first_not_of(" \t"));
//            value.erase(value.find_last_not_of(" \t") + 1);
//
//            if (key == "model_name") cfg.model_name = value;
//            else if (key == "epochs") cfg.epochs = std::stoi(value);
//            else if (key == "steps") {
//                std::stringstream ss(value);
//                std::string item;
//                while (std::getline(ss, item, ',')) {
//                    cfg.steps.push_back(std::stoi(item));
//                }
//            }
//            else if (key == "batch_size") cfg.batch_size = std::stoi(value);
//            else if (key == "subdivisions") cfg.subdivisions = std::stoi(value);
//            else if (key == "learning_rate") cfg.learning_rate = std::stof(value);
//            else if (key == "pre_weights") cfg.pre_weights = value;
//            else if (key == "classes") cfg.classes = std::stoi(value);
//            else if (key == "width") cfg.width = std::stoi(value);
//            else if (key == "height") cfg.height = std::stoi(value);
//            else if (key == "anchor_num") cfg.anchor_num = std::stoi(value);
//            else if (key == "anchors") {
//                std::stringstream ss(value);
//                std::string item;
//                while (std::getline(ss, item, ',')) {
//                    cfg.anchors.push_back(std::stof(item));
//                }
//            }
//            else if (key == "val") cfg.val = value;
//            else if (key == "train") cfg.train = value;
//            else if (key == "names") cfg.names = value;
//        }
//    }
//
//    return cfg;
//}
//
//std::vector<std::string> load_label_names(const std::string& names_path) {
//    std::vector<std::string> label_names;
//    std::ifstream file(names_path);
//    std::string line;
//
//    while (std::getline(file, line)) {
//        if (!line.empty()) {
//            label_names.push_back(line);
//        }
//    }
//
//    return label_names;
//}
