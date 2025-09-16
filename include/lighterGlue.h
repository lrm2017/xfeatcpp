#ifndef LIGTHERGLUE_H
#define LIGTHERGLUE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <string>
#include "ort_run.h"

class LighterGlue {
public:
    LighterGlue(const std::string& model_path);
    ~LighterGlue();
    bool initializeOrt();
    bool loadModel();
    bool preprocess(std::vector<Ort::Value>& inputs);
    bool postprocess(std::vector<Ort::Value>& outputs);
    bool run(const std::vector<cv::Point2f>& keypoints1,
             const std::vector<cv::Point2f>& keypoints2,
             const cv::Mat& descriptors1,
             const cv::Mat& descriptors2,
             std::vector<cv::DMatch>& matches,
             std::vector<float>& scores);
    void match(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches);

    std::vector<Ort::Value> preprocessMatchingData(
        const std::vector<cv::Point2f>& keypoints1,
        const std::vector<cv::Point2f>& keypoints2,
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2);
    bool postprocessMatchingResults(const std::vector<Ort::Value>& data,
                                    std::vector<cv::DMatch>& matches,
                                    std::vector<float>& scores);

private:
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::SessionOptions> ort_session_options_;
    std::unique_ptr<Ort::MemoryInfo> ort_memory_info_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> ort_allocator_;
    std::string model_path_;
    std::unique_ptr<Ort::Session> session_;
    
    // 用于存储当前匹配数据
    std::vector<cv::Point2f> current_keypoints1_;
    std::vector<cv::Point2f> current_keypoints2_;
    cv::Mat current_descriptors1_;
    cv::Mat current_descriptors2_;
    std::vector<cv::DMatch>* output_matches_;
    std::vector<float>* output_scores_;
    
    // 输入输出名称和形状
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    // 输入输出名称的C字符串版本
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;
    
    // 辅助方法
    std::vector<cv::Point2f> normalizeKeypoints(const std::vector<cv::Point2f>& keypoints, int height, int width);
    Ort::Value createTensor(const float* data, size_t data_size, const std::vector<int64_t>& shape);
};


#endif