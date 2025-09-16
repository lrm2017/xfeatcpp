#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include "ort_run.h"

class FeatExtractor : public OrtRun {
public:
    /**
     * 构造函数
     * @param model_path ONNX模型路径
     * @param force_cpu 是否强制使用CPU
     */
    explicit FeatExtractor(const std::string& model_path, const std::string& name, bool force_cpu = true);
    
    /**
     * 析构函数
     */
    ~FeatExtractor() = default;
    
    /**
     * 运行特征提取
     * @param img 输入图像
     * @param keypoints 输出的关键点
     * @param descriptors 输出的描述符
     * @return 是否成功
     */
    bool run(const cv::Mat& img, 
             std::vector<cv::Point2f>& keypoints, 
             cv::Mat& descriptors);

    // 获取模型输入的形状
    cv::Size getInputWH();

protected:
    /**
     * 预处理图像 - 重写基类虚函数
     * @param inputs 输入张量
     * @return 是否成功
     */
    bool preprocess(std::vector<Ort::Value>& inputs) override;
    
    /**
     * 后处理特征提取结果 - 重写基类虚函数
     * @param outputs 模型输出
     * @return 是否成功
     */
    bool postprocess(std::vector<Ort::Value>& outputs) override;

private:
    // 当前处理的图像
    cv::Mat current_image_;
    
    // 输出的关键点和描述符
    std::vector<cv::Point2f>* output_keypoints_;
    cv::Mat* output_descriptors_;
    
    /**
     * 预处理图像数据
     * @param img 输入图像
     * @return 预处理后的图像数据
     */
    std::vector<float> preprocessImage(const cv::Mat& img);
    
    /**
     * 后处理特征提取结果
     * @param data ONNX模型输出
     * @param keypoints 输出的关键点
     * @param descriptors 输出的描述符
     * @return 是否成功
     */
    bool postprocessFeatures(const std::vector<Ort::Value>& data, 
                            std::vector<cv::Point2f>& keypoints, 
                            cv::Mat& descriptors);
};