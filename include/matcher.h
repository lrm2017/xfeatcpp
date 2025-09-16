#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include "ort_run.h"

class Matcher : public OrtRun {
public:
    /**
     * 构造函数
     * @param model_path ONNX模型路径
     * @param force_cpu 是否强制使用CPU
     * @param threshold 匹配阈值
     */
    explicit Matcher(const std::string& model_path, const std::string& name, bool force_cpu = true, float threshold = 0.5f, cv::Size input_wh={3072, 2048});
    
    /**
     * 析构函数
     */
    ~Matcher() = default;
    
    /**
     * 运行匹配
     * @param keypoints1 第一组关键点
     * @param keypoints2 第二组关键点
     * @param descriptors1 第一组描述符
     * @param descriptors2 第二组描述符
     * @param matches 输出的匹配对
     * @param scores 输出的匹配分数
     * @return 是否成功
     */
    bool run(const std::vector<cv::Point2f>& keypoints1,
             const std::vector<cv::Point2f>& keypoints2,
             const cv::Mat& descriptors1,
             const cv::Mat& descriptors2,
             std::vector<cv::DMatch>& matches,
             std::vector<float>& scores);
    
    /**
     * 设置匹配阈值
     * @param threshold 新的阈值
     */
    void setThreshold(float threshold) { threshold_ = threshold; }
    
    /**
     * 获取当前阈值
     * @return 当前阈值
     */
    float getThreshold() const { return threshold_; }

    void setInputWH(cv::Size input_wh={3072, 2048}) { input_wh_ = input_wh; }

protected:
    /**
     * 预处理匹配数据 - 重写基类虚函数
     * @param inputs 输入张量
     * @return 是否成功
     */
    bool preprocess(std::vector<Ort::Value>& inputs) override;
    
    /**
     * 后处理匹配结果 - 重写基类虚函数
     * @param outputs 模型输出
     * @return 是否成功
     */
    bool postprocess(std::vector<Ort::Value>& outputs) override;

private:
    // 匹配阈值
    float threshold_;

    // 模型输入的形状
    cv::Size input_wh_;
    
    // 当前处理的匹配数据
    std::vector<cv::Point2f> current_keypoints1_;
    std::vector<cv::Point2f> current_keypoints2_;
    cv::Mat current_descriptors1_;
    cv::Mat current_descriptors2_;
    
    // 输出的匹配结果
    std::vector<cv::DMatch>* output_matches_;
    std::vector<float>* output_scores_;
    
    /**
     * 预处理匹配数据
     * @param keypoints1 第一组关键点
     * @param keypoints2 第二组关键点
     * @param descriptors1 第一组描述符
     * @param descriptors2 第二组描述符
     * @return 预处理后的数据
     */
    std::vector<Ort::Value> preprocessMatchingData(const std::vector<cv::Point2f>& keypoints1,
                                                   const std::vector<cv::Point2f>& keypoints2,
                                                   const cv::Mat& descriptors1,
                                                   const cv::Mat& descriptors2);
    
    /**
     * 后处理匹配结果
     * @param data ONNX模型输出
     * @param matches 输出的匹配对
     * @param scores 输出的匹配分数
     * @return 是否成功
     */
    bool postprocessMatchingResults(const std::vector<Ort::Value>& data,
                                   std::vector<cv::DMatch>& matches,
                                   std::vector<float>& scores);
    
    /**
     * 归一化关键点坐标
     * @param keypoints 关键点
     * @param height 图像高度
     * @param width 图像宽度
     * @return 归一化后的关键点
     */
    std::vector<cv::Point2f> normalizeKeypoints(const std::vector<cv::Point2f>& keypoints,
                                               int height, int width);
};