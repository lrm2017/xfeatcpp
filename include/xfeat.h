#ifndef XFEAT_H
#define XFEAT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <string>

// ONNX Runtime 前向声明
namespace Ort {
    class Env;
    class Session;
    class SessionOptions;
    class MemoryInfo;
    class AllocatorWithDefaultOptions;
    class Value;
}

class XFeat {
public:
    /**
     * 构造函数
     * @param xfeat_model_path XFeat模型文件路径
     * @param lighterglue_model_path LighterGlue模型文件路径
     * @param device 运行设备 ("cpu" 或 "cuda")
     */
    explicit XFeat(const std::string& xfeat_model_path = "onnx/xfeat_4096_3072x2048.onnx",
                   const std::string& device = "cpu");
    XFeat();
    
    /**
     * 析构函数
     */
    ~XFeat();

    void setModelPath(const std::string& xfeat_model_path);
    void setDevice(const std::string& device);
    
    /**
     * 从图像中提取特征点和描述符
     * @param image 输入图像 (BGR格式)
     * @param keypoints 输出的关键点坐标
     * @param descriptors 输出的描述符
     * @param max_keypoints 最大关键点数量 (默认4096)
     * @return 是否成功
     */
    bool detectAndCompute(const cv::Mat& image, 
                         std::vector<cv::Point2f>& keypoints,
                         cv::Mat& descriptors,
                         int max_keypoints = 4096);
    
    /**
     * 匹配两组描述符
     * @param descriptors1 第一组描述符
     * @param descriptors2 第二组描述符
     * @param matches 输出的匹配对索引
     * @param ratio_threshold 最近邻比率阈值 (默认0.8)
     * @return 匹配点数量
     */
    int match(const cv::Mat& descriptors1, 
              const cv::Mat& descriptors2,
              std::vector<cv::DMatch>& matches,
              float ratio_threshold = 0.8f);
    
    /**
     * 使用XFeat自带的双向最近邻匹配器
     * @param descriptors1 第一组描述符
     * @param descriptors2 第二组描述符
     * @param matches 输出的匹配对索引
     * @param min_cossim 最小余弦相似度阈值 (默认0.82)
     * @return 匹配点数量
     */
    
    /**
     * 完整的特征提取和匹配流程
     * @param image1 第一张图像
     * @param image2 第二张图像
     * @param keypoints1 第一张图像的关键点
     * @param keypoints2 第二张图像的关键点
     * @param matches 匹配结果
     * @return 是否成功
     */
    bool detectAndMatch(const cv::Mat& image1, 
                       const cv::Mat& image2,
                       std::vector<cv::Point2f>& keypoints1,
                       std::vector<cv::Point2f>& keypoints2,
                       std::vector<cv::DMatch>& matches);
    
    /**
     * 使用LighterGlue进行高质量匹配
     * @param image1 第一张图像
     * @param image2 第二张图像
     * @param keypoints1 第一张图像的关键点
     * @param keypoints2 第二张图像的关键点
     * @param matches 匹配结果
     * @param min_score 最小匹配分数阈值
     * @return 匹配点数量
     */
    int matchWithLighterGlue(const cv::Mat& image1,
                            const cv::Mat& image2,
                            std::vector<cv::Point2f>& keypoints1,
                            std::vector<cv::Point2f>& keypoints2,
                            std::vector<cv::DMatch>& matches,
                            float min_score = 0.3f);
    
    /**
     * 智能匹配：先提取特征，再使用LighterGlue匹配
     * @param image1 第一张图像
     * @param image2 第二张图像
     * @param keypoints1 第一张图像的关键点
     * @param keypoints2 第二张图像的关键点
     * @param matches 匹配结果
     * @param min_score 最小匹配分数阈值
     * @return 匹配结果统计
     */
    struct MatchResult {
        int total_matches;
        int good_matches;
        double total_time;
        std::vector<cv::Point2f> keypoints1;
        std::vector<cv::Point2f> keypoints2;
        std::vector<cv::DMatch> matches;
    };
    
    MatchResult smartMatch(const cv::Mat& image1,
                          const cv::Mat& image2,
                          float min_score = 0.3f);

private:
    // ONNX Runtime 相关成员
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::unique_ptr<Ort::Session> lighterglue_session_;
    std::unique_ptr<Ort::SessionOptions> ort_session_options_;
    std::unique_ptr<Ort::MemoryInfo> ort_memory_info_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> ort_allocator_;
    
    // 模型信息
    std::string xfeat_model_path_;
    std::string lighterglue_model_path_;
    std::string device_;
    std::vector<std::string> xfeat_input_names_;
    std::vector<std::string> xfeat_output_names_;
    std::vector<std::string> lighterglue_input_names_;
    std::vector<std::string> lighterglue_output_names_;
    std::vector<std::vector<int64_t>> xfeat_input_shapes_;
    std::vector<std::vector<int64_t>> xfeat_output_shapes_;
    std::vector<std::vector<int64_t>> lighterglue_input_shapes_;
    std::vector<std::vector<int64_t>> lighterglue_output_shapes_;
    
    // 图像预处理参数
    static constexpr int INPUT_WIDTH = 3072;
    static constexpr int INPUT_HEIGHT = 2048;
    static constexpr int MAX_KEYPOINTS = 4096;
    static constexpr int DESCRIPTOR_DIM = 64;  // XFeat描述符维度
    
    /**
     * 初始化ONNX Runtime环境
     * @return 是否成功
     */
    bool initializeOrt();
    
    /**
     * 加载ONNX模型
     * @return 是否成功
     */
    bool loadModel();
    
    /**
     * 预处理图像
     * @param image 输入图像
     * @param processed 预处理后的图像张量
     * @return 是否成功
     */
    bool preprocessImage(const cv::Mat& image, std::vector<float>& processed);
    
    /**
     * 后处理特征提取结果
     * @param keypoint_output 关键点输出
     * @param descriptor_output 描述符输出
     * @param keypoints 输出的关键点
     * @param descriptors 输出的描述符
     * @param max_keypoints 最大关键点数量
     * @return 是否成功
     */
    bool postprocessFeatures(const std::vector<float>& keypoint_output,
                           const std::vector<float>& descriptor_output,
                           std::vector<cv::Point2f>& keypoints,
                           cv::Mat& descriptors,
                           int max_keypoints);
    
    /**
     * 使用FLANN进行描述符匹配
     * @param descriptors1 第一组描述符
     * @param descriptors2 第二组描述符
     * @param matches 输出的匹配对
     * @param ratio_threshold 最近邻比率阈值
     * @return 匹配点数量
     */
    int matchWithFLANN(const cv::Mat& descriptors1, 
                      const cv::Mat& descriptors2,
                      std::vector<cv::DMatch>& matches,
                      float ratio_threshold);
    
    /**
     * 加载LighterGlue模型
     * @return 是否成功
     */
    bool loadLighterGlueModel();
    
    /**
     * 使用LighterGlue进行匹配推理
     * @param keypoints1 第一张图像的关键点
     * @param keypoints2 第二张图像的关键点
     * @param descriptors1 第一张图像的描述符
     * @param descriptors2 第二张图像的描述符
     * @param matches 输出的匹配结果
     * @param min_score 最小匹配分数阈值
     * @return 匹配点数量
     */
    int lighterGlueInference(const std::vector<cv::Point2f>& keypoints1,
                            const std::vector<cv::Point2f>& keypoints2,
                            const cv::Mat& descriptors1,
                            const cv::Mat& descriptors2,
                            std::vector<cv::DMatch>& matches,
                            float min_score);
};

/**
 * 独立的XFeat匹配函数 (不依赖类)
 * @param descriptors1 第一组描述符
 * @param descriptors2 第二组描述符
 * @param matches 输出的匹配对索引
 * @param min_cossim 最小余弦相似度阈值 (默认0.82)
 * @return 匹配点数量
 */
int matchWithXFeat(const cv::Mat& descriptors1, 
                   const cv::Mat& descriptors2,
                   std::vector<cv::DMatch>& matches,
                   float min_cossim = 0.82f);

#endif // XFEAT_H