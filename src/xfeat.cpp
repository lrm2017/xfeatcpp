#include "xfeat.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>

XFeat::XFeat(const std::string& xfeat_model_path, const std::string& device)
    : xfeat_model_path_(xfeat_model_path), lighterglue_model_path_(""), device_(device) {
    if (!initializeOrt()) {
        throw std::runtime_error("Failed to initialize ONNX Runtime");
    }
    if (!loadModel()) {
        throw std::runtime_error("Failed to load XFeat model");
    }
    if (!loadLighterGlueModel()) {
        throw std::runtime_error("Failed to load LighterGlue model");
    }
}

XFeat::~XFeat() = default;

bool XFeat::initializeOrt() {
    try {
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "XFeat");
        ort_session_options_ = std::make_unique<Ort::SessionOptions>();
        ort_memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        ort_allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();
        
        // 设置线程数
        ort_session_options_->SetIntraOpNumThreads(4);
        ort_session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing ONNX Runtime: " << e.what() << std::endl;
        return false;
    }
}

bool XFeat::loadModel() {
    try {
        ort_session_ = std::make_unique<Ort::Session>(*ort_env_, xfeat_model_path_.c_str(), *ort_session_options_);
        
        // 获取XFeat模型输入输出信息
        size_t num_input_nodes = ort_session_->GetInputCount();
        size_t num_output_nodes = ort_session_->GetOutputCount();
        
        xfeat_input_names_.resize(num_input_nodes);
        xfeat_output_names_.resize(num_output_nodes);
        xfeat_input_shapes_.resize(num_input_nodes);
        xfeat_output_shapes_.resize(num_output_nodes);
        
        // 获取输入信息
        for (size_t i = 0; i < num_input_nodes; i++) {
            xfeat_input_names_[i] = ort_session_->GetInputNameAllocated(i, *ort_allocator_).get();
            auto input_type_info = ort_session_->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            xfeat_input_shapes_[i] = input_tensor_info.GetShape();
        }
        
        // 获取输出信息
        for (size_t i = 0; i < num_output_nodes; i++) {
            xfeat_output_names_[i] = ort_session_->GetOutputNameAllocated(i, *ort_allocator_).get();
            auto output_type_info = ort_session_->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            xfeat_output_shapes_[i] = output_tensor_info.GetShape();
        }
        
        std::cout << "XFeat model loaded successfully. Input nodes: " << num_input_nodes 
                  << ", Output nodes: " << num_output_nodes << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading XFeat model: " << e.what() << std::endl;
        return false;
    }
}

bool XFeat::preprocessImage(const cv::Mat& image, std::vector<float>& processed) {
    try {
        // 调整图像大小到模型输入尺寸
        // cv::Mat resized;
        // std::cout << "image.size() = " << image.size() << std::endl;
        // cv::resize(image, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
        // std::cout << "resized.size() = " << resized.size() << std::endl;
        
        // 转换BGR到RGB
        cv::Mat rgb;
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
        
        // 归一化到[0, 1]
        rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
        
        // 转换为CHW格式 (Channel, Height, Width)
        std::vector<cv::Mat> channels;
        cv::split(rgb, channels);
        
        processed.clear();
        processed.reserve(3 * INPUT_HEIGHT * INPUT_WIDTH);
        
        for (const auto& channel : channels) {
            processed.insert(processed.end(), channel.begin<float>(), channel.end<float>());
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error preprocessing image: " << e.what() << std::endl;
        return false;
    }
}

bool XFeat::postprocessFeatures(const std::vector<float>& keypoint_output,
                               const std::vector<float>& descriptor_output,
                               std::vector<cv::Point2f>& keypoints,
                               cv::Mat& descriptors,
                               int max_keypoints) {
    try {
        // 假设输出格式为 [batch, num_keypoints, 2] 对于关键点
        // 和 [batch, num_keypoints, descriptor_dim] 对于描述符
        
        int num_detected = keypoint_output.size() / 2;  // 每个关键点2个坐标
        int actual_keypoints = std::min(num_detected, max_keypoints);
        
        keypoints.clear();
        keypoints.reserve(actual_keypoints);
        
        // 提取关键点坐标
        for (int i = 0; i < actual_keypoints; i++) {
            float x = keypoint_output[i * 2];
            float y = keypoint_output[i * 2 + 1];
            
            // 检查坐标是否需要归一化（如果坐标在0-1范围内，则乘以图像尺寸）
            if (x <= 1.0f && y <= 1.0f) {
                x *= INPUT_WIDTH;
                y *= INPUT_HEIGHT;
            }
            
            // 确保坐标在图像范围内
            x = std::max(0.0f, std::min(x, static_cast<float>(INPUT_WIDTH - 1)));
            y = std::max(0.0f, std::min(y, static_cast<float>(INPUT_HEIGHT - 1)));
            
            keypoints.emplace_back(x, y);
        }
        
        // 提取描述符
        int descriptor_size = descriptor_output.size() / num_detected;
        descriptors = cv::Mat(actual_keypoints, descriptor_size, CV_32F);
        
        for (int i = 0; i < actual_keypoints; i++) {
            for (int j = 0; j < descriptor_size; j++) {
                descriptors.at<float>(i, j) = descriptor_output[i * descriptor_size + j];
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error postprocessing features: " << e.what() << std::endl;
        return false;
    }
}

bool XFeat::detectAndCompute(const cv::Mat& image, 
                            std::vector<cv::Point2f>& keypoints,
                            cv::Mat& descriptors,
                            int max_keypoints) {
    try {
        // 预处理图像
        std::vector<float> input_tensor;
        if (!preprocessImage(image, input_tensor)) {
            return false;
        }
        
        // 创建输入张量
        std::vector<int64_t> input_shape = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};
        auto input_tensor_ort = Ort::Value::CreateTensor<float>(
            *ort_memory_info_, input_tensor.data(), input_tensor.size(),
            input_shape.data(), input_shape.size());
        
        // 运行推理
        std::vector<const char*> input_names_cstr = {xfeat_input_names_[0].c_str()};
        std::vector<const char*> output_names_cstr = {xfeat_output_names_[0].c_str(), xfeat_output_names_[1].c_str()};
        
        auto output_tensors = ort_session_->Run(Ort::RunOptions{nullptr},
                                               input_names_cstr.data(), &input_tensor_ort, 1,
                                               output_names_cstr.data(), 2);
        
        // 提取输出数据
        auto& keypoint_tensor = output_tensors[0];
        auto& descriptor_tensor = output_tensors[1];
        
        float* keypoint_data = keypoint_tensor.GetTensorMutableData<float>();
        float* descriptor_data = descriptor_tensor.GetTensorMutableData<float>();
        
        auto keypoint_shape = keypoint_tensor.GetTensorTypeAndShapeInfo().GetShape();
        auto descriptor_shape = descriptor_tensor.GetTensorTypeAndShapeInfo().GetShape();
        
        // 转换为vector
        std::vector<float> keypoint_vector(keypoint_data, 
            keypoint_data + keypoint_tensor.GetTensorTypeAndShapeInfo().GetElementCount());
        std::vector<float> descriptor_vector(descriptor_data, 
            descriptor_data + descriptor_tensor.GetTensorTypeAndShapeInfo().GetElementCount());
        
        // 后处理
        return postprocessFeatures(keypoint_vector, descriptor_vector, 
                                 keypoints, descriptors, max_keypoints);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in detectAndCompute: " << e.what() << std::endl;
        return false;
    }
}

int matchWithXFeat(const cv::Mat& descriptors1, 
    const cv::Mat& descriptors2,
    std::vector<cv::DMatch>& matches,
    float min_cossim)
{
    // 检查输入有效性
    if (descriptors1.empty() || descriptors2.empty()) {
        return 0;
    }

    if (descriptors1.cols != descriptors2.cols) {
        return 0;
    }

    int num_desc1 = descriptors1.rows;
    int num_desc2 = descriptors2.rows;

    // 转换为float并归一化
    cv::Mat desc1_float, desc2_float;
    descriptors1.convertTo(desc1_float, CV_32F);
    descriptors2.convertTo(desc2_float, CV_32F);

    cv::Mat desc1_norm, desc2_norm;
    cv::normalize(desc1_float, desc1_norm, 1.0, 0.0, cv::NORM_L2);
    cv::normalize(desc2_float, desc2_norm, 1.0, 0.0, cv::NORM_L2);

    // 计算余弦相似度矩阵
    cv::Mat cossim = desc1_norm * desc2_norm.t();
    cv::Mat cossim_t = desc2_norm * desc1_norm.t();

    // 使用minMaxLoc找到每行的最大值和索引
    std::vector<int> match12(num_desc1);
    std::vector<int> match21(num_desc2);
    std::vector<float> max_sim1(num_desc1);
    std::vector<float> max_sim2(num_desc2);

    // 特征1到特征2的最佳匹配
    for (int i = 0; i < num_desc1; i++) {
        cv::Mat row = cossim.row(i);
        double max_val;
        cv::Point max_loc;
        cv::minMaxLoc(row, nullptr, &max_val, nullptr, &max_loc);
        match12[i] = max_loc.x;
        max_sim1[i] = static_cast<float>(max_val);
    }

    // 特征2到特征1的最佳匹配
    for (int j = 0; j < num_desc2; j++) {
        cv::Mat row = cossim_t.row(j);
        double max_val;
        cv::Point max_loc;
        cv::minMaxLoc(row, nullptr, &max_val, nullptr, &max_loc);
        match21[j] = max_loc.x;
        max_sim2[j] = static_cast<float>(max_val);
    }

    // 双向一致性检查和相似度过滤
    matches.clear();
    for (int i = 0; i < num_desc1; i++) {
        int j = match12[i];
        float sim = max_sim1[i];

        // 检查双向一致性
        if (match21[j] == i && sim > min_cossim) {
            cv::DMatch match;
            match.queryIdx = i;
            match.trainIdx = j;
            match.distance = 1.0f - sim;
            matches.push_back(match);
        }
    }

    return static_cast<int>(matches.size());
}

int XFeat::matchWithFLANN(const cv::Mat& descriptors1, 
                         const cv::Mat& descriptors2,
                         std::vector<cv::DMatch>& matches,
                         float ratio_threshold) {
    try {
        // 确保描述符是CV_32F类型
        cv::Mat desc1, desc2;
        descriptors1.convertTo(desc1, CV_32F);
        descriptors2.convertTo(desc2, CV_32F);
        
        // 尝试使用BFMatcher (暴力匹配器) 来匹配Python版本
        cv::BFMatcher matcher(cv::NORM_L2);
        
        // 进行KNN匹配 (k=2)
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(desc1, desc2, knn_matches, 2);
        
        // 应用Lowe's ratio test
        matches.clear();
        for (const auto& knn_match : knn_matches) {
            if (knn_match.size() >= 2) {
                const cv::DMatch& best_match = knn_match[0];
                const cv::DMatch& second_best_match = knn_match[1];
                
                if (best_match.distance < ratio_threshold * second_best_match.distance) {
                    matches.push_back(best_match);
                }
            }
        }
        
        return static_cast<int>(matches.size());
    } catch (const std::exception& e) {
        std::cerr << "Error in BF matching: " << e.what() << std::endl;
        return 0;
    }
}

int XFeat::match(const cv::Mat& descriptors1, 
                const cv::Mat& descriptors2,
                std::vector<cv::DMatch>& matches,
                float /* ratio_threshold */) {
    // 使用XFeat自带的匹配器（双向最近邻）
    return matchWithXFeat(descriptors1, descriptors2, matches, 0.82f);
}

bool XFeat::detectAndMatch(const cv::Mat& image1, 
                          const cv::Mat& image2,
                          std::vector<cv::Point2f>& keypoints1,
                          std::vector<cv::Point2f>& keypoints2,
                          std::vector<cv::DMatch>& matches) {
    try {
        // 提取第一张图像的特征
        cv::Mat descriptors1;
        if (!detectAndCompute(image1, keypoints1, descriptors1)) {
            std::cerr << "Failed to detect features in image1" << std::endl;
            return false;
        }
        
        // 提取第二张图像的特征
        cv::Mat descriptors2;
        if (!detectAndCompute(image2, keypoints2, descriptors2)) {
            std::cerr << "Failed to detect features in image2" << std::endl;
            return false;
        }
        
        // 匹配特征
        int num_matches = match(descriptors1, descriptors2, matches);
        
        std::cout << "Detected " << keypoints1.size() << " keypoints in image1" << std::endl;
        std::cout << "Detected " << keypoints2.size() << " keypoints in image2" << std::endl;
        std::cout << "Found " << num_matches << " matches" << std::endl;
        
        return num_matches > 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in detectAndMatch: " << e.what() << std::endl;
        return false;
    }
}

bool XFeat::loadLighterGlueModel() {
    try {
        lighterglue_session_ = std::make_unique<Ort::Session>(*ort_env_, lighterglue_model_path_.c_str(), *ort_session_options_);
        
        // 获取LighterGlue模型输入输出信息
        size_t num_input_nodes = lighterglue_session_->GetInputCount();
        size_t num_output_nodes = lighterglue_session_->GetOutputCount();
        
        lighterglue_input_names_.resize(num_input_nodes);
        lighterglue_output_names_.resize(num_output_nodes);
        lighterglue_input_shapes_.resize(num_input_nodes);
        lighterglue_output_shapes_.resize(num_output_nodes);
        
        // 获取输入信息
        for (size_t i = 0; i < num_input_nodes; i++) {
            lighterglue_input_names_[i] = lighterglue_session_->GetInputNameAllocated(i, *ort_allocator_).get();
            auto input_type_info = lighterglue_session_->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            lighterglue_input_shapes_[i] = input_tensor_info.GetShape();
        }
        
        // 获取输出信息
        for (size_t i = 0; i < num_output_nodes; i++) {
            lighterglue_output_names_[i] = lighterglue_session_->GetOutputNameAllocated(i, *ort_allocator_).get();
            auto output_type_info = lighterglue_session_->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            lighterglue_output_shapes_[i] = output_tensor_info.GetShape();
        }
        
        std::cout << "LighterGlue model loaded successfully. Input nodes: " << num_input_nodes 
                  << ", Output nodes: " << num_output_nodes << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading LighterGlue model: " << e.what() << std::endl;
        return false;
    }
}

int XFeat::lighterGlueInference(const std::vector<cv::Point2f>& keypoints1,
                                const std::vector<cv::Point2f>& keypoints2,
                                const cv::Mat& descriptors1,
                                const cv::Mat& descriptors2,
                                std::vector<cv::DMatch>& matches,
                                float min_score) {
    try {
        // 准备LighterGlue输入数据
        int num_kpts1 = static_cast<int>(keypoints1.size());
        int num_kpts2 = static_cast<int>(keypoints2.size());
        
        // 创建关键点张量 [N, 2] 格式
        std::vector<float> kpts1_data, kpts2_data;
        kpts1_data.reserve(num_kpts1 * 2);
        kpts2_data.reserve(num_kpts2 * 2);
        
        for (const auto& kp : keypoints1) {
            kpts1_data.push_back(kp.x);
            kpts1_data.push_back(kp.y);
        }
        
        for (const auto& kp : keypoints2) {
            kpts2_data.push_back(kp.x);
            kpts2_data.push_back(kp.y);
        }
        
        // 创建描述符张量 [N, D] 格式
        std::vector<float> desc1_data, desc2_data;
        desc1_data.reserve(num_kpts1 * descriptors1.cols);
        desc2_data.reserve(num_kpts2 * descriptors2.cols);
        
        for (int i = 0; i < num_kpts1; i++) {
            for (int j = 0; j < descriptors1.cols; j++) {
                desc1_data.push_back(descriptors1.at<float>(i, j));
            }
        }
        
        for (int i = 0; i < num_kpts2; i++) {
            for (int j = 0; j < descriptors2.cols; j++) {
                desc2_data.push_back(descriptors2.at<float>(i, j));
            }
        }
        
        // 创建输入张量
        std::vector<int64_t> kpts1_shape = {1, num_kpts1, 2};
        std::vector<int64_t> kpts2_shape = {1, num_kpts2, 2};
        std::vector<int64_t> desc1_shape = {1, num_kpts1, descriptors1.cols};
        std::vector<int64_t> desc2_shape = {1, num_kpts2, descriptors2.cols};
        
        auto kpts1_tensor = Ort::Value::CreateTensor<float>(
            *ort_memory_info_, kpts1_data.data(), kpts1_data.size(),
            kpts1_shape.data(), kpts1_shape.size());
        
        auto kpts2_tensor = Ort::Value::CreateTensor<float>(
            *ort_memory_info_, kpts2_data.data(), kpts2_data.size(),
            kpts2_shape.data(), kpts2_shape.size());
        
        auto desc1_tensor = Ort::Value::CreateTensor<float>(
            *ort_memory_info_, desc1_data.data(), desc1_data.size(),
            desc1_shape.data(), desc1_shape.size());
        
        auto desc2_tensor = Ort::Value::CreateTensor<float>(
            *ort_memory_info_, desc2_data.data(), desc2_data.size(),
            desc2_shape.data(), desc2_shape.size());
        
        // 准备输入
        std::vector<const char*> input_names = {
            lighterglue_input_names_[0].c_str(),
            lighterglue_input_names_[1].c_str(),
            lighterglue_input_names_[2].c_str(),
            lighterglue_input_names_[3].c_str()
        };
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(kpts1_tensor));
        input_tensors.push_back(std::move(kpts2_tensor));
        input_tensors.push_back(std::move(desc1_tensor));
        input_tensors.push_back(std::move(desc2_tensor));
        
        // 运行LighterGlue推理
        std::vector<const char*> output_names_cstr;
        for (const auto& name : lighterglue_output_names_) {
            output_names_cstr.push_back(name.c_str());
        }
        
        auto output_tensors = lighterglue_session_->Run(Ort::RunOptions{nullptr},
                                                       input_names.data(), input_tensors.data(), input_tensors.size(),
                                                       output_names_cstr.data(), output_names_cstr.size());
        
        // 解析输出结果
        auto& matches_tensor = output_tensors[0];
        auto& scores_tensor = output_tensors[1];
        
        float* matches_data = const_cast<float*>(matches_tensor.GetTensorData<float>());
        float* scores_data = const_cast<float*>(scores_tensor.GetTensorData<float>());
        
        auto matches_shape = matches_tensor.GetTensorTypeAndShapeInfo().GetShape();
        auto scores_shape = scores_tensor.GetTensorTypeAndShapeInfo().GetShape();
        
        // 根据实际形状解析匹配结果
        int num_matches = 0;
        if (matches_shape.size() == 2) {
            // 形状为 [N, 2] 的情况
            num_matches = static_cast<int>(matches_shape[0]);
        } else if (matches_shape.size() == 3) {
            // 形状为 [1, N, 2] 的情况
            num_matches = static_cast<int>(matches_shape[1]);
        }
        
        // 过滤匹配结果
        matches.clear();
        for (int i = 0; i < num_matches; i++) {
            int query_idx = static_cast<int>(matches_data[i * 2]);
            int train_idx = static_cast<int>(matches_data[i * 2 + 1]);
            float score = scores_data[i];
            
            if (score >= min_score) {
                cv::DMatch match;
                match.queryIdx = query_idx;
                match.trainIdx = train_idx;
                match.distance = 1.0f - score; // 转换为距离
                matches.push_back(match);
            }
        }
        
        return static_cast<int>(matches.size());
        
    } catch (const std::exception& e) {
        std::cerr << "Error in LighterGlue inference: " << e.what() << std::endl;
        return 0;
    }
}

int XFeat::matchWithLighterGlue(const cv::Mat& image1,
                                const cv::Mat& image2,
                                std::vector<cv::Point2f>& keypoints1,
                                std::vector<cv::Point2f>& keypoints2,
                                std::vector<cv::DMatch>& matches,
                                float min_score) {
    try {
        // 先提取特征
        cv::Mat descriptors1, descriptors2;
        if (!detectAndCompute(image1, keypoints1, descriptors1)) {
            std::cerr << "Failed to detect features in image1" << std::endl;
            return 0;
        }
        
        if (!detectAndCompute(image2, keypoints2, descriptors2)) {
            std::cerr << "Failed to detect features in image2" << std::endl;
            return 0;
        }
        
        // 使用LighterGlue进行匹配
        return lighterGlueInference(keypoints1, keypoints2, descriptors1, descriptors2, matches, min_score);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in matchWithLighterGlue: " << e.what() << std::endl;
        return 0;
    }
}

XFeat::MatchResult XFeat::smartMatch(const cv::Mat& image1,
                                    const cv::Mat& image2,
                                    float min_score) {
    MatchResult result;
    result.total_time = 0.0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 使用LighterGlue进行智能匹配
        int num_matches = matchWithLighterGlue(image1, image2, result.keypoints1, result.keypoints2, result.matches, min_score);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
        
        result.total_matches = num_matches;
        result.good_matches = num_matches; // LighterGlue已经过滤了低质量匹配
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in smartMatch: " << e.what() << std::endl;
        result.total_matches = 0;
        result.good_matches = 0;
        return result;
    }
}