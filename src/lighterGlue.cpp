#include "lighterGlue.h"
#include <stdexcept>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

LighterGlue::LighterGlue(const std::string& model_path)
    : model_path_(model_path), output_matches_(nullptr), output_scores_(nullptr) {
        if (!initializeOrt()) {
        throw std::runtime_error("Failed to initialize ONNX Runtime");
    }
    if (!loadModel()) {
        throw std::runtime_error("Failed to load LighterGlue model");
    }
}

bool LighterGlue::initializeOrt() {
    try {
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "LighterGlue");
        ort_session_options_ = std::make_unique<Ort::SessionOptions>();
        ort_memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        ort_allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();
        
        // 设置线程数
        ort_session_options_->SetIntraOpNumThreads(4);
        ort_session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // 设置确定性推理
        ort_session_options_->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        ort_session_options_->SetInterOpNumThreads(1);
        
        // 设置随机种子以获得可重现的结果
        ort_session_options_->AddConfigEntry("session.use_env_allocators", "1");
        ort_session_options_->AddConfigEntry("session.disable_prepacking", "1");
        
        // 设置随机种子
        std::srand(30);  // 固定随机种子
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing ONNX Runtime: " << e.what() << std::endl;
        return false;
    }
}

bool LighterGlue::loadModel() {
    try {
        session_ = std::make_unique<Ort::Session>(*ort_env_, model_path_.c_str(), *ort_session_options_);
        
        // 获取模型输入输出信息
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();
        
        input_names_.resize(num_input_nodes);
        output_names_.resize(num_output_nodes);
        input_shapes_.resize(num_input_nodes);
        output_shapes_.resize(num_output_nodes);
        
        // 获取输入信息
        for (size_t i = 0; i < num_input_nodes; i++) {
            input_names_[i] = session_->GetInputNameAllocated(i, *ort_allocator_).get();
            auto input_type_info = session_->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            input_shapes_[i] = input_tensor_info.GetShape();
        }
        
        // 获取输出信息
        for (size_t i = 0; i < num_output_nodes; i++) {
            output_names_[i] = session_->GetOutputNameAllocated(i, *ort_allocator_).get();
            auto output_type_info = session_->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            output_shapes_[i] = output_tensor_info.GetShape();
        }
        
        // 准备C字符串数组
        input_names_cstr_.resize(input_names_.size());
        output_names_cstr_.resize(output_names_.size());
        for (size_t i = 0; i < input_names_.size(); i++) {
            input_names_cstr_[i] = input_names_[i].c_str();
        }
        for (size_t i = 0; i < output_names_.size(); i++) {
            output_names_cstr_[i] = output_names_[i].c_str();
        }
        
        std::cout << "Model loaded successfully: " << model_path_ << std::endl;
        std::cout << "Input nodes: " << num_input_nodes << ", Output nodes: " << num_output_nodes << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

LighterGlue::~LighterGlue() {
    session_.reset();
    ort_env_.reset();
    ort_session_options_.reset();
    ort_memory_info_.reset();
    ort_allocator_.reset();
}

std::vector<Ort::Value> LighterGlue::preprocessMatchingData(
    const std::vector<cv::Point2f>& keypoints1,
    const std::vector<cv::Point2f>& keypoints2,
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2) {
    try {
        // 归一化关键点坐标
        auto norm_kpts1 = normalizeKeypoints(keypoints1, input_shapes_[0][2], input_shapes_[0][3]);
        auto norm_kpts2 = normalizeKeypoints(keypoints2, input_shapes_[1][2], input_shapes_[1][3]);
        
        // 调试信息：打印前几个归一化关键点
        std::cout << "First 3 normalized keypoints1: ";
        for (int i = 0; i < std::min(3, (int)norm_kpts1.size()); i++) {
            std::cout << "(" << norm_kpts1[i].x << ", " << norm_kpts1[i].y << ") ";
        }
        std::cout << std::endl;
        
        std::cout << "First 3 normalized keypoints2: ";
        for (int i = 0; i < std::min(3, (int)norm_kpts2.size()); i++) {
            std::cout << "(" << norm_kpts2[i].x << ", " << norm_kpts2[i].y << ") ";
        }
        std::cout << std::endl;
        
        // 准备关键点数据
        std::vector<float> kpts1_data, kpts2_data;
        for (const auto& kpt : norm_kpts1) {
            kpts1_data.push_back(kpt.x);
            kpts1_data.push_back(kpt.y);
        }
        for (const auto& kpt : norm_kpts2) {
            kpts2_data.push_back(kpt.x);
            kpts2_data.push_back(kpt.y);
        }

        int descriptor_src_size = descriptors1.rows * descriptors1.cols;
        int descriptor_dst_size = descriptors2.rows * descriptors2.cols;
        
        float* descriptor_src_data;
        cv::Mat temp1;
        if ( descriptors1.isContinuous() )
        {
            descriptor_src_data = const_cast<float*>( descriptors1.ptr<float>( 0 ) );
        }
        else
        {
            temp1 = descriptors1.clone();
            descriptor_src_data = const_cast<float*>( temp1.ptr<float>( 0 ) );
        }

        float* descriptor_dst_data;
        cv::Mat temp2;
        if ( descriptors2.isContinuous() )
        {
            descriptor_dst_data = const_cast<float*>( descriptors2.ptr<float>( 0 ) );
        }
        else
        {
            temp2 = descriptors2.clone();
            descriptor_dst_data = const_cast<float*>( temp2.ptr<float>( 0 ) );
        }
        
        // 调试信息：检查数据范围
        auto minmax_kpts1 = std::minmax_element(kpts1_data.begin(), kpts1_data.end());
        auto minmax_kpts2 = std::minmax_element(kpts2_data.begin(), kpts2_data.end());
        std::cout << "数据范围: (" << *minmax_kpts1.first << ", " << *minmax_kpts1.second << ") (" << *minmax_kpts2.first << ", " << *minmax_kpts2.second << ")" << std::endl;

        std::vector<int64_t> kpts1_shape = {1, static_cast<int64_t>(keypoints1.size()), 2};
        std::vector<int64_t> kpts2_shape = {1, static_cast<int64_t>(keypoints2.size()), 2};
        std::vector<int64_t> desc1_shape = {1, static_cast<int64_t>(descriptors1.rows), static_cast<int64_t>(descriptors1.cols)};
        std::vector<int64_t> desc2_shape = {1, static_cast<int64_t>(descriptors2.rows), static_cast<int64_t>(descriptors2.cols)};
        
        std::cout << "kpts1_shape: " << kpts1_shape[0] << " " << kpts1_shape[1] << " " << kpts1_shape[2] << std::endl;
        std::cout << "kpts2_shape: " << kpts2_shape[0] << " " << kpts2_shape[1] << " " << kpts2_shape[2] << std::endl;
        std::cout << "desc1_shape: " << desc1_shape[0] << " " << desc1_shape[1] << " " << desc1_shape[2] << std::endl;
        std::cout << "desc2_shape: " << desc2_shape[0] << " " << desc2_shape[1] << " " << desc2_shape[2] << std::endl;

        auto kpts1_tensor = createTensor(kpts1_data.data(), kpts1_data.size(), kpts1_shape);
        auto kpts2_tensor = createTensor(kpts2_data.data(), kpts2_data.size(), kpts2_shape);
        auto desc1_tensor = createTensor(descriptor_src_data, descriptor_src_size, desc1_shape);
        auto desc2_tensor = createTensor(descriptor_dst_data, descriptor_dst_size, desc2_shape);

        std::vector<Ort::Value> input_tensors;
        // LighterGlue模型输入顺序: kpts0, kpts1, desc0, desc1
        input_tensors.push_back(std::move(kpts1_tensor));  // kpts0
        input_tensors.push_back(std::move(kpts2_tensor));  // kpts1
        input_tensors.push_back(std::move(desc1_tensor));  // desc0
        input_tensors.push_back(std::move(desc2_tensor));  // desc1
        
        return input_tensors;
    } catch (const std::exception& e) {
        std::cerr << "Error in preprocessing matching data: " << e.what() << std::endl;
        return {};
    }
}



bool LighterGlue::postprocessMatchingResults(const std::vector<Ort::Value>& data,
                                        std::vector<cv::DMatch>& matches,
                                        std::vector<float>& scores) {
    if (data.size() < 2) {
        std::cerr << "Invalid output data size" << std::endl;
        return false;
    }
    
    try {
        // 获取匹配结果 - 匹配索引应该是整数类型
        int64_t* matches_data = const_cast<Ort::Value&>(data[0]).GetTensorMutableData<int64_t>();
        float* scores_data = const_cast<Ort::Value&>(data[1]).GetTensorMutableData<float>();
        
        auto matches_shape = data[0].GetTensorTypeAndShapeInfo().GetShape();
        auto scores_shape = data[1].GetTensorTypeAndShapeInfo().GetShape();
        
        // 调试信息：打印输出张量形状
        std::cout << "Matches output shape: [";
        for (size_t i = 0; i < matches_shape.size(); i++) {
            std::cout << matches_shape[i];
            if (i < matches_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "Scores output shape: [";
        for (size_t i = 0; i < scores_shape.size(); i++) {
            std::cout << scores_shape[i];
            if (i < scores_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // 检查输出张量是否有效
        if (matches_shape.empty() || scores_shape.empty()) {
            std::cerr << "Invalid output tensor shapes" << std::endl;
            return false;
        }
        
        // 检查输出数据是否有效
        if (matches_data == nullptr || scores_data == nullptr) {
            std::cerr << "Invalid output tensor data" << std::endl;
            return false;
        }
        
        // 处理不同的输出形状格式
        int num_matches;
        if (matches_shape.size() == 2) {
            // 形状是 [num_matches, 2]
            num_matches = matches_shape[0];
        } else if (matches_shape.size() == 3) {
            // 形状是 [1, num_matches, 2]
            num_matches = matches_shape[1];
        } else {
            std::cerr << "Unexpected matches output shape: [";
            for (size_t i = 0; i < matches_shape.size(); i++) {
                std::cerr << matches_shape[i];
                if (i < matches_shape.size() - 1) std::cerr << ", ";
            }
            std::cerr << "]" << std::endl;
            return false;
        }
        std::cout << "Number of matches from model: " << num_matches << std::endl;
        
        matches.clear();
        scores.clear();
        
        // 统计分数分布
        std::vector<float> all_scores(scores_data, scores_data + num_matches);
        std::sort(all_scores.begin(), all_scores.end());
        std::cout << "Score statistics: min=" << all_scores[0] 
                  << ", max=" << all_scores[num_matches-1] 
                  << ", median=" << all_scores[num_matches/2] << std::endl;
        
        // 解析匹配结果
        float threshold = 0.7f;  // 与Python代码draw_matches相同的阈值（不过滤）
        for (int i = 0; i < num_matches; i++) {
            float score = scores_data[i];
            
            // 调试信息：打印前几个分数
            if (i < 10) {
                std::cout << "Score " << i << ": " << score << " (threshold: " << threshold << ")" << std::endl;
            }
            
            // 应用阈值过滤 - 分数大于阈值表示匹配质量好
            if (score > threshold) {
                // 根据输出形状调整数据访问偏移量
                int matches_offset = (matches_shape.size() == 3) ? 1 : 0;  // 如果有batch维度，跳过第一个元素
                int query_idx = static_cast<int>(matches_data[matches_offset + i * 2]);
                int train_idx = static_cast<int>(matches_data[matches_offset + i * 2 + 1]);
                
                // 调试信息：打印前几个匹配的索引和原始数据
                if (i < 10) {
                    std::cout << "Match " << i << ": raw_data=[" << matches_data[matches_offset + i * 2] 
                              << ", " << matches_data[matches_offset + i * 2 + 1] 
                              << "], query_idx=" << query_idx << ", train_idx=" << train_idx << std::endl;
                }
                
                // 检查索引有效性
                if (query_idx >= 0 && query_idx < static_cast<int>(current_keypoints1_.size()) &&
                    train_idx >= 0 && train_idx < static_cast<int>(current_keypoints2_.size())) {
                    
                    cv::DMatch match;
                    match.queryIdx = query_idx;
                    match.trainIdx = train_idx;
                    match.distance = 1.0f - score;  // 距离 = 1.0 - 分数
                    
                    matches.push_back(match);
                    scores.push_back(score);
                }
            }
        }
        
        std::cout << "Found " << matches.size() << " valid matches" << std::endl;
 
    } catch (const std::exception& e) {
        std::cerr << "Error in postprocessing matching results: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool LighterGlue::run(const std::vector<cv::Point2f>& keypoints1,
                      const std::vector<cv::Point2f>& keypoints2,
                      const cv::Mat& descriptors1,
                      const cv::Mat& descriptors2,
                      std::vector<cv::DMatch>& matches,
                      std::vector<float>& scores) {
    if (keypoints1.empty() || keypoints2.empty() || 
        descriptors1.empty() || descriptors2.empty()) {
        std::cerr << "Invalid input data for matching" << std::endl;
        return false;
    }
    
    // 保存当前匹配数据和输出指针
    current_keypoints1_ = keypoints1;
    current_keypoints2_ = keypoints2;
    current_descriptors1_ = descriptors1.clone();
    current_descriptors2_ = descriptors2.clone();
    output_matches_ = &matches;
    output_scores_ = &scores;
    
    // 预处理匹配数据
    std::vector<Ort::Value> input_tensors = preprocessMatchingData(
        keypoints1, keypoints2, descriptors1, descriptors2);
    
    // 运行推理
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                      input_names_cstr_.data(), input_tensors.data(), input_tensors.size(),
                                      output_names_cstr_.data(), output_names_cstr_.size());
    } catch (const std::exception& e) {
        std::cerr << "Error running LighterGlue inference: " << e.what() << std::endl;
        return false;
    }
    
    // 后处理匹配结果
    if (!postprocessMatchingResults(output_tensors, matches, scores)) {
        std::cerr << "Failed to postprocess matching results" << std::endl;
        return false;
    }
    
    // 筛选出匹配点的坐标
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : matches) {
        pts1.push_back(keypoints1[match.queryIdx]);
        pts2.push_back(keypoints2[match.trainIdx]);
    }
    
    // 绘制匹配结果 - 需要转换为KeyPoint格式
    cv::Mat img_matches;
    std::vector<cv::KeyPoint> kp1, kp2;
    for (const auto& pt : pts1) {
        kp1.emplace_back(pt.x, pt.y, 1.0f);
    }
    for (const auto& pt : pts2) {
        kp2.emplace_back(pt.x, pt.y, 1.0f);
    }
    
    // 注意：这里需要实际的图像，而不是描述符
    // cv::drawMatches(img1, kp1, img2, kp2, matches, img_matches);
    // cv::imwrite("matching_result.jpg", img_matches);
    
    return true;
}

bool LighterGlue::preprocess(std::vector<Ort::Value>& inputs) {
    // 对于匹配，预处理已经在run函数中完成
    return true;
}

bool LighterGlue::postprocess(std::vector<Ort::Value>& outputs) {
    return true;
}

std::vector<cv::Point2f> LighterGlue::normalizeKeypoints(const std::vector<cv::Point2f>& keypoints,
                                                        int height, int width) {
    std::vector<cv::Point2f> normalized_keypoints;
    normalized_keypoints.reserve(keypoints.size());
    
    // 使用与image_matching_onnx.py相同的归一化方式
    // size = [width, height], shift = size / 2, scale = size.max() / 2
    // kpts = (kpts - shift) / scale
    float shift_x = width / 2.0f;
    float shift_y = height / 2.0f;
    float scale = std::max(width, height) / 2.0f;
    
    for (const auto& kpt : keypoints) {
        float x = (kpt.x - shift_x) / scale;
        float y = (kpt.y - shift_y) / scale;
        normalized_keypoints.emplace_back(x, y);
    }
    
    return normalized_keypoints;
}

Ort::Value LighterGlue::createTensor(const float* data, size_t data_size, const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<float>(*ort_memory_info_, const_cast<float*>(data), data_size, 
                                          shape.data(), shape.size());
}

void LighterGlue::match(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches) {
    // 创建虚拟关键点（因为LighterGlue需要关键点）
    std::vector<cv::Point2f> dummy_keypoints1(descriptors1.rows);
    std::vector<cv::Point2f> dummy_keypoints2(descriptors2.rows);
    
    for (int i = 0; i < descriptors1.rows; i++) {
        dummy_keypoints1[i] = cv::Point2f(i % 100, i / 100);  // 简单的虚拟坐标
    }
    for (int i = 0; i < descriptors2.rows; i++) {
        dummy_keypoints2[i] = cv::Point2f(i % 100, i / 100);  // 简单的虚拟坐标
    }
    
    std::vector<float> scores;
    run(dummy_keypoints1, dummy_keypoints2, descriptors1, descriptors2, matches, scores);
}


