#include "feat_extractor.h"
#include <iostream>
#include <chrono>

FeatExtractor::FeatExtractor(const std::string& model_path, const std::string& name, bool force_cpu)
    : OrtRun(model_path, name, force_cpu), output_keypoints_(nullptr), output_descriptors_(nullptr) {
}

bool FeatExtractor::run(const cv::Mat& img, 
                       std::vector<cv::Point2f>& keypoints, 
                       cv::Mat& descriptors) {
    if (img.empty()) {
        std::cerr << "Input image is empty" << std::endl;
        return false;
    }
    
    // 保存当前图像和输出指针
    current_image_ = img.clone();
    output_keypoints_ = &keypoints;
    output_descriptors_ = &descriptors;
    
    // 预处理图像
    std::vector<float> input_data = preprocessImage(img);
    if (input_data.empty()) {
        std::cerr << "Failed to preprocess image" << std::endl;
        return false;
    }
    auto input_shape = getInputShapes()[0];
    // 创建输入张量
    // std::vector<int64_t> input_shape = {1, 3, 2048, 3072};  // [B, C, H, W]
    auto input_tensor = createTensor(input_data, input_shape);
    
    // 运行推理
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    std::vector<Ort::Value> output_tensors;
    
    if (!OrtRun::run(std::move(input_tensors), output_tensors)) {
        std::cerr << "Failed to run inference" << std::endl;
        return false;
    }
    
    // 后处理结果
    if (!postprocessFeatures(output_tensors, keypoints, descriptors)) {
        std::cerr << "Failed to postprocess features" << std::endl;
        return false;
    }
    
    return true;
}

cv::Size FeatExtractor::getInputWH() {
    auto input_shape = getInputShapes()[0];
    return cv::Size(input_shape[3], input_shape[2]);
}

bool FeatExtractor::preprocess(std::vector<Ort::Value>& /*inputs*/) {
    // 对于特征提取，预处理已经在run函数中完成
    return true;
}

bool FeatExtractor::postprocess(std::vector<Ort::Value>& /*outputs*/) {
    // 对于特征提取，后处理已经在run函数中完成
    return true;
}

std::vector<float> FeatExtractor::preprocessImage(const cv::Mat& img) {
    if (img.empty()) {
        return {};
    }
    
    // 调整图像大小到模型期望的尺寸 [3072, 2048] (宽x高)
    cv::Mat resized_img;
    auto input_shape = getInputShapes()[0];
    cv::resize(img, resized_img, cv::Size(input_shape[3], input_shape[2]));
    
    // 转换颜色空间 BGR -> RGB
    cv::Mat rgb_img;
    cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);
    
    // 归一化到 [0, 1]
    rgb_img.convertTo(rgb_img, CV_32F, 1.0 / 255.0);
    
    // 转换为CHW格式 (Channel, Height, Width)
    std::vector<cv::Mat> channels;
    cv::split(rgb_img, channels);
    
    std::vector<float> input_data;
    input_data.reserve(3 * input_shape[2] * input_shape[3]);
    
    for (const auto& channel : channels) {
        input_data.insert(input_data.end(), 
                         channel.ptr<float>(), 
                         channel.ptr<float>() + channel.total());
    }
    
    return input_data;
}

bool FeatExtractor::postprocessFeatures(const std::vector<Ort::Value>& data, 
                                       std::vector<cv::Point2f>& keypoints, 
                                       cv::Mat& descriptors) {
    if (data.size() < 2) {
        std::cerr << "Invalid output data size" << std::endl;
        return false;
    }
    
    try {
        // 获取关键点数据和类型信息
        auto kpt_type_info = data[0].GetTensorTypeAndShapeInfo();
        auto kpt_shape = kpt_type_info.GetShape();
        auto kpt_type = kpt_type_info.GetElementType();
        
        std::cout << "Keypoints data type: " << kpt_type << std::endl;
        std::cout << "Keypoints shape: [";
        for (size_t i = 0; i < kpt_shape.size(); i++) {
            std::cout << kpt_shape[i];
            if (i < kpt_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // 根据数据类型获取数据指针
        void* kpt_data_ptr = nullptr;
        if (kpt_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            kpt_data_ptr = const_cast<Ort::Value&>(data[0]).GetTensorMutableData<float>();
            std::cout << "Using float data type" << std::endl;
        } else if (kpt_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
            kpt_data_ptr = const_cast<Ort::Value&>(data[0]).GetTensorMutableData<double>();
            std::cout << "Using double data type" << std::endl;
        } else if (kpt_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            kpt_data_ptr = const_cast<Ort::Value&>(data[0]).GetTensorMutableData<int32_t>();
            std::cout << "Using int32 data type" << std::endl;
        } else if (kpt_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            kpt_data_ptr = const_cast<Ort::Value&>(data[0]).GetTensorMutableData<int64_t>();
            std::cout << "Using int64 data type" << std::endl;
        } else {
            std::cerr << "Unsupported data type: " << kpt_type << std::endl;
            return false;
        }
        
        // 解析关键点 - 形状是 [num_keypoints, 2]
        keypoints.clear();
        int num_keypoints = kpt_shape[0];
        std::cout << "Number of keypoints: " << num_keypoints << std::endl;
        
        if (num_keypoints <= 0) {
            std::cerr << "Invalid number of keypoints: " << num_keypoints << std::endl;
            return false;
        }
        
        // 根据实际数据类型处理关键点数据
        std::vector<float> kpt_float_data;
        if (kpt_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            int64_t* kpt_int64_data = static_cast<int64_t*>(kpt_data_ptr);
            kpt_float_data.resize(num_keypoints * 2);
            for (int i = 0; i < num_keypoints * 2; i++) {
                kpt_float_data[i] = static_cast<float>(kpt_int64_data[i]);
            }
        } else {
            kpt_float_data.resize(num_keypoints * 2);
            float* kpt_float_ptr = static_cast<float*>(kpt_data_ptr);
            for (int i = 0; i < num_keypoints * 2; i++) {
                kpt_float_data[i] = kpt_float_ptr[i];
            }
        }
        
        float* kpt_data = kpt_float_data.data();
        
        // 获取描述符数据和类型信息
        auto desc_type_info = data[1].GetTensorTypeAndShapeInfo();
        auto desc_shape = desc_type_info.GetShape();
        auto desc_type = desc_type_info.GetElementType();
        
        std::cout << "Descriptors data type: " << desc_type << std::endl;
        
        // 根据数据类型获取描述符数据指针
        void* desc_data_ptr = nullptr;
        if (desc_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            desc_data_ptr = const_cast<Ort::Value&>(data[1]).GetTensorMutableData<float>();
        } else if (desc_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
            desc_data_ptr = const_cast<Ort::Value&>(data[1]).GetTensorMutableData<double>();
        } else {
            std::cerr << "Unsupported descriptor data type: " << desc_type << std::endl;
            return false;
        }
        
        float* desc_data = static_cast<float*>(desc_data_ptr);
        
        // 调试信息：打印张量形状
        std::cout << "Keypoints tensor shape: [";
        for (size_t i = 0; i < kpt_shape.size(); i++) {
            std::cout << kpt_shape[i];
            if (i < kpt_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "Descriptors tensor shape: [";
        for (size_t i = 0; i < desc_shape.size(); i++) {
            std::cout << desc_shape[i];
            if (i < desc_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        for (int i = 0; i < num_keypoints; i++) {
            float x = kpt_data[i * 2];
            float y = kpt_data[i * 2 + 1];
            
            // // 将归一化坐标转换回图像坐标
            // x = x * current_image_.cols;
            // y = y * current_image_.rows;
            // if( i< 10)
            //     std::cout << "x: " << x << ", y: " << y << std::endl;
            
            keypoints.emplace_back(x, y);
        }
        
        // 解析描述符 - 形状是 [num_keypoints, desc_dim]
        int desc_dim = desc_shape[1];
        std::cout << "Descriptor dimension: " << desc_dim << std::endl;
        
        if (desc_dim <= 0) {
            std::cerr << "Invalid descriptor dimension: " << desc_dim << std::endl;
            return false;
        }
        
        descriptors = cv::Mat(num_keypoints, desc_dim, CV_32F);
        for (int i = 0; i < num_keypoints; i++) {
            for (int j = 0; j < desc_dim; j++) {
                descriptors.at<float>(i, j) = desc_data[i * desc_dim + j];
            }
        }
        
        std::cout << "Extracted " << keypoints.size() << " keypoints with " 
                  << desc_dim << " dimensional descriptors" << std::endl;
        
        // 调试信息：打印前几个描述符值
        // std::cout << "First descriptor values: ";
        // for (int i = 0; i < std::min(5, desc_dim); i++) {
        //     std::cout << descriptors.at<float>(0, i) << " ";
        // }
        // std::cout << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error in postprocessing: " << e.what() << std::endl;
        return false;
    }
}