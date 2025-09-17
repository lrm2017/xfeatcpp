#include "ort_run.h"
#include <iostream>
#include <chrono>

OrtRun::OrtRun(const std::string& model_path, const std::string& name, bool force_cpu)
    : model_path_(model_path), force_cpu_(force_cpu), last_execution_time_(0.0f), name_(name) {
    if (!initializeOrt()) {
        throw std::runtime_error("Failed to initialize ONNX Runtime");
    }
    if (!loadModel()) {
        throw std::runtime_error("Failed to load model: " + model_path);
    }
}

OrtRun::~OrtRun() {
    // 智能指针会自动清理资源
}

bool OrtRun::initializeOrt() {
    try {
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, name_.c_str());
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
        std::srand(42);  // 固定随机种子
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing ONNX Runtime: " << e.what() << std::endl;
        return false;
    }
}

bool OrtRun::loadModel() {
    try {
        ort_session_ = std::make_unique<Ort::Session>(*ort_env_, model_path_.c_str(), *ort_session_options_);
        
        // 获取模型输入输出信息
        size_t num_input_nodes = ort_session_->GetInputCount();
        size_t num_output_nodes = ort_session_->GetOutputCount();
        
        input_names_.resize(num_input_nodes);
        output_names_.resize(num_output_nodes);
        input_shapes_.resize(num_input_nodes);
        output_shapes_.resize(num_output_nodes);
        
        // 获取输入信息
        for (size_t i = 0; i < num_input_nodes; i++) {
            input_names_[i] = ort_session_->GetInputNameAllocated(i, *ort_allocator_).get();
            auto input_type_info = ort_session_->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            input_shapes_[i] = input_tensor_info.GetShape();
        }
        
        // 获取输出信息
        for (size_t i = 0; i < num_output_nodes; i++) {
            output_names_[i] = ort_session_->GetOutputNameAllocated(i, *ort_allocator_).get();
            auto output_type_info = ort_session_->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            output_shapes_[i] = output_tensor_info.GetShape();
        }
        
        std::cout << "Model loaded successfully: " << model_path_ << std::endl;
        std::cout << "Input nodes: " << num_input_nodes << ", Output nodes: " << num_output_nodes << std::endl;

        for (const auto& shape : input_shapes_) {
            std::cout << "Input shape: ";
            for (const auto& dim : shape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }
        for (const auto& shape : output_shapes_) {
            std::cout << "Output shape: ";
            for (const auto& dim : shape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }
        
        // 注意：上面显示的是模型静态形状定义，实际运行时输出形状可能不同
        std::cout << "注意：上面显示的是模型静态形状定义，实际运行时输出形状可能不同" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

bool OrtRun::run(std::vector<Ort::Value>&& input_tensors,
                 std::vector<Ort::Value>& output_tensors) {
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 预处理
        if (!preprocess(input_tensors)) {
            std::cerr << "Preprocessing failed" << std::endl;
            return false;
        }
        
        // 准备输入输出名称
        std::vector<const char*> input_names_cstr;
        std::vector<const char*> output_names_cstr;
        
        for (const auto& name : input_names_) {
            input_names_cstr.push_back(name.c_str());
        }
        for (const auto& name : output_names_) {
            output_names_cstr.push_back(name.c_str());
        }
        
        // 执行推理
        output_tensors = ort_session_->Run(Ort::RunOptions{nullptr},
                                         input_names_cstr.data(),
                                         input_tensors.data(),
                                         input_tensors.size(),
                                         output_names_cstr.data(),
                                         output_names_cstr.size());
        
        // 后处理
        if (!postprocess(output_tensors)) {
            std::cerr << "Postprocessing failed" << std::endl;
            return false;
        }
        
        // 计算执行时间
        auto end_time = std::chrono::high_resolution_clock::now();
        last_execution_time_ = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return false;
    }
}

float OrtRun::getLastExecutionTime() const {
    return last_execution_time_;
}

Ort::Value OrtRun::createTensor(const std::vector<float>& data, 
                               const std::vector<int64_t>& shape) {
    // 使用与参考代码相同的MemoryInfo配置
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
    return Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(data.data()), 
        data.size(),
        shape.data(), 
        shape.size());
}

Ort::Value OrtRun::createTensor(const float* data, 
                               size_t data_size,
                               const std::vector<int64_t>& shape) {
    // 使用与参考代码相同的MemoryInfo配置
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
    return Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(data), 
        data_size,
        shape.data(), 
        shape.size());
}