#ifndef ORT_RUN_H
#define ORT_RUN_H

#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

class OrtRun {
public:
    OrtRun(const std::string& model_path, const std::string& name = "OrtRun", bool force_cpu = true);
    virtual ~OrtRun();
    
    // 通用推理接口 - 使用移动语义
    bool run(std::vector<Ort::Value>&& input_tensors,
             std::vector<Ort::Value>& output_tensors);
    
    // 获取执行时间
    float getLastExecutionTime() const;
    
    // 获取模型信息
    const std::vector<std::string>& getInputNames() const { return input_names_; }
    const std::vector<std::string>& getOutputNames() const { return output_names_; }
    const std::vector<std::vector<int64_t>>& getInputShapes() const { return input_shapes_; }
    const std::vector<std::vector<int64_t>>& getOutputShapes() const { return output_shapes_; }
    
    // 创建张量的辅助函数
    Ort::Value createTensor(const std::vector<float>& data, 
                           const std::vector<int64_t>& shape);
    Ort::Value createTensor(const float* data, 
                           size_t data_size,
                           const std::vector<int64_t>& shape);

protected:
    // 子类可以重写的虚函数
    virtual bool preprocess(std::vector<Ort::Value>& /*inputs*/) { return true; }
    virtual bool postprocess(std::vector<Ort::Value>& /*outputs*/) { return true; }

private:
    // ONNX Runtime 相关成员
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::unique_ptr<Ort::SessionOptions> ort_session_options_;
    std::unique_ptr<Ort::MemoryInfo> ort_memory_info_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> ort_allocator_;
    
    // 模型信息
    std::string model_path_;
    std::string name_;
    bool force_cpu_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    // 执行时间统计
    mutable float last_execution_time_;
    
    // 内部方法
    bool initializeOrt();
    bool loadModel();
};

#endif