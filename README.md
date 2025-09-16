# XFeat C++ 特征提取与匹配

基于ONNX Runtime的XFeat特征提取与匹配C++实现，支持3072x2048分辨率的图像处理。

## 功能特性

- ✅ 基于ONNX Runtime的高性能推理
- ✅ 支持XFeat特征提取
- ✅ FLANN-based特征匹配
- ✅ 支持3072x2048分辨率图像
- ✅ 完整的C++ API接口
- ✅ 可视化匹配结果
- ✅ 性能统计和测试

## 系统要求

### 必需依赖

- **CMake** >= 3.16
- **OpenCV** >= 4.0 (推荐4.5+)
- **ONNX Runtime** >= 1.12.0
- **C++17** 编译器 (GCC 7+ 或 Clang 5+)

### 推荐环境

- Ubuntu 20.04+ / CentOS 8+ / Debian 11+
- 8GB+ RAM
- 支持AVX2的CPU (可选，用于优化)

## 安装依赖

### 1. 安装OpenCV

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libopencv-dev

# 或者从源码编译 (推荐)
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_GENERATE_PKGCONFIG=ON ..
make -j$(nproc)
sudo make install
```

### 2. 安装ONNX Runtime

#### 方法1: 使用预编译版本 (推荐)

```bash
# 下载并安装ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
sudo mv onnxruntime-linux-x64-1.16.3 /usr/local/onnxruntime

# 设置环境变量
echo 'export ONNXRUNTIME_ROOT_DIR=/usr/local/onnxruntime' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/onnxruntime/lib' >> ~/.bashrc
source ~/.bashrc
```

#### 方法2: 使用包管理器

```bash
# Ubuntu (如果可用)
sudo apt install onnxruntime-dev

# 或者使用conda
conda install onnxruntime
```

### 3. 安装其他依赖

```bash
# Ubuntu/Debian
sudo apt install build-essential cmake pkg-config

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake3 pkgconfig
```

## 构建项目

### 使用构建脚本 (推荐)

```bash
# 构建发布版本
./build.sh release

# 构建调试版本
./build.sh debug

# 清理构建目录
./build.sh clean

# 运行测试
./build.sh test

# 查看帮助
./build.sh help
```

### 手动构建

```bash
# 创建构建目录
mkdir build && cd build

# 配置CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)

# 运行
cd bin
./xfeatcpp
```

## 使用方法

### 基本用法

```bash
# 单张图像特征提取
./xfeatcpp image1.jpg

# 两张图像特征匹配
./xfeatcpp image1.jpg image2.jpg
```

### 编程接口

```cpp
#include "xfeat.h"

// 创建XFeat实例
XFeat xfeat("path/to/model.onnx");

// 加载图像
cv::Mat image = cv::imread("image.jpg");

// 提取特征
std::vector<cv::Point2f> keypoints;
cv::Mat descriptors;
xfeat.detectAndCompute(image, keypoints, descriptors);

// 匹配特征
std::vector<cv::Point2f> keypoints1, keypoints2;
std::vector<cv::DMatch> matches;
xfeat.detectAndMatch(image1, image2, keypoints1, keypoints2, matches);
```

## API 文档

### XFeat 类

#### 构造函数
```cpp
XFeat(const std::string& model_path = "onnx/xfeat_4096_3072x2048.onnx", 
      const std::string& device = "cpu");
```

#### 主要方法

- `detectAndCompute()`: 从图像中提取特征点和描述符
- `match()`: 匹配两组描述符
- `detectAndMatch()`: 完整的特征提取和匹配流程

详细API文档请参考 `include/xfeat.h` 文件。

## 性能优化

### 编译优化

```bash
# 使用Release模式
cmake .. -DCMAKE_BUILD_TYPE=Release

# 启用更多优化选项
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O3 -march=native -DNDEBUG"
```

### 运行时优化

- 使用GPU加速 (需要CUDA版本的ONNX Runtime)
- 调整图像预处理参数
- 优化FLANN匹配参数

## 测试

项目包含完整的测试用例：

1. **特征提取测试**: 验证单张图像的特征检测
2. **特征匹配测试**: 验证两张图像的特征匹配
3. **性能测试**: 测量处理时间和内存使用

运行测试：
```bash
./build.sh test
```

## 故障排除

### 常见问题

1. **ONNX Runtime未找到**
   ```bash
   # 设置环境变量
   export ONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/onnxruntime/lib
   ```

2. **OpenCV版本不兼容**
   ```bash
   # 检查OpenCV版本
   pkg-config --modversion opencv4
   
   # 如果版本过低，请升级到4.0+
   ```

3. **内存不足**
   - 减少输入图像分辨率
   - 降低最大关键点数量
   - 使用更少的内存密集型匹配算法

4. **模型加载失败**
   - 检查ONNX模型文件路径
   - 验证模型文件完整性
   - 确认ONNX Runtime版本兼容性

### 调试模式

```bash
# 构建调试版本
./build.sh debug

# 使用GDB调试
gdb build/bin/xfeatcpp
```

## 项目结构

```
xfeatcpp/
├── CMakeLists.txt          # CMake构建配置
├── build.sh               # 构建脚本
├── README.md              # 项目说明
├── include/
│   └── xfeat.h            # XFeat类头文件
├── src/
│   ├── xfeat.cpp          # XFeat类实现
│   └── main.cpp           # 主程序
├── onnx/
│   └── xfeat_4096_3072x2048.onnx  # ONNX模型文件
└── assets/                # 测试图像
    ├── move1.png
    └── big6.png
```

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 致谢

- [XFeat](https://github.com/verlab/XFeat) - 原始Python实现
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - 推理引擎
- [OpenCV](https://opencv.org/) - 计算机视觉库