#!/bin/bash

# 使用系统OpenCV构建项目（推荐）

echo "=========================================="
echo "使用系统OpenCV构建项目"
echo "=========================================="

# 检查系统OpenCV版本
echo "检查系统OpenCV版本..."
pkg-config --modversion opencv4

# 检查Eigen3版本
echo "检查Eigen3版本..."
if [ -f "/usr/include/eigen3/Eigen/src/Core/util/Macros.h" ]; then
    echo "Eigen3版本："
    grep -E "EIGEN_(WORLD|MAJOR|MINOR)_VERSION" /usr/include/eigen3/Eigen/src/Core/util/Macros.h
fi

# 创建构建目录
mkdir -p build_system
cd build_system

# 运行CMake（强制使用系统OpenCV）
echo "运行CMake配置..."
cmake .. -DUSE_SYSTEM_OPENCV=ON

# 编译
echo "开始编译..."
make -j$(nproc)

echo "=========================================="
echo "构建完成！"
echo "可执行文件: build_system/bin/xfeatcpp"
echo "=========================================="
