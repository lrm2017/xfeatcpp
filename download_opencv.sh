#!/bin/bash

# 下载OpenCV 4.8.0预编译版本的简单脚本

echo "=========================================="
echo "下载OpenCV 4.8.0预编译版本"
echo "=========================================="

# 创建目录
mkdir -p local_opencv
cd local_opencv

# 下载OpenCV 4.8.0
echo "正在下载OpenCV 4.8.0..."
wget -O opencv-4.8.0-linux-x64.tar.gz https://github.com/opencv/opencv/releases/download/4.8.0/opencv-4.8.0-linux-x64.tar.gz

# 解压
echo "正在解压..."
tar -xzf opencv-4.8.0-linux-x64.tar.gz

# 检查USAC_MAGSAC支持
echo "检查USAC_MAGSAC支持..."
if grep -q "USAC_MAGSAC" opencv-4.8.0/include/opencv4/opencv2/calib3d.hpp; then
    echo "✓ 支持USAC_MAGSAC"
else
    echo "✗ 不支持USAC_MAGSAC"
fi

echo "=========================================="
echo "OpenCV 4.8.0下载完成！"
echo "目录: $(pwd)/opencv-4.8.0"
echo "=========================================="
