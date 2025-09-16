#!/bin/bash

# 下载与Eigen3 3.3.7兼容的OpenCV版本

echo "=========================================="
echo "下载与Eigen3 3.3.7兼容的OpenCV版本"
echo "=========================================="

# 创建目录
mkdir -p local_opencv
cd local_opencv

# 下载OpenCV 4.2.0（与Eigen3 3.3.7兼容）
echo "正在下载OpenCV 4.2.0..."
wget -O opencv-4.2.0-linux-x64.tar.gz https://github.com/opencv/opencv/releases/download/4.2.0/opencv-4.2.0-linux-x64.tar.gz

# 解压
echo "正在解压..."
tar -xzf opencv-4.2.0-linux-x64.tar.gz

echo "=========================================="
echo "OpenCV 4.2.0下载完成！"
echo "目录: $(pwd)/opencv-4.2.0"
echo "注意：此版本不支持USAC_MAGSAC，但兼容Eigen3 3.3.7"
echo "=========================================="
