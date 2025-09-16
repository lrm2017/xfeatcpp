#!/bin/bash

# 使用本地OpenCV构建项目的脚本

echo "=========================================="
echo "使用本地OpenCV构建项目"
echo "=========================================="

# 检查本地OpenCV是否存在
if [ ! -d "local_opencv/opencv-4.8.0" ]; then
    echo "本地OpenCV不存在，正在下载..."
    ./download_opencv.sh
fi

# 设置环境变量
export OPENCV_ROOT_DIR="$(pwd)/local_opencv/opencv-4.8.0"
export OpenCV_DIR="$OPENCV_ROOT_DIR"
export OpenCV_INCLUDE_DIRS="$OPENCV_ROOT_DIR/include"
export OpenCV_LIBS_DIR="$OPENCV_ROOT_DIR/lib"

echo "使用OpenCV版本: $OPENCV_ROOT_DIR"
echo "包含目录: $OPENCV_INCLUDE_DIRS"
echo "库目录: $OPENCV_LIBS_DIR"

# 创建构建目录
mkdir -p build_local
cd build_local

# 运行CMake
echo "运行CMake配置..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
echo "开始编译..."
make -j$(nproc)

echo "=========================================="
echo "构建完成！"
echo "可执行文件: build_local/bin/xfeatcpp"
echo "=========================================="
