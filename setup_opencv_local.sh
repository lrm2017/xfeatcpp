#!/bin/bash

# 下载并使用本地OpenCV版本的脚本
# 支持USAC_MAGSAC的OpenCV 4.8.0版本

set -e

echo "=========================================="
echo "设置本地OpenCV环境 (支持USAC_MAGSAC)"
echo "=========================================="

# 创建本地OpenCV目录
LOCAL_OPENCV_DIR="./local_opencv"
mkdir -p $LOCAL_OPENCV_DIR
cd $LOCAL_OPENCV_DIR

echo "1. 下载OpenCV 4.8.0预编译版本..."

# 下载OpenCV 4.8.0预编译版本
if [ ! -f "opencv-4.8.0-linux-x64.tar.gz" ]; then
    echo "正在下载OpenCV 4.8.0..."
    wget -O opencv-4.8.0-linux-x64.tar.gz https://github.com/opencv/opencv/releases/download/4.8.0/opencv-4.8.0-linux-x64.tar.gz
else
    echo "OpenCV 4.8.0已存在，跳过下载"
fi

# 解压
if [ ! -d "opencv-4.8.0" ]; then
    echo "正在解压OpenCV..."
    tar -xzf opencv-4.8.0-linux-x64.tar.gz
else
    echo "OpenCV已解压，跳过"
fi

# 设置环境变量
echo "2. 设置环境变量..."
export OPENCV_ROOT_DIR="$(pwd)/opencv-4.8.0"
export OpenCV_DIR="$OPENCV_ROOT_DIR"
export OpenCV_INCLUDE_DIRS="$OPENCV_ROOT_DIR/include"
export OpenCV_LIBS_DIR="$OPENCV_ROOT_DIR/lib"

echo "OpenCV_ROOT_DIR: $OPENCV_ROOT_DIR"
echo "OpenCV_INCLUDE_DIRS: $OPENCV_INCLUDE_DIRS"
echo "OpenCV_LIBS_DIR: $OpenCV_LIBS_DIR"

# 检查USAC_MAGSAC支持
echo "3. 检查USAC_MAGSAC支持..."
if [ -f "$OPENCV_ROOT_DIR/include/opencv4/opencv2/calib3d.hpp" ]; then
    if grep -q "USAC_MAGSAC" "$OPENCV_ROOT_DIR/include/opencv4/opencv2/calib3d.hpp"; then
        echo "✓ 支持USAC_MAGSAC"
    else
        echo "✗ 不支持USAC_MAGSAC"
    fi
else
    echo "✗ 找不到calib3d.hpp文件"
fi

# 创建环境设置脚本
echo "4. 创建环境设置脚本..."
cat > ../set_opencv_env.sh << EOF
#!/bin/bash
# OpenCV 4.8.0 环境设置脚本

export OPENCV_ROOT_DIR="$OPENCV_ROOT_DIR"
export OpenCV_DIR="$OPENCV_ROOT_DIR"
export OpenCV_INCLUDE_DIRS="$OPENCV_ROOT_DIR/include"
export OpenCV_LIBS_DIR="$OPENCV_ROOT_DIR/lib"

# 添加到库路径
export LD_LIBRARY_PATH="$OPENCV_ROOT_DIR/lib:\$LD_LIBRARY_PATH"

echo "OpenCV 4.8.0环境已设置"
echo "OPENCV_ROOT_DIR: \$OPENCV_ROOT_DIR"
echo "OpenCV_INCLUDE_DIRS: \$OPENCV_INCLUDE_DIRS"
echo "OpenCV_LIBS_DIR: \$OpenCV_LIBS_DIR"
EOF

chmod +x ../set_opencv_env.sh

echo "=========================================="
echo "OpenCV 4.8.0设置完成！"
echo "=========================================="
echo "使用方法："
echo "1. 运行: source set_opencv_env.sh"
echo "2. 然后编译项目: mkdir build && cd build && cmake .. && make"
echo "=========================================="
