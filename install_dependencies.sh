#!/bin/bash

# XFeat C++ 依赖安装脚本
# 支持 Ubuntu/Debian 和 CentOS/RHEL

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# 检测操作系统
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    else
        print_error "无法检测操作系统"
        exit 1
    fi
    
    print_info "检测到操作系统: $OS $VER"
}

# 安装基础依赖
install_basic_deps() {
    print_header "安装基础依赖"
    
    if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
        sudo apt update
        sudo apt install -y build-essential cmake pkg-config wget curl git
    elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y cmake3 pkgconfig wget curl git
        # 创建cmake符号链接
        if [ ! -f /usr/bin/cmake ]; then
            sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
        fi
    else
        print_warning "未识别的操作系统，请手动安装基础依赖"
    fi
}

# 安装OpenCV
install_opencv() {
    print_header "安装OpenCV"
    
    if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
        # 尝试安装OpenCV4
        if sudo apt install -y libopencv-dev; then
            print_info "OpenCV4 安装成功"
        else
            print_warning "OpenCV4 安装失败，尝试安装OpenCV3"
            sudo apt install -y libopencv-dev libopencv-contrib-dev
        fi
    elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
        # CentOS需要从EPEL安装
        sudo yum install -y epel-release
        sudo yum install -y opencv-devel opencv-contrib-devel
    fi
    
    # 验证OpenCV安装
    if pkg-config --exists opencv4; then
        OPENCV_VERSION=$(pkg-config --modversion opencv4)
        print_info "OpenCV4 版本: $OPENCV_VERSION"
    elif pkg-config --exists opencv; then
        OPENCV_VERSION=$(pkg-config --modversion opencv)
        print_info "OpenCV 版本: $OPENCV_VERSION"
    else
        print_error "OpenCV 安装失败"
        exit 1
    fi
}

# 安装ONNX Runtime
install_onnxruntime() {
    print_header "安装ONNX Runtime"
    
    local version="1.16.3"
    local platform="linux-x64"
    local filename="onnxruntime-linux-x64-${version}.tgz"
    local install_dir="/usr/local/onnxruntime"
    
    # 检查是否已安装
    if [ -d "$install_dir" ]; then
        print_info "ONNX Runtime 已安装: $install_dir"
        return 0
    fi
    
    # 下载ONNX Runtime
    print_info "下载 ONNX Runtime ${version}..."
    if [ ! -f "$filename" ]; then
        wget "https://github.com/microsoft/onnxruntime/releases/download/v${version}/${filename}"
    fi
    
    # 解压并安装
    print_info "安装 ONNX Runtime..."
    sudo mkdir -p "$install_dir"
    sudo tar -xzf "$filename" -C /tmp/
    sudo mv "/tmp/onnxruntime-linux-x64-${version}"/* "$install_dir/"
    sudo rm -rf "/tmp/onnxruntime-linux-x64-${version}"
    
    # 设置环境变量
    print_info "设置环境变量..."
    echo 'export ONNXRUNTIME_ROOT_DIR=/usr/local/onnxruntime' | sudo tee -a /etc/environment
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/onnxruntime/lib' | sudo tee -a /etc/environment
    
    # 更新ldconfig
    echo '/usr/local/onnxruntime/lib' | sudo tee /etc/ld.so.conf.d/onnxruntime.conf
    sudo ldconfig
    
    # 清理下载文件
    rm -f "$filename"
    
    print_info "ONNX Runtime 安装完成"
}

# 验证安装
verify_installation() {
    print_header "验证安装"
    
    # 检查CMake
    if command -v cmake &> /dev/null; then
        CMAKE_VERSION=$(cmake --version | head -n1)
        print_info "✓ $CMAKE_VERSION"
    else
        print_error "✗ CMake 未安装"
        return 1
    fi
    
    # 检查OpenCV
    if pkg-config --exists opencv4; then
        OPENCV_VERSION=$(pkg-config --modversion opencv4)
        print_info "✓ OpenCV4 $OPENCV_VERSION"
    elif pkg-config --exists opencv; then
        OPENCV_VERSION=$(pkg-config --modversion opencv)
        print_info "✓ OpenCV $OPENCV_VERSION"
    else
        print_error "✗ OpenCV 未安装"
        return 1
    fi
    
    # 检查ONNX Runtime
    if [ -d "/usr/local/onnxruntime" ]; then
        print_info "✓ ONNX Runtime 已安装"
    else
        print_error "✗ ONNX Runtime 未安装"
        return 1
    fi
    
    print_info "所有依赖安装完成！"
}

# 主函数
main() {
    print_header "XFeat C++ 依赖安装脚本"
    
    detect_os
    install_basic_deps
    install_opencv
    install_onnxruntime
    verify_installation
    
    print_header "安装完成"
    print_info "请重新加载环境变量:"
    print_info "  source /etc/environment"
    print_info "或者重新登录终端"
    print_info ""
    print_info "然后可以运行构建脚本:"
    print_info "  ./build.sh release"
}

# 运行主函数
main "$@"