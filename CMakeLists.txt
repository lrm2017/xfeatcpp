cmake_minimum_required(VERSION 3.16)
project(xfeatcpp)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Wall -Wextra")

# 查找OpenCV - 优先使用系统版本（避免版本冲突）
if(DEFINED ENV{OPENCV_ROOT_DIR} AND NOT USE_SYSTEM_OPENCV)
    message(STATUS "使用本地OpenCV版本: $ENV{OPENCV_ROOT_DIR}")
    set(OpenCV_DIR $ENV{OPENCV_ROOT_DIR})
    set(OpenCV_INCLUDE_DIRS $ENV{OpenCV_INCLUDE_DIRS})
    set(OpenCV_LIBS_DIR $ENV{OpenCV_LIBS_DIR})
    
    # 手动设置OpenCV库
    file(GLOB OpenCV_LIBS "${OpenCV_LIBS_DIR}/libopencv_*.so")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries dir: ${OpenCV_LIBS_DIR}")
    message(STATUS "Found ${OpenCV_LIBS} OpenCV libraries")
    
    # 设置OpenCV版本
    set(OpenCV_VERSION "4.8.0")
    set(OpenCV_FOUND TRUE)
else()
    # 使用系统OpenCV（推荐，避免版本冲突）
    message(STATUS "使用系统OpenCV版本（推荐）")
    find_package(OpenCV REQUIRED)
    if(OpenCV_FOUND)
        message(STATUS "OpenCV版本: ${OpenCV_VERSION}")
        message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
        message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
        
        # 检查是否支持USAC_MAGSAC
        if(OpenCV_VERSION VERSION_GREATER_EQUAL "4.5.0")
            message(STATUS "✓ 支持USAC_MAGSAC算法")
        else()
            message(STATUS "✗ 不支持USAC_MAGSAC，将使用RANSAC")
        endif()
    else()
        message(FATAL_ERROR "OpenCV not found")
    endif()
endif()

# 查找Eigen3
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    # Eigen3版本信息通常为空，我们手动检查
    if(EIGEN3_VERSION)
        message(STATUS "Found Eigen3 version: ${EIGEN3_VERSION}")
    else()
        message(STATUS "Found Eigen3 (version info not available)")
    endif()
    message(STATUS "Eigen3 include dirs: ${EIGEN3_INCLUDE_DIR}")
    
    # 如果EIGEN3_INCLUDE_DIR为空，手动查找
    if(NOT EIGEN3_INCLUDE_DIR)
        find_path(EIGEN3_INCLUDE_DIR 
            NAMES Eigen/Dense
            PATHS /usr/include/eigen3
                  /usr/local/include/eigen3
                  /opt/eigen3/include
        )
        if(EIGEN3_INCLUDE_DIR)
            message(STATUS "Manually found Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
        endif()
    endif()
    
    # 检查Eigen3头文件中的版本信息
    if(EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h")
        file(STRINGS "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h" EIGEN_MACROS)
        foreach(line ${EIGEN_MACROS})
            if(line MATCHES "#define EIGEN_WORLD_VERSION ([0-9]+)")
                set(EIGEN_WORLD_VERSION ${CMAKE_MATCH_1})
            elseif(line MATCHES "#define EIGEN_MAJOR_VERSION ([0-9]+)")
                set(EIGEN_MAJOR_VERSION ${CMAKE_MATCH_1})
            elseif(line MATCHES "#define EIGEN_MINOR_VERSION ([0-9]+)")
                set(EIGEN_MINOR_VERSION ${CMAKE_MATCH_1})
            endif()
        endforeach()
        if(EIGEN_WORLD_VERSION AND EIGEN_MAJOR_VERSION AND EIGEN_MINOR_VERSION)
            message(STATUS "Eigen3 detected version: ${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION}")
        else()
            message(STATUS "Eigen3 version detection failed, but library found")
        endif()
    endif()
else()
    message(FATAL_ERROR "Eigen3 not found. Please install Eigen3: sudo apt-get install libeigen3-dev")
endif()

# 查找ONNX Runtime - 使用手动查找方式避免版本冲突
message(STATUS "Searching for ONNX Runtime manually...")

# 设置ONNX Runtime路径（根据实际安装路径调整）
set(ONNXRUNTIME_ROOT_DIR "/usr/local/onnxruntime" CACHE PATH "ONNX Runtime root directory")

# 查找ONNX Runtime头文件
find_path(ONNXRUNTIME_INCLUDE_DIRS
    NAMES onnxruntime_cxx_api.h
    PATHS ${ONNXRUNTIME_ROOT_DIR}/include
          /usr/include
          /usr/local/include
          /opt/onnxruntime/include
    NO_DEFAULT_PATH
)

# 查找ONNX Runtime库文件
find_library(ONNXRUNTIME_LIBRARIES
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOT_DIR}/lib
          ${ONNXRUNTIME_ROOT_DIR}/lib64
          /usr/lib
          /usr/local/lib
          /opt/onnxruntime/lib
    NO_DEFAULT_PATH
)

if(ONNXRUNTIME_INCLUDE_DIRS AND ONNXRUNTIME_LIBRARIES)
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARIES}")
    set(ONNXRUNTIME_FOUND TRUE)
else()
    message(FATAL_ERROR "ONNX Runtime not found. Please install ONNX Runtime or set ONNXRUNTIME_ROOT_DIR")
endif()

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})
if(ONNXRUNTIME_INCLUDE_DIRS)
    include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
endif()

# 添加源文件
set(SOURCES
    src/ort_run.cpp
    src/feat_extractor.cpp
    src/matcher.cpp
    src/main.cpp
    src/xfeat.cpp
    src/lighterGlue.cpp
)

# 添加头文件
set(HEADERS
    include/ort_run.h
    include/feat_extractor.h
    include/matcher.h
    include/xfeat.h
    include/lighterGlue.h
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# 链接库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
if(ONNXRUNTIME_LIBRARIES)
    target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIBRARIES})
    # 添加ONNX Runtime的依赖库
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

# 设置输出目录
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 复制ONNX模型文件到构建目录
file(COPY ${CMAKE_SOURCE_DIR}/onnx/xfeat_4096_3072x2048.onnx
     DESTINATION ${CMAKE_BINARY_DIR}/bin/onnx/)

# 复制测试图片到构建目录
file(COPY ${CMAKE_SOURCE_DIR}/assets/
     DESTINATION ${CMAKE_BINARY_DIR}/bin/assets/)

# 安装目标
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

# 打印配置信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")