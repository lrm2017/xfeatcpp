#include "feat_extractor.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <string>
#include <xfeat.h>
#include <Eigen/Dense>
#include <cstdlib>
#include <chrono>
#include <atomic>
#include "DVPCamera.h"  // 海康摄像头API

// OpenCV Mat 转 Eigen Matrix 的辅助函数
Eigen::MatrixXf cvMatToEigen(const cv::Mat& cv_mat) {
    cv::Mat cvMat = cv_mat.clone();
    if (!cvMat.isContinuous()) cvMat = cvMat.clone();
    
    if (cvMat.type() == CV_64F) {
        return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            (double*)cvMat.data, 
            cvMat.rows, 
            cvMat.cols
        ).cast<float>();
    } else if (cvMat.type() == CV_32F) {
        return Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            (float*)cvMat.data, 
            cvMat.rows, 
            cvMat.cols
        );
    }
    
    Eigen::MatrixXf eigen_mat(cv_mat.rows, cv_mat.cols);
    for (int i = 0; i < cv_mat.rows; i++) {
        for (int j = 0; j < cv_mat.cols; j++) {
            eigen_mat(i, j) = cv_mat.at<float>(i, j);
        }
    }
    return eigen_mat;
}

// 双向匹配函数
void bidirectionalMatch(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches, float min_cossim=0.82f) {
    Eigen::MatrixXf desc1 = cvMatToEigen(descriptors1);
    Eigen::MatrixXf desc2 = cvMatToEigen(descriptors2);
    
    Eigen::MatrixXf cossim = desc1 * desc2.transpose();
    
    Eigen::VectorXf row_max_values = cossim.rowwise().maxCoeff();
    Eigen::VectorXi row_max_indices(cossim.rows());
    Eigen::VectorXi col_max_indices(cossim.cols());
    
    #pragma omp parallel for
    for (int i = 0; i < cossim.rows(); i++) {
        cossim.row(i).maxCoeff(&row_max_indices(i));
    }
    
    #pragma omp parallel for
    for (int j = 0; j < cossim.cols(); j++) {
        cossim.col(j).maxCoeff(&col_max_indices(j));
    }

    matches.clear();
    
    #pragma omp parallel for
    for (int i = 0; i < cossim.rows(); i++) {
        int j = row_max_indices(i);
        bool isvalid = (row_max_values(i) > min_cossim);
        if (j < cossim.cols() && col_max_indices(j) == i && isvalid) {
            #pragma omp critical
            {
                matches.push_back(cv::DMatch(i, j, row_max_values(i)));
            }
        }
    }
}

// 绘制匹配结果
void drawMatchesOnFrame(const cv::Mat& img1, const cv::Mat& img2,
                       const std::vector<cv::Point2f>& keypoints1,
                       const std::vector<cv::Point2f>& keypoints2,
                       const std::vector<cv::DMatch>& matches,
                       cv::Mat& output_frame) {
    cv::Mat img_matches;
    std::vector<cv::KeyPoint> kp1, kp2;
    
    for (const auto& pt : keypoints1) {
        kp1.emplace_back(pt.x, pt.y, 1.0f);
    }
    for (const auto& pt : keypoints2) {
        kp2.emplace_back(pt.x, pt.y, 1.0f);
    }
    
    cv::drawMatches(img1, kp1, img2, kp2, matches, img_matches,
                   cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    cv::resize(img_matches, output_frame, cv::Size(1280, 720));
}

// 验证匹配点
int validatePoints(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, 
                   std::vector<cv::Point2f>& valid_ref, std::vector<cv::Point2f>& valid_dst) {
    valid_ref.clear();
    valid_dst.clear();
    for (size_t i = 0; i < pts1.size(); i++) {
        if (std::isfinite(pts1[i].x) and std::isfinite(pts1[i].y) and 
            std::isfinite(pts2[i].x) and std::isfinite(pts2[i].y) and
            pts1[i].x >= 0 and pts1[i].y >= 0 and
            pts2[i].x >= 0 and pts2[i].y >= 0) {
            valid_ref.push_back(pts1[i]);
            valid_dst.push_back(pts2[i]);
        }
    }
    if (valid_ref.size() < 10) {
        std::cout << "警告：有效匹配点数量不足（" << valid_ref.size() << "），无法计算单应性矩阵" << std::endl;
        return -1;
    }
    return 0;
}

// 摄像头类型枚举
enum CameraType {
    CAMERA_NONE = 0,
    CAMERA_DVP,      // 海康摄像头
    CAMERA_OPENCV    // 本地摄像头
};

// 混合摄像头管理器
class HybridCameraManager {
private:
    CameraType camera_type_;
    cv::VideoCapture opencv_cap_;
    dvpHandle dvp_handle_;
    bool is_initialized_;
    std::string camera_name_;

public:
    HybridCameraManager() : camera_type_(CAMERA_NONE), dvp_handle_(0), is_initialized_(false) {}
    
    ~HybridCameraManager() {
        close();
    }
    
    bool initialize() {
        // 首先尝试初始化海康摄像头
        if (initializeDvpCamera()) {
            camera_type_ = CAMERA_DVP;
            is_initialized_ = true;
            std::cout << "成功初始化海康摄像头: " << camera_name_ << std::endl;
            return true;
        }
        
        // 如果没有海康摄像头，尝试本地摄像头
        if (initializeOpenCVCamera()) {
            camera_type_ = CAMERA_OPENCV;
            is_initialized_ = true;
            std::cout << "成功初始化本地摄像头" << std::endl;
            return true;
        }
        
        std::cerr << "无法初始化任何摄像头" << std::endl;
        return false;
    }
    
    bool initializeDvpCamera() {
        dvpUint32 count = 0;
        dvpCameraInfo info[8];
        dvpStatus status;
        
        // 刷新设备列表
        status = dvpRefresh(&count);
        if (status != DVP_STATUS_OK || count == 0) {
            std::cout << "未发现海康摄像头设备" << std::endl;
            return false;
        }
        
        if (count > 8) count = 8;
        
        // 尝试打开第一个可用的海康摄像头
        for (dvpUint32 i = 0; i < count; i++) {
            if (dvpEnum(i, &info[i]) == DVP_STATUS_OK) {
                std::cout << "发现海康摄像头[" << i << "]: " << info[i].FriendlyName << std::endl;
                
                // 尝试打开摄像头
                status = dvpOpenByName(info[i].FriendlyName, OPEN_NORMAL, &dvp_handle_);
                if (status == DVP_STATUS_OK) {
                    camera_name_ = std::string(info[i].FriendlyName);
                    
                    // 开始视频流
                    status = dvpStart(dvp_handle_);
                    if (status == DVP_STATUS_OK) {
                        return true;
                    } else {
                        dvpClose(dvp_handle_);
                        dvp_handle_ = 0;
                    }
                }
            }
        }
        
        return false;
    }
    
    bool initializeOpenCVCamera(int camera_id = 0) {
        opencv_cap_.open(camera_id);
        if (!opencv_cap_.isOpened()) {
            return false;
        }
        
        // 设置摄像头参数
        opencv_cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        opencv_cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        opencv_cap_.set(cv::CAP_PROP_FPS, 30);
        
        return true;
    }
    
    bool read(cv::Mat& frame) {
        if (!is_initialized_) {
            return false;
        }
        
        if (camera_type_ == CAMERA_DVP) {
            return readDvpFrame(frame);
        } else if (camera_type_ == CAMERA_OPENCV) {
            return opencv_cap_.read(frame);
        }
        
        return false;
    }
    
    bool readDvpFrame(cv::Mat& frame) {
        dvpFrame dvp_frame;
        void* pBuffer;
        dvpStatus status;
        
        status = dvpGetFrame(dvp_handle_, &dvp_frame, &pBuffer, 1000);  // 1秒超时
        if (status != DVP_STATUS_OK) {
            return false;
        }
        
        // 将DVP帧转换为OpenCV Mat
        if (dvp_frame.format == FORMAT_MONO) {
            frame = cv::Mat(dvp_frame.iHeight, dvp_frame.iWidth, CV_8UC1, pBuffer);
        } else if (dvp_frame.format == FORMAT_RGB24) {
            frame = cv::Mat(dvp_frame.iHeight, dvp_frame.iWidth, CV_8UC3, pBuffer);
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        } else if (dvp_frame.format == FORMAT_BGR24) {
            frame = cv::Mat(dvp_frame.iHeight, dvp_frame.iWidth, CV_8UC3, pBuffer);
        } else {
            std::cerr << "不支持的DVP图像格式: " << dvp_frame.format << std::endl;
            return false;
        }
        
        return true;
    }
    
    void close() {
        if (camera_type_ == CAMERA_DVP && dvp_handle_ != 0) {
            dvpStop(dvp_handle_);
            dvpClose(dvp_handle_);
            dvp_handle_ = 0;
        } else if (camera_type_ == CAMERA_OPENCV && opencv_cap_.isOpened()) {
            opencv_cap_.release();
        }
        
        camera_type_ = CAMERA_NONE;
        is_initialized_ = false;
    }
    
    CameraType getCameraType() const {
        return camera_type_;
    }
    
    std::string getCameraName() const {
        if (camera_type_ == CAMERA_DVP) {
            return "海康摄像头: " + camera_name_;
        } else if (camera_type_ == CAMERA_OPENCV) {
            return "本地摄像头";
        }
        return "未知摄像头";
    }
};

// 混合实时匹配器
class HybridRealtimeMatcher {
private:
    FeatExtractor extractor_;
    HybridCameraManager camera_manager_;
    cv::Mat template_image_;
    std::vector<cv::Point2f> template_keypoints_;
    cv::Mat template_descriptors_;
    cv::Size input_size_;
    float min_score_;
    bool template_set_;
    std::atomic<bool> running_;
    
public:
    HybridRealtimeMatcher(const std::string& xfeat_model_path, float min_score = 0.82f) 
        : extractor_(xfeat_model_path, "xfeat"), min_score_(min_score), template_set_(false), running_(true) {
        input_size_ = extractor_.getInputWH();
        std::cout << "XFeat输入尺寸: " << input_size_ << std::endl;
    }
    
    ~HybridRealtimeMatcher() {
        stop();
    }
    
    bool initializeCamera() {
        return camera_manager_.initialize();
    }
    
    bool setTemplateFromCamera() {
        cv::Mat frame;
        if (!camera_manager_.read(frame)) {
            std::cerr << "无法从摄像头读取图像" << std::endl;
            return false;
        }
        
        return setTemplate(frame);
    }

    bool ishasSetTemplate() {
        return template_set_;
    }
    
    bool setTemplate(const cv::Mat& image) {
        template_image_ = image.clone();
        
        cv::Mat resized_template;
        cv::resize(template_image_, resized_template, input_size_);
        
        std::cout << "正在提取模板特征..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (!extractor_.run(resized_template, template_keypoints_, template_descriptors_)) {
            std::cerr << "模板特征提取失败" << std::endl;
            return false;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "模板特征提取完成，耗时: " << duration.count() << " ms" << std::endl;
        std::cout << "模板关键点数量: " << template_keypoints_.size() << std::endl;
        
        template_set_ = true;
        return true;
    }
    
    void run() {
        if (!template_set_) {
            std::cerr << "请先设置模板图像" << std::endl;
            return;
        }
        
        cv::Mat frame, resized_frame;
        cv::Mat display_frame;
        std::vector<cv::Point2f> current_keypoints;
        cv::Mat current_descriptors;
        std::vector<cv::DMatch> matches;
        
        int frame_count = 0;
        auto last_fps_time = std::chrono::high_resolution_clock::now();
        int fps_frame_count = 0;
        double current_fps = 0.0;
        
        std::vector<cv::Point2f> valid_ref, valid_dst;
        std::vector<cv::DMatch> inlier_matches;
        std::vector<cv::Point2f> inlier_pts1, inlier_pts2;
        
        std::cout << "开始实时匹配，按 'q' 退出，按 'r' 重新设置模板..." << std::endl;
        std::cout << "使用摄像头: " << camera_manager_.getCameraName() << std::endl;
        
        while (running_ && camera_manager_.getCameraType() != CAMERA_NONE) {
            // 读取当前帧
            if (!camera_manager_.read(frame)) {
                std::cerr << "无法读取摄像头帧" << std::endl;
                break;
            }
            
            frame_count++;
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            // 调整到模型输入尺寸
            cv::resize(frame, resized_frame, input_size_);
            
            // 提取当前帧特征
            if (extractor_.run(resized_frame, current_keypoints, current_descriptors)) {
                // 进行匹配
                bidirectionalMatch(template_descriptors_, current_descriptors, matches, min_score_);
                
                // 筛选出匹配点的坐标
                std::vector<cv::Point2f> pts1, pts2;
                for (const auto& match : matches) {
                    pts1.push_back(template_keypoints_[match.queryIdx]);
                    pts2.push_back(current_keypoints[match.trainIdx]);
                }

                int ret = validatePoints(pts1, pts2, valid_ref, valid_dst);
                if (ret < 0) {
                    continue;
                }
                
                cv::Mat mask;
                cv::Mat H = cv::findHomography(valid_ref, valid_dst, cv::RANSAC, 3.5, mask, 1000, 0.999);
                if(H.empty()) {
                    continue;
                }
                
                inlier_matches.clear();
                inlier_pts1.clear();
                inlier_pts2.clear();
                for (int i = 0; i < static_cast<int>(matches.size()) && i < mask.rows; ++i) {
                    if (mask.at<uchar>(i)) {
                        inlier_matches.push_back(matches[i]);
                        inlier_pts1.push_back(template_keypoints_[matches[i].queryIdx]);
                        inlier_pts2.push_back(current_keypoints[matches[i].trainIdx]);
                    }
                }

                // 绘制匹配结果
                cv::Mat resized_template;
                cv::resize(template_image_, resized_template, input_size_);
                drawMatchesOnFrame(resized_template, resized_frame, 
                                 template_keypoints_, current_keypoints, 
                                 inlier_matches, display_frame);
                
                // 计算FPS
                auto current_time = std::chrono::high_resolution_clock::now();
                fps_frame_count++;
                
                // 每秒更新一次FPS值
                auto fps_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_time);
                if (fps_duration.count() >= 1000) {
                    current_fps = (double)fps_frame_count * 1000.0 / fps_duration.count();
                    last_fps_time = current_time;
                    fps_frame_count = 0;
                }
                
                // 添加信息文本（包含摄像头类型和FPS）
                std::string camera_info = (camera_manager_.getCameraType() == CAMERA_DVP) ? "DVP" : "USB";
                std::string info_text = "Camera: " + camera_info + 
                                      " | Matches: " + std::to_string(inlier_matches.size()) + 
                                      " | Frame: " + std::to_string(frame_count) + 
                                      " | FPS: " + std::to_string(current_fps).substr(0, 4);
                cv::putText(display_frame, info_text, cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
                
                // 显示图像
                cv::imshow("混合实时匹配结果", display_frame);
            }
            
            // 处理按键
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {
                std::cout << "退出程序..." << std::endl;
                break;
            } else if (key == 'r') {
                std::cout << "重新设置模板..." << std::endl;
                if (setTemplateFromCamera()) {
                    std::cout << "模板重新设置成功" << std::endl;
                } else {
                    std::cout << "模板重新设置失败" << std::endl;
                }
            }
            
            auto frame_end = std::chrono::high_resolution_clock::now();
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);
            
            // 每100帧打印一次性能信息
            if (frame_count % 100 == 0) {
                std::cout << "帧 " << frame_count << ": 匹配点数 " << inlier_matches.size() 
                         << ", 处理时间 " << frame_duration.count() << " ms, 摄像头: " 
                         << camera_manager_.getCameraName() << std::endl;
            }
        }
    }
    
    void stop() {
        running_ = false;
        camera_manager_.close();
        cv::destroyAllWindows();
    }
};

int main(int argc, char* argv[]) {
    std::cout << "混合摄像头实时图像匹配程序" << std::endl;
    std::cout << "=============================" << std::endl;
    
    std::string xfeat_model_path = "onnx/xfeat_2048_600x800.onnx";
    std::string img_path = "";
    float min_score = 0.82f;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg.find("xfeat") != std::string::npos && arg.find(".onnx") != std::string::npos) {
            xfeat_model_path = arg;
        } else if (arg.find(".png") != std::string::npos 
            || (arg.find(".bmp") != std::string::npos 
            || arg.find(".jpg") != std::string::npos 
            || arg.find(".jpeg") != std::string::npos)) {
            img_path = arg;
        } else if (std::isdigit(arg[0])) {
            min_score = std::stof(arg);
        }
    }
    
    std::cout << "XFeat模型: " << xfeat_model_path << std::endl;
    std::cout << "匹配阈值: " << min_score << std::endl;
    
    try {
        // 创建混合实时匹配器
        HybridRealtimeMatcher matcher(xfeat_model_path, min_score);
        if(!img_path.empty()) {
            cv::Mat img = cv::imread(img_path);
            if(!img.empty()) {
                matcher.setTemplate(img);
            }
            
        }
        
        // 初始化摄像头（自动选择海康或本地摄像头）
        std::cout << "\n正在初始化摄像头..." << std::endl;
        if (!matcher.initializeCamera()) {
            std::cerr << "摄像头初始化失败" << std::endl;
            return -1;
        }
        
        // 等待用户准备好设置模板
        std::cout << "\n请调整摄像头位置，确保目标物体在视野中，然后按任意键开始..." << std::endl;
        cv::waitKey(0);
        
        if(!matcher.ishasSetTemplate()) {
            // 设置模板
            std::cout << "正在设置模板图像..." << std::endl;
            if (!matcher.setTemplateFromCamera()) {
                std::cerr << "模板设置失败" << std::endl;
                return -1;
            }
        }
        
        // 开始实时匹配
        matcher.run();
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "程序结束" << std::endl;
    return 0;
}
