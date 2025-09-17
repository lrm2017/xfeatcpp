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

// OpenCV Mat 转 Eigen Matrix 的辅助函数 - 正确处理数据布局
Eigen::MatrixXf cvMatToEigen(const cv::Mat& cv_mat) {
    cv::Mat cvMat = cv_mat.clone();
    // 确保矩阵连续且数据类型匹配
    if (!cvMat.isContinuous()) cvMat = cvMat.clone();
    
    // 根据数据类型转换
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
    // 确保矩阵连续且数据类型匹配
    Eigen::MatrixXf eigen_mat(cv_mat.rows, cv_mat.cols);
    for (int i = 0; i < cv_mat.rows; i++) {
        for (int j = 0; j < cv_mat.cols; j++) {
            eigen_mat(i, j) = cv_mat.at<float>(i, j);
        }
    }
    return eigen_mat;
}

// 双向匹配 - 使用Eigen库简化矩阵操作
void bidirectionalMatch(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches, float min_cossim=0.82f) {
    // 转换为Eigen矩阵
    Eigen::MatrixXf desc1 = cvMatToEigen(descriptors1);
    Eigen::MatrixXf desc2 = cvMatToEigen(descriptors2);
    
    // 计算余弦相似度矩阵 - 只需要一个矩阵！
    Eigen::MatrixXf cossim = desc1 * desc2.transpose();
    
    // 找出每行的最大值和索引
    Eigen::VectorXf row_max_values = cossim.rowwise().maxCoeff();
    Eigen::VectorXi row_max_indices(cossim.rows());
    
    // 找出每列的最大值索引（用于双向一致性检查）
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
        // 检查：点(i,j)既是第i行的最大值，也是第j列的最大值
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
    // 创建匹配图像
    cv::Mat img_matches;
    std::vector<cv::KeyPoint> kp1, kp2;
    
    // 转换关键点格式
    for (const auto& pt : keypoints1) {
        kp1.emplace_back(pt.x, pt.y, 1.0f);
    }
    for (const auto& pt : keypoints2) {
        kp2.emplace_back(pt.x, pt.y, 1.0f);
    }
    
    // 绘制匹配
    cv::drawMatches(img1, kp1, img2, kp2, matches, img_matches,
                   cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // 缩放显示到合适大小
    cv::resize(img_matches, output_frame, cv::Size(1280, 720));
}


int validatePoints(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, std::vector<cv::Point2f>& valid_ref, std::vector<cv::Point2f>& valid_dst) {

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

class RealtimeMatcher {
private:
    FeatExtractor extractor_;
    cv::VideoCapture cap_;
    cv::Mat template_image_;
    std::vector<cv::Point2f> template_keypoints_;
    cv::Mat template_descriptors_;
    cv::Size input_size_;
    float min_score_;
    bool template_set_;
    std::atomic<bool> running_;
    
public:
    RealtimeMatcher(const std::string& xfeat_model_path, float min_score = 0.82f) 
        : extractor_(xfeat_model_path, "xfeat"), min_score_(min_score), template_set_(false), running_(true) {
        input_size_ = extractor_.getInputWH();
        std::cout << "XFeat输入尺寸: " << input_size_ << std::endl;
    }
    
    ~RealtimeMatcher() {
        stop();
    }
    
    bool initializeCamera(int camera_id = 0) {
        cap_.open(camera_id);
        if (!cap_.isOpened()) {
            std::cerr << "无法打开摄像头 " << camera_id << std::endl;
            return false;
        }
        
        // 设置摄像头参数
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap_.set(cv::CAP_PROP_FPS, 30);
        
        std::cout << "摄像头已打开，分辨率: " 
                  << cap_.get(cv::CAP_PROP_FRAME_WIDTH) << "x" 
                  << cap_.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
        return true;
    }
    
    bool setTemplateFromCamera() {
        cv::Mat frame;
        if (!cap_.read(frame)) {
            std::cerr << "无法从摄像头读取图像" << std::endl;
            return false;
        }
        
        return setTemplate(frame);
    }
    
    bool setTemplate(const cv::Mat& image) {
        template_image_ = image.clone();
        
        // 调整到模型输入尺寸
        cv::Mat resized_template;
        cv::resize(template_image_, resized_template, input_size_);
        
        // 提取模板特征
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
        double current_fps = 0.0;  // 当前FPS值

        std::vector<cv::Point2f> valid_ref, valid_dst;
        std::vector<cv::DMatch> inlier_matches;
        std::vector<cv::Point2f> inlier_pts1, inlier_pts2;
        
        std::cout << "开始实时匹配，按 'q' 退出，按 'r' 重新设置模板..." << std::endl;
        
        while (running_ && cap_.isOpened()) {
            // 读取当前帧
            if (!cap_.read(frame)) {
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
                    // std::cout << "Match: " << keypoints1[match.queryIdx] << " " << keypoints2[match.trainIdx] << std::endl;
                }

                int ret = validatePoints(pts1, pts2, valid_ref, valid_dst);
                std::cout << "valid_ref: " << valid_ref.size() << std::endl;
                std::cout << "valid_dst: " << valid_dst.size() << std::endl;
                if (ret < 0) {
                    std::cerr << "Failed to validate points" << std::endl;
                    continue;
                }
                cv::Mat mask;
                cv::Mat H = cv::findHomography(valid_ref, valid_dst, cv::RANSAC, 3.5, mask, 1000, 0.999);
                if(H.empty()) { // 如果H为空，则不进行绘制
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
                // std::cout << "H: " << H << std::endl;
                // std::cout << "mask: " << mask << std::endl;
                // std::cout << "current_keypoints: " << current_keypoints.size() << std::endl;
                // std::cout << "template_keypoints_: " << template_keypoints_.size() << std::endl;
                // std::cout << "matches: " << matches.size() << std::endl;
                // std::cout << "min_score_: " << min_score_ << std::endl;
            

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
                if (fps_duration.count() >= 1000) {  // 每1000毫秒（1秒）更新一次
                    current_fps = (double)fps_frame_count * 1000.0 / fps_duration.count();
                    last_fps_time = current_time;
                    fps_frame_count = 0;
                }
                
                // 添加信息文本（包含FPS）
                std::string info_text = "Matches: " + std::to_string(inlier_matches.size()) + 
                                      " | Frame: " + std::to_string(frame_count) + 
                                      " | FPS: " + std::to_string(current_fps).substr(0, 4);
                cv::putText(display_frame, info_text, cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                
                // 显示图像
                cv::imshow("实时匹配结果", display_frame);
            }
            
            // 处理按键
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {  // 'q' 或 ESC
                std::cout << "退出程序..." << std::endl;
                break;
            } else if (key == 'r') {  // 'r' 重新设置模板
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
                std::cout << "帧 " << frame_count << ": 匹配点数 " << matches.size() 
                         << ", 处理时间 " << frame_duration.count() << " ms" << std::endl;
            }
        }
    }
    
    void stop() {
        running_ = false;
        if (cap_.isOpened()) {
            cap_.release();
        }
        cv::destroyAllWindows();
    }
};

int main(int argc, char* argv[]) {
    std::cout << "实时图像匹配程序" << std::endl;
    std::cout << "=================" << std::endl;
    
    std::string xfeat_model_path = "onnx/xfeat_2048_600x800.onnx";
    float min_score = 0.82f;
    int camera_id = 0;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg.find("xfeat") != std::string::npos && arg.find(".onnx") != std::string::npos) {
            xfeat_model_path = arg;
        } else if (arg.find(".png") != std::string::npos || 
                   arg.find(".jpg") != std::string::npos || 
                   arg.find(".bmp") != std::string::npos) {
            // 如果提供了图像文件，可以作为模板
            std::cout << "检测到图像文件参数，但当前版本仅支持摄像头模板" << std::endl;
        } else if (std::isdigit(arg[0])) {
            if (std::stof(arg) < 10) {  // 可能是摄像头ID
                camera_id = std::stoi(arg);
            } else {  // 可能是匹配阈值
                min_score = std::stof(arg);
            }
        }
    }
    
    std::cout << "XFeat模型: " << xfeat_model_path << std::endl;
    std::cout << "摄像头ID: " << camera_id << std::endl;
    std::cout << "匹配阈值: " << min_score << std::endl;
    
    try {
        // 创建实时匹配器
        RealtimeMatcher matcher(xfeat_model_path, min_score);
        
        // 初始化摄像头
        if (!matcher.initializeCamera(camera_id)) {
            std::cerr << "摄像头初始化失败" << std::endl;
            return -1;
        }
        
        // 等待用户准备好设置模板
        std::cout << "\n请调整摄像头位置，确保目标物体在视野中，然后按任意键开始..." << std::endl;
        cv::waitKey(0);
        
        // 设置模板
        std::cout << "正在设置模板图像..." << std::endl;
        if (!matcher.setTemplateFromCamera()) {
            std::cerr << "模板设置失败" << std::endl;
            return -1;
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
