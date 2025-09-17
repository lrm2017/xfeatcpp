#include "feat_extractor.h"
#include "matcher.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <string>
#include <xfeat.h>
#include <Eigen/Dense>
#include <cstdlib>

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

// Eigen Matrix 转 OpenCV Mat 的辅助函数
cv::Mat eigenToCvMat(const Eigen::MatrixXf& eigen_mat) {
    cv::Mat cv_mat(eigen_mat.rows(), eigen_mat.cols(), CV_32F);
    for (int i = 0; i < eigen_mat.rows(); i++) {
        for (int j = 0; j < eigen_mat.cols(); j++) {
            cv_mat.at<float>(i, j) = eigen_mat(i, j);
        }
    }
    return cv_mat;
}

// 双向匹配 - 使用Eigen库简化矩阵操作
void bidirectionalMatch(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches, float min_cossim=0.82f) {

    
    // 转换为Eigen矩阵
    Eigen::MatrixXf desc1 = cvMatToEigen(descriptors1);
    Eigen::MatrixXf desc2 = cvMatToEigen(descriptors2);
    // cv::Mat cv_cossim = descriptors1 * descriptors2.t(); // cv矩阵运算速度很慢
    
    // 计算余弦相似度矩阵 - 只需要一个矩阵！
    Eigen::MatrixXf cossim = desc1 * desc2.transpose(); //cvMatToEigen(cv_cossim);//
    
    // 找出每行的最大值和索引
    Eigen::VectorXf row_max_values = cossim.rowwise().maxCoeff();
    Eigen::VectorXi row_max_indices(cossim.rows());
    
    // 找出每列的最大值索引（用于双向一致性检查）
    Eigen::VectorXi col_max_indices(cossim.cols());
    
    #pragma omp parallel for
    for (int i = 0; i < cossim.rows(); i++) {
        cossim.row(i).maxCoeff(&row_max_indices(i));
        // if( i<5)
        //     std::cout <<"row_max_indx" << i << ":" << row_max_indices(i) << std::endl;
    }
    
    #pragma omp parallel for
    for (int j = 0; j < cossim.cols(); j++) {
        cossim.col(j).maxCoeff(&col_max_indices(j));
        // if( j<5)
        //     std::cout <<"col_max_indx" << j << ":" << col_max_indices(j) << std::endl;
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
    
    // // Calculate the Homography matrix
    // cv::Mat H, mask;
    
    // // 尝试使用USAC_MAGSAC（如果支持），否则使用RANSAC
    // try {
    //     // 检查OpenCV版本是否支持USAC_MAGSAC
    //     #if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 5
    //     H = cv::findHomography(valid_ref, valid_dst, cv::USAC_MAGSAC, 3.5, mask, 1000, 0.999);
    //     std::cout << "使用USAC_MAGSAC算法" << std::endl;
    //     #else
    //     H = cv::findHomography(valid_ref, valid_dst, cv::RANSAC, 3.5, mask, 1000, 0.999);
    //     std::cout << "使用RANSAC算法" << std::endl;
    //     #endif
    // } catch (const cv::Exception& e) {
    //     // 如果USAC_MAGSAC不可用，回退到RANSAC
    //     std::cout << "USAC_MAGSAC不可用，使用RANSAC: " << e.what() << std::endl;
    //     H = cv::findHomography(valid_ref, valid_dst, cv::RANSAC, 3.5, mask, 1000, 0.999);
    // }
    
    // if (H.empty()) {
    //     std::cout << "警告：无法计算单应性矩阵" << std::endl;
    //     return cv::Mat();
    // }
    // // std::cout << "H: " << H << std::endl;
    // return H;
}

// void drawMatches(const cv::Mat& img1, const cv::Mat& img2,
//                 const std::vector<cv::Point2f>& keypoints1,
//                 const std::vector<cv::Point2f>& keypoints2,
//                 const std::vector<cv::DMatch>& matches,
//                 const std::string& window_name) {
//     // 创建匹配图像
//     cv::Mat img_matches;
//     std::vector<cv::KeyPoint> kp1, kp2;
    
//     // 转换关键点格式
//     valid_dst = cv::Mat(valid_dst);
    
// }


void drawMatches(const cv::Mat& img1, const cv::Mat& img2,
                const std::vector<cv::Point2f>& keypoints1,
                const std::vector<cv::Point2f>& keypoints2,
                const std::vector<cv::DMatch>& matches,
                const std::string& window_name) {
    // 创建匹配图像
    cv::Mat img_matches;
    std::vector<cv::KeyPoint> kp1, kp2;
    
    // 转换关键点格式
    for (const auto& pt : keypoints1) {
        kp1.emplace_back(pt.x, pt.y, 1.0f);
        if(pt.x < 0 or pt.y < 0) {
            std::cout << "kp1: " << pt.x << " " << pt.y << std::endl;
        }
    }
    for (const auto& pt : keypoints2) {
        kp2.emplace_back(pt.x, pt.y, 1.0f);
        if(pt.x < 0 or pt.y < 0) {
            std::cout << "kp2: " << pt.x << " " << pt.y << std::endl;
        }
    }
    
    // 绘制匹配
    cv::drawMatches(img1, kp1, img2, kp2, matches, img_matches,
                   cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // 缩放显示
    cv::Mat display_img;
    // double scale = 0.3;
    // cv::resize(img_matches, display_img, cv::Size(), 1.0, 1.0);
    // 固定1920x1080
    cv::resize(img_matches, display_img, cv::Size(1920, 1080));
    
    cv::imshow(window_name, display_img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
    std::cout << "XFeat C++ 特征提取与匹配测试程序" << std::endl;
    std::cout << "=====================================" << std::endl;

    std::string image1_path;
    std::string image2_path;
    std::string xfeat_model_path;
    std::string lighterglue_model_path;
    float min_score = 0.82f;
    
    // 智能解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // 检查是否是XFeat模型文件
        if (arg.find("xfeat") != std::string::npos && arg.find(".onnx") != std::string::npos) {
            xfeat_model_path = arg;
        }
        // 检查是否是LightGlue模型文件
        else if (arg.find("lighterglue") != std::string::npos && arg.find(".onnx") != std::string::npos) {
            lighterglue_model_path = arg;
        }
        // 检查是否是图像文件
        else if (arg.find(".png") != std::string::npos 
            || (arg.find(".bmp") != std::string::npos 
            || arg.find(".jpg") != std::string::npos 
            || arg.find(".jpeg") != std::string::npos)) {
            if (image1_path.empty()) {
                image1_path = arg;
            } else if (image2_path.empty()) {
                image2_path = arg;
            }
        }
        // 检查是否是数值（阈值）
        else if (std::isdigit(arg[0]) || (arg[0] == '-' && std::isdigit(arg[1]))) {
            min_score = atof(arg.c_str());
        }
    }
    
    // 打印解析结果
    std::cout << "\n=== 命令行参数解析结果 ===" << std::endl;
    std::cout << "图像1: " << (image1_path.empty() ? "未指定" : image1_path) << std::endl;
    std::cout << "图像2: " << (image2_path.empty() ? "未指定" : image2_path) << std::endl;
    std::cout << "XFeat模型: " << (xfeat_model_path.empty() ? "未指定" : xfeat_model_path) << std::endl;
    std::cout << "LightGlue模型: " << (lighterglue_model_path.empty() ? "未指定" : lighterglue_model_path) << std::endl;
    std::cout << "匹配阈值: " << min_score << std::endl;
    std::cout << "=========================" << std::endl;
    // 否则使用默认的
    if (image1_path.empty()) {
        // image1_path = "assets/move0.png";
        image1_path = "assets/big2.png";
    }
    if (image2_path.empty()) {
        // image2_path = "assets/move1.png";
        image2_path = "assets/big3.png";
    }
    if (xfeat_model_path.empty()) {
        // xfeat_model_path = "onnx/xfeat_2048_2048x3072.onnx";
        xfeat_model_path = "onnx/xfeat_2048_2048x3072.onnx";
    }
    // if (lighterglue_model_path.empty()) {
    //     lighterglue_model_path = "onnx/lighterglue_2048_L3.onnx";
    // }

    // 图像路径
    // std::string image1_path = "assets/move0.png";    
    // std::string image2_path = "assets/move1.png";

    // 模型路径
    // std::string xfeat_model_path = "onnx/xfeat_2048_2048x3072.onnx"; // 关键点，高x宽
    // std::string lighterglue_model_path = "onnx/lighterglue_L3.onnx";

    // 读取图像
    cv::Mat image1 = cv::imread(image1_path);
    cv::Mat image2 = cv::imread(image2_path);

    if (image1.empty() || image2.empty()) {
        std::cerr << "Error: Could not load images" << std::endl;
        return -1;
    }

    std::cout << "Image1 size: " << image1.size() << std::endl;
    std::cout << "Image2 size: " << image2.size() << std::endl;

    try {
        // 创建特征提取器
        std::cout << "\n=== 创建特征提取器 ===" << std::endl;
        FeatExtractor extractor(xfeat_model_path, "xfeat");

        std::cout << "XFeat input shapes: " << extractor.getInputWH() << std::endl;
        cv::Mat resized_img1, resized_img2;
        cv::resize(image1, resized_img1, extractor.getInputWH());
        cv::resize(image2, resized_img2, extractor.getInputWH());
        

        // 提取特征
        std::cout << "\n=== 提取图像1特征 ===" << std::endl;
        std::vector<cv::Point2f> keypoints1;
        cv::Mat descriptors1;
        // 运行时间
        auto start_time = std::chrono::high_resolution_clock::now();
        if (!extractor.run(resized_img1, keypoints1, descriptors1)) {
            std::cerr << "Failed to extract features from image1" << std::endl;
            return -1;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "XFeat extraction time: " << duration.count() << " ms" << std::endl;
        std::cout << "Extracted " << keypoints1.size() << " keypoints from image1" << std::endl;
        
        std::cout << "\n=== 提取图像2特征 ===" << std::endl;
        std::vector<cv::Point2f> keypoints2;
        cv::Mat descriptors2;
        start_time = std::chrono::high_resolution_clock::now();
        if (!extractor.run(resized_img2, keypoints2, descriptors2)) {
            std::cerr << "Failed to extract features from image2" << std::endl;
            return -1;
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "XFeat extraction time: " << duration.count() << " ms" << std::endl;
        std::cout << "Extracted " << keypoints2.size() << " keypoints from image2" << std::endl;
        
        // 匹配特征
        std::cout << "\n=== 匹配特征 ===" << std::endl;
        std::vector<cv::DMatch> matches;
        std::vector<float> scores;

        if(!lighterglue_model_path.empty()){
            // 创建匹配器 (注释掉LighterGlue匹配器)
            std::cout << "\n=== 创建匹配器 ===" << std::endl;
            Matcher lighterglue_matcher(lighterglue_model_path, "lighterglue", true, min_score, extractor.getInputWH());  // 设置很低的阈值，保留所有匹配
            // LighterGlue lighterglue_matcher(lighterglue_model_path);

            //使用真实关键点进行匹配
            if (!lighterglue_matcher.run(keypoints1, keypoints2, descriptors1, descriptors2, matches, scores)) {
                std::cerr << "Failed to run LighterGlue matching" << std::endl;
                return -1;
            }
            //运行时间
            start_time = std::chrono::high_resolution_clock::now();
            lighterglue_matcher.run(keypoints1, keypoints2, descriptors1, descriptors2, matches, scores);
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "LighterGlue matching time: " << duration.count() << " ms" << std::endl;

            drawMatches(resized_img1, resized_img2, keypoints1, keypoints2, matches, "All Matches");
        }
        else{
            // 运行时间
            start_time = std::chrono::high_resolution_clock::now();
            bidirectionalMatch(descriptors1, descriptors2, matches, min_score);
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Bidirectional matching time: " << duration.count() << " ms" << std::endl;
            std::vector<cv::Point2f> valid_ref, valid_dst;
            std::cout << "Matches: " << matches.size() << std::endl;

            // 筛选出匹配点的坐标
            std::vector<cv::Point2f> pts1, pts2;
            for (const auto& match : matches) {
                pts1.push_back(keypoints1[match.queryIdx]);
                pts2.push_back(keypoints2[match.trainIdx]);
                // std::cout << "Match: " << keypoints1[match.queryIdx] << " " << keypoints2[match.trainIdx] << std::endl;
            }

            int ret = validatePoints(pts1, pts2, valid_ref, valid_dst);
            std::cout << "valid_ref: " << valid_ref.size() << std::endl;
            std::cout << "valid_dst: " << valid_dst.size() << std::endl;
            if (ret < 0) {
                std::cerr << "Failed to validate points" << std::endl;
                return -1;
            }
            start_time = std::chrono::high_resolution_clock::now();
            cv::Mat mask;
            cv::Mat H = cv::findHomography(valid_ref, valid_dst, cv::RANSAC, 3.5, mask, 1000, 0.999);
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "RANSAC time: " << duration.count() << " ms" << std::endl;
            if (H.empty()) {
                std::cerr << "Failed to find homography" << std::endl;
                // return -1;
            }
            else {        
                std::cout << "H: " << H << std::endl;
                // 行列式
                double det = cv::determinant(H(cv::Rect(0, 0, 2, 2)));
                std::cout << "行列式: " << det << std::endl;
                std::cout << "valid_ref: " << valid_ref.size() << std::endl;
                std::cout << "valid_dst: " << valid_dst.size() << std::endl;
            }
        
            if (!H.empty() && !mask.empty()) {
                std::vector<cv::DMatch> inlier_matches;
                std::vector<cv::Point2f> inlier_pts1, inlier_pts2;
                
                // 筛选RANSAC内点
                for (int i = 0; i < static_cast<int>(matches.size()) && i < mask.rows; ++i) {
                    if (mask.at<uchar>(i)) {
                        inlier_matches.push_back(matches[i]);
                        inlier_pts1.push_back(keypoints1[matches[i].queryIdx]);
                        inlier_pts2.push_back(keypoints2[matches[i].trainIdx]);
                    }
                }
                
                std::cout << "RANSAC筛选后的内点数量: " << inlier_matches.size() << std::endl;
                
                // 绘制RANSAC筛选后的匹配
                drawMatches(resized_img1, resized_img2, keypoints1, keypoints2, inlier_matches, "RANSAC Filtered Matches");
            } else 
            {
                std::cout << "RANSAC失败，绘制所有匹配点" << std::endl;
                // 如果RANSAC失败，绘制所有匹配点
                drawMatches(resized_img1, resized_img2, keypoints1, keypoints2, matches, "All Matches");
            }
        }


        // matchWithXFeat(descriptors1, descriptors2, matches, 0.82f);
        // std::cout << "Matches: " << matches.size() << std::endl;

        // 调试：检查输入数据的一致性
        std::cout << "调试信息 - 检查输入数据一致性:" << std::endl;
        std::cout << "Keypoints1 count: " << keypoints1.size() << std::endl;
        std::cout << "Keypoints2 count: " << keypoints2.size() << std::endl;
        std::cout << "Descriptors1 shape: " << descriptors1.rows << "x" << descriptors1.cols << std::endl;
        std::cout << "Descriptors2 shape: " << descriptors2.rows << "x" << descriptors2.cols << std::endl;
        
        // // 打印前几个关键点和描述符值用于验证
        // if (!keypoints1.empty()) {
        //     std::cout << "First keypoint1: (" << keypoints1[0].x << ", " << keypoints1[0].y << ")" << std::endl;
        // }
        // if (!keypoints2.empty()) {
        //     std::cout << "First keypoint2: (" << keypoints2[0].x << ", " << keypoints2[0].y << ")" << std::endl;
        // }
        // if (!descriptors1.empty()) {
        //     std::cout << "First descriptor1 values: " << descriptors1.at<float>(0, 0) << " " << descriptors1.at<float>(0, 1) << std::endl;
        // }
        // if (!descriptors2.empty()) {
        //     std::cout << "First descriptor2 values: " << descriptors2.at<float>(0, 0) << " " << descriptors2.at<float>(0, 1) << std::endl;
        // }
        
        // // 关键点归一化由Matcher类内部处理，这里不需要手动归一化
        // if (!lighterglue_matcher.run(keypoints1, keypoints2, descriptors1, descriptors2, matches, scores)) {
        //     std::cerr << "Failed to match features" << std::endl;
        //     return -1;
        // }
        

        // std::cout << "Pts1 size: " << pts1.size() << std::endl;
        // std::cout << "Pts2 size: " << pts2.size() << std::endl;

        // 使用所有关键点进行绘制，drawMatches函数会根据matches中的索引来连接对应的关键点
        // drawMatches(image1, image2, keypoints1, keypoints2, matches, "Feature Matching Result");
        // std::cout << "Matches size: " << matches.size() << std::endl;

    //  // 传统方法
    //  // 3. 暴力匹配
    // auto t_match_start = std::chrono::high_resolution_clock::now();
    // cv::BFMatcher matcher(cv::NORM_L2);  // 使用L2距离，适用于CV_32F类型
    // std::vector<cv::DMatch> matches;
    // matcher.match(descriptors1, descriptors2, matches);

    // auto t_match_end = std::chrono::high_resolution_clock::now();
    // double match_ms = std::chrono::duration<double, std::milli>(t_match_end - t_match_start).count();
    // std::cout << "特征匹配耗时: " << match_ms << " ms" << std::endl;

    //  if (matches.size() < 10) {
    //      std::cerr << "匹配点太少！" << std::endl;
    //      return -1;
    //  }

    // // 3.1. 改进的特征匹配质量筛选
    // auto t_sort_start = std::chrono::high_resolution_clock::now();
    
    // std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
    //     return a.distance < b.distance;
    // });

    // size_t N = std::min<size_t>(150, matches.size());
    // std::vector<cv::Point2f> pts1, pts2;
    // std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + N);

    // for (const auto& m : good_matches)
    // {
    //     pts1.push_back(keypoints1[m.queryIdx]);
    //     pts2.push_back(keypoints2[m.trainIdx]);
    // }

    // // 4. RANSAC筛选
    // cv::Mat inlier_mask;
    //  if (pts1.size() < 4 || pts2.size() < 4) {
    //      std::cerr << "有效点太少！" << std::endl;
    //      return -1;
    //  }
    
    // auto t_ransac_start = std::chrono::high_resolution_clock::now();
    
    // // 改进的RANSAC参数设置
    // double ransac_thresh = 2.0;  // 降低阈值，提高精度
    // int max_iterations = 2000;    // 增加最大迭代次数
    // double confidence = 0.99;     // 提高置信度
    
    // cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, ransac_thresh, inlier_mask, max_iterations, confidence);
    
    // auto t_ransac_end = std::chrono::high_resolution_clock::now();
    // double ransac_ms = std::chrono::duration<double, std::milli>(t_ransac_end - t_ransac_start).count();
    // std::cout << "RANSAC耗时: " << ransac_ms << " ms" << std::endl;
    
    //  if (H.empty()) {
    //      std::cerr << "RANSAC失败！" << std::endl;
    //      return -1;
    //  }
    
    // // 4.1. 验证单应矩阵的合理性
    // std::cout << "\n单应矩阵 H:" << std::endl << H << std::endl;
    
    // // 检查单应矩阵的条件数（衡量数值稳定性）
    // cv::SVD svd_check(H);
    // double condition_number = svd_check.w.at<double>(0) / svd_check.w.at<double>(2);
    // std::cout << "单应矩阵条件数: " << condition_number << std::endl;
    
    // if (condition_number > 1e6) {
    //     std::cout << "警告: 单应矩阵条件数过大，可能存在数值不稳定！" << std::endl;
    // }
    
    // // 检查变换是否保持方向性（行列式应该为正）
    // double det = cv::determinant(H(cv::Rect(0, 0, 2, 2)));
    // std::cout << "仿射部分行列式: " << det << std::endl;
    
    // if (det < 0) {
    //     std::cout << "警告: 仿射变换改变了方向性，可能导致镜像效果！" << std::endl;
    // }
        
        // matchWithXFeat(descriptors1, descriptors2, matches, 0.82f);
        
        // // 显示结果
        // std::cout << "\n=== 匹配结果 ===" << std::endl;
        // std::cout << "Total matches: " << matches.size() << std::endl;
        // std::cout << "Feature extraction time: " << extractor.getLastExecutionTime() << " ms" << std::endl;
        // std::cout << "Matching time: " << matcher.getLastExecutionTime() << " ms" << std::endl;
        
        // if (!matches.empty()) {
        //     // 调试信息：打印前几个匹配的坐标
        //     // std::cout << "First 3 match coordinates:" << std::endl;
        //     // for (int i = 0; i < std::min(3, (int)matches.size()); i++) {
        //     //     const auto& match = matches[i];
        //     //     std::cout << "  Match " << i << ": (" 
        //     //              << keypoints1[match.queryIdx].x << ", " << keypoints1[match.queryIdx].y 
        //     //              << ") <-> (" 
        //     //              << keypoints2[match.trainIdx].x << ", " << keypoints2[match.trainIdx].y 
        //     //              << "), distance: " << match.distance << std::endl;
        //     // }
            
        //     // // 计算匹配统计
        //     // std::vector<float> distances;
        //     // for (const auto& match : matches) {
        //     //     distances.push_back(match.distance);
        //     // }
            
        //     // std::sort(distances.begin(), distances.end());
        //     // float avg_distance = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();
        //     // float median_distance = distances[distances.size() / 2];
        //     // float min_distance = distances[0];
        //     // float max_distance = distances[distances.size() - 1];
            
        //     // std::cout << "Match distance statistics:" << std::endl;
        //     // std::cout << "  Average distance: " << std::fixed << std::setprecision(3) << avg_distance << std::endl;
        //     // std::cout << "  Median distance: " << std::fixed << std::setprecision(3) << median_distance << std::endl;
        //     // std::cout << "  Min distance: " << std::fixed << std::setprecision(3) << min_distance << std::endl;
        //     // std::cout << "  Max distance: " << std::fixed << std::setprecision(3) << max_distance << std::endl;
            
        //                  // 显示匹配结果 - 使用原始关键点和筛选后的匹配
            //  drawMatches(image1, image2, keypoints1, keypoints2, good_matches, "Feature Matching Result");
             
            //  // 保存结果
            //  cv::Mat img_matches;
            //  std::vector<cv::KeyPoint> kp1, kp2;
            //  for (const auto& pt : keypoints1) {
            //      kp1.emplace_back(pt.x, pt.y, 1.0f);
            //  }
            //  for (const auto& pt : keypoints2) {
            //      kp2.emplace_back(pt.x, pt.y, 1.0f);
            //  }
             
            //  cv::drawMatches(image1, kp1, image2, kp2, good_matches, img_matches,
            //                 cv::Scalar::all(-1), cv::Scalar::all(-1),
            //                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            
            // cv::imwrite("matching_result.jpg", img_matches);
        //     std::cout << "Matching result saved to: matching_result.jpg" << std::endl;
        // } else {
        //     std::cout << "No matches found!" << std::endl;
        // }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "\n测试完成!" << std::endl;
    return 0;
}