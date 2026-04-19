//
// Created by entropy on 2026/3/11.
//
#include <vector>
#include <algorithm>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
/*
cv::Rect是矩形类，包含四个成员x、y、width、height，分别表示矩形的左上角坐标和宽高
cv::Point是点类，包含成员x、y，表示点的坐标
cv::Point2f是二维点类，包含成员x、y，表示点的坐标
cv::Mat是矩阵类，存图像的数据
cv::cvtColor函数用于颜色空间转换
cv::inRange函数用于生成黑白掩码
cv::findContours函数用于轮廓检测
cv::boundingRect函数用于计算轮廓的边界矩形，即外接矩形
cv::norm函数用于计算点之间的直线距离
*/
#define BAUDRATE B115200
// 卡尔曼滤波
struct Kalman
{
	double x, vx;
	double p_xx, p_xv, p_vv;
	double dt, q, r;

	// 初始化
	Kalman(double q_, double r_) : x(0), vx(0), p_xx(100.0), p_xv(0), p_vv(1), dt(0.01), q(q_), r(r_) {}

	void predict(double dt_new)
	{
		dt = dt_new;
		x = x + dt * vx;
		p_xx = p_xx + 2 * dt * p_xv + dt * dt * p_vv + q;
		p_xv = p_xv + dt * p_vv;
		p_vv = p_vv + q;
	}

	void update(double z)
	{
		double y = z - x;
		double S = p_xx + r;
		double K_x = p_xx / S;
		double K_v = p_xv / S;
		x = x + K_x * y;
		vx = vx + K_v * y;
		p_xx = (1 - K_x) * p_xx;
		p_xv = (1 - K_x) * p_xv;
		p_vv = p_vv - K_v * p_xv;
	}
};
class ShooterNode : public rclcpp::Node
{
public:
	ShooterNode() : Node("shooter"),
					serial_fd_(-1),
					serial_num_(1), // 串口号默认1
					open_fire_(true),
					target_present_(false),
					enemy_h_(0),
					have_current_target_(false),
					have_prev_(false),
					kf_q_(0.5), kf_r_(0.5),
					has_last_target_(false),
					catch_own_color_(false),
					stop_fire_distance_(400.0),
					lock_duration_(3.4),
					match_distance_threshold_(30.0)
	{
		// 声明参数，rqt改
		this->declare_parameter<double>("stop_fire_distance", stop_fire_distance_);
		this->declare_parameter<double>("lock_duration", lock_duration_);
		this->declare_parameter<double>("match_distance_threshold", match_distance_threshold_);
		this->declare_parameter<int>("serial_num", serial_num_);
		this->declare_parameter<double>("kf_q", kf_q_);
		this->declare_parameter<double>("kf_r", kf_r_);

		lock_start_time_ = this->now();

		// 初始化串口
		serial_num_ = this->get_parameter("serial_num").as_int();
		std::string serial_port = "/dev/pts/" + std::to_string(serial_num_);
		serial_fd_ = open(serial_port.c_str(), O_RDWR | O_NOCTTY);

		// 开火线程
		fire_thread_ = std::thread(&ShooterNode::fireLoop, this);

		// 创建订阅
		sub_ = this->create_subscription<sensor_msgs::msg::Image>("/image_raw", 10, std::bind(&ShooterNode::image_callback, this, std::placeholders::_1));

		RCLCPP_INFO(this->get_logger(), "shooter process was successfully launched. serial_num:%d", serial_num_);
	}

	~ShooterNode() {}

private:
	int serial_fd_;	 // 串口
	int serial_num_; // 串口号
	bool open_fire_;
	bool target_present_;
	std::thread fire_thread_;

	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

	int enemy_h_;
	cv::Rect current_target_rect_;
	bool have_current_target_;
	rclcpp::Time lock_start_time_;
	cv::Point2f prev_target_pos_;
	rclcpp::Time prev_time_;
	Kalman kf_x_{kf_q_, kf_r_};
	bool kf_initialized_ = false;
	bool have_prev_;
	float filtered_angle_;
	bool has_last_target_;
	bool catch_own_color_;

	// 可调节的参数
	double stop_fire_distance_;
	double lock_duration_;
	double match_distance_threshold_;
	double kf_q_;
	double kf_r_;

	void TurnToAngle(float angle) // 指定炮台转角
	{
		unsigned char buf[5] = {0x01};
		memcpy(buf + 1, &angle, 4);
		write(serial_fd_, buf, 5);
		RCLCPP_INFO(this->get_logger(), "Angle:%.2f;", angle);
	}

	void fireLoop() // 开火线程函数
	{
		unsigned char cmd = 0x02;
		while (open_fire_)
		{
			if (target_present_)
			{
				write(serial_fd_, &cmd, 1);
			}
			usleep(6000);
		}
	}

	bool detectOwnColor(const cv::Mat &frame) // 识别敌我颜色
	{
		cv::Mat frame_hsv;
		cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
		cv::Vec3b pixel = frame_hsv.at<cv::Vec3b>(610, 576);
		int h = pixel[0], s = pixel[1], v = pixel[2];
		RCLCPP_INFO(this->get_logger(), "Own color HSV: (%d, %d, %d)", h, s, v);
		if (h == 0)
		{
			RCLCPP_INFO(this->get_logger(), "ally color:RED");
			enemy_h_ = 107;
			return true;
		}
		else if (h == 107)
		{
			RCLCPP_INFO(this->get_logger(), "ally color:BLUE");
			enemy_h_ = 0;
			return true;
		}
		return false;
	}

	cv::Mat GetMask(const cv::Mat &hsv) // 生成敌人黑白掩码
	{
		cv::Mat mask;
		cv::inRange(hsv, cv::Scalar(enemy_h_, 30, 30), cv::Scalar(enemy_h_, 255, 255), mask);
		return mask;
	}

	int findClosestMatch(const std::vector<cv::Rect> &candidates, const cv::Rect &target, double threshold) // 调用的时候已经排序好了
	{
		if (candidates.empty()) // 没有敌人
			return -1;
		cv::Point2f target_center(target.x + target.width / 2.0, target.y + target.height / 2.0); // 上一个中心
		int best_idx = -1;
		double min_dist = threshold;
		for (int i = 0; i < 50; i++)
		{
			cv::Point2f center(candidates[i].x + candidates[i].width / 2.0, candidates[i].y + candidates[i].height / 2.0); // 候选中心
			double dist = cv::norm(center - target_center);																   // 直线距离
			if (dist < min_dist)
			{
				min_dist = dist;
				best_idx = i;
			}
		}
		return best_idx;
	}

	void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
	{
		// rqt获取参数
		stop_fire_distance_ = this->get_parameter("stop_fire_distance").as_double();
		lock_duration_ = this->get_parameter("lock_duration").as_double();
		match_distance_threshold_ = this->get_parameter("match_distance_threshold").as_double();
		kf_q_ = this->get_parameter("kf_q").as_double();
		kf_r_ = this->get_parameter("kf_r").as_double();

		// 转换到opencv
		cv::Mat frame;
		cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
		frame = cv_ptr->image;

		cv::Point2f own(frame.cols / 2.0, frame.rows - 40.0); // 炮台坐标
		if (!catch_own_color_)
		{
			if (detectOwnColor(frame))
				catch_own_color_ = true;
			else
				return;
		}

		// 得到敌方掩码，储存在enemy_mask中
		cv::Mat hsv;
		cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
		cv::Mat enemy_mask = GetMask(hsv);

		// 敌方候选目标
		std::vector<std::vector<cv::Point>> enemy_contours;
		cv::findContours(enemy_mask, enemy_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // 取最外层轮廓
		std::vector<cv::Rect> enemy_candidates;
		for (int i = 0; i < enemy_contours.size(); i++)
		{
			double area = cv::contourArea(enemy_contours[i]); // 面积
			cv::Rect r = cv::boundingRect(enemy_contours[i]); // 外接矩形
			if (area > 30)
				enemy_candidates.push_back(r);
		}

		if (enemy_candidates.empty())
		{
			have_current_target_ = false;
			have_prev_ = false;
			has_last_target_ = false;
			target_present_ = false;
			kf_initialized_ = false;
			return;
		}
		else
		{
			target_present_ = true;
		}

		// 识别友军
		int ALLY_H;
		if (enemy_h_ == 0)
			ALLY_H = 107;
		else
			ALLY_H = 0;

		cv::Mat friend_mask;
		cv::inRange(hsv, cv::Scalar(ALLY_H, 30, 30), cv::Scalar(ALLY_H, 255, 255), friend_mask);
		std::vector<std::vector<cv::Point>> friend_contours;
		cv::findContours(friend_mask, friend_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		int ally_count = 0;
		for (int i = 0; i < friend_contours.size(); i++)
		{
			// 排除
			double area = cv::contourArea(friend_contours[i]);
			if (area < 10)
				continue;
			cv::Rect r = cv::boundingRect(friend_contours[i]);
			if (r.y + r.height / 2 > frame.rows - 80)
				continue;
			ally_count++;
			RCLCPP_INFO(this->get_logger(), "ally count:%d;", ally_count);
			cv::Point2f center(r.x + r.width / 2.0, r.y + r.height / 2.0);
			float dist = cv::norm(center - own);
			RCLCPP_INFO(this->get_logger(), "Friend dist:%.2f;", dist);
			if (dist < stop_fire_distance_ || ally_count > 5) // 防误伤
			{
				RCLCPP_INFO(this->get_logger(), "stop fire!");
				target_present_ = false;
				break;
			}
		}

		// lambda，排序，就近原则
		std::sort(enemy_candidates.begin(), enemy_candidates.end(), [&own](const cv::Rect &a, const cv::Rect &b)
				  {
			cv::Point2f ca(a.x + a.width / 2.0, a.y + a.height / 2.0);
			cv::Point2f cb(b.x + b.width / 2.0, b.y + b.height / 2.0);
			float da = (ca.x - own.x) * (ca.x - own.x) + (ca.y - own.y) * (ca.y - own.y);
			float db = (cb.x - own.x) * (cb.x - own.x) + (cb.y - own.y) * (cb.y - own.y);
			return da < db; });

		cv::Rect selected_rect;
		rclcpp::Time now = this->now();
		if (have_current_target_)
		{
			int match_idx = findClosestMatch(enemy_candidates, current_target_rect_, match_distance_threshold_);
			if (match_idx >= 0)
			{
				double delta_time = (now - lock_start_time_).seconds();
				if (delta_time >= lock_duration_) // 避免长时间锁定同意目标
				{
					int next_idx = (match_idx + 1) % enemy_candidates.size();
					selected_rect = enemy_candidates[next_idx];
					lock_start_time_ = now;
					current_target_rect_ = selected_rect;
				}
				else
				{
					selected_rect = enemy_candidates[match_idx];
					current_target_rect_ = selected_rect;
				}
			}
			else
			{
				selected_rect = enemy_candidates[0];
				lock_start_time_ = now;
				current_target_rect_ = selected_rect;
			}
		}
		else
		{
			selected_rect = enemy_candidates[0];
			lock_start_time_ = now;
			current_target_rect_ = selected_rect;
		}
		have_current_target_ = true;

		cv::Point2f target_pos(selected_rect.x + selected_rect.width / 2.0, selected_rect.y + selected_rect.height / 2.0);

		double dt = 0.0;
		if (have_prev_)
		{
			dt = (rclcpp::Time(msg->header.stamp) - prev_time_).seconds();
		}
		if (dt <= 0.001)
			dt = 0.01; // 保护

		if (!kf_initialized_)
		{
			kf_x_.x = target_pos.x;
			kf_x_.vx = 0.0;
			kf_initialized_ = true;
		}
		else
		{
			kf_x_.predict(dt);
			kf_x_.update(target_pos.x);
		}

		// 滤波后的位置与速度（仅 x 方向）
		cv::Point2f filtered_pos(kf_x_.x, target_pos.y);
		cv::Point2f filtered_vel(kf_x_.vx, 0.0f);

		// 提前量预测
		cv::Point2f future_pos = filtered_pos;
		if (have_prev_)
		{
			float dx = filtered_pos.x - own.x, dy = filtered_pos.y - own.y;
			float dist = sqrt(dx * dx + dy * dy);
			float BulletTime = dist / 600.0f;
			future_pos.x = filtered_pos.x + filtered_vel.x * BulletTime * 0.7; // 0.7修正
		}

		prev_target_pos_ = filtered_pos;
		prev_time_ = rclcpp::Time(msg->header.stamp);
		have_prev_ = true;

		float dx = future_pos.x - own.x;
		float dy = (frame.rows - future_pos.y) - (frame.rows - own.y);
		float angle = atan2(dy, dx) * 180.0 / CV_PI;

		// 低通滤波
		float alpha = 0.25f;
		if (has_last_target_)
		{
			filtered_angle_ = alpha * angle + (1 - alpha) * filtered_angle_; // 加权上一帧
		}
		else
		{
			filtered_angle_ = angle;
			has_last_target_ = true;
		}
		TurnToAngle(filtered_angle_);
	}
};

int main(int argc, char **argv)
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<ShooterNode>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}