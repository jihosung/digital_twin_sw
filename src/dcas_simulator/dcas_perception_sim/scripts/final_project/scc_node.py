#!/usr/bin/env python
"""
Smart Cruise Control (SCC) Node

이 노드는 LiDAR와 Radar 센서 데이터를 사용하여
전방 차량을 추적하고 안전 거리를 유지하는 SCC를 구현합니다.

입력:
- /sensors/lidar/points: LiDAR 포인트 클라우드
- /sensors/radar/points: Radar 포인트 클라우드
- /vehicle/state: 차량 상태

출력:
- /scc/target_speed: SCC에서 계산된 목표 속도
- /scc/target_vehicle: 추적 중인 전방 차량 정보 (시각화용)
- (추가) logs/scc_metrics.csv: SCC 평가/디버깅 로그
"""

import os
import csv
import rospy
import numpy as np
import math
from sensor_msgs.msg import PointCloud2
from dcas_msgs.msg import VehicleState
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
import rospkg


class SCCNode:
    """Smart Cruise Control 노드"""

    def __init__(self):
        """노드 초기화"""

        # -------------------------
        # ROS 파라미터
        # -------------------------
        self.desired_speed = rospy.get_param("~desired_speed", 10.0)          # 순항 속도 (m/s)
        self.time_gap = rospy.get_param("~time_gap", 1.5)                     # 시간 간격 (sec)
        self.min_distance = rospy.get_param("~min_distance", 5.0)             # 최소 안전 거리 (m)
        self.detection_width = rospy.get_param("~detection_width", 3.5)       # 차선 폭 (m)
        self.detection_range = rospy.get_param("~detection_range", 50.0)      # 최대 검출 거리 (m)

        # [추가] 로깅 설정
        self.log_rate_hz = rospy.get_param("~log_rate_hz", 20.0)              # 로깅 주기(기본 20Hz=timer와 동일)
        self.log_flush_every = int(rospy.get_param("~log_flush_every", 20))   # N행마다 파일 flush
        self.log_csv_path = rospy.get_param("~log_csv_path", "")              # 비어있으면 pkg/logs/scc_metrics.csv

        # 타이머 dt (현재 코드 0.05s 고정)
        self.dt = 0.05

        # -------------------------
        # 상태 변수
        # -------------------------
        self.lidar_points = None
        self.radar_points = None
        self.vehicle_state = None
        self.target_vehicle = None  # dict: {x,y,distance,relative_velocity,matched}

        # [추가] 마지막 계산값(로깅용)
        self.last_safe_distance = float('nan')
        self.last_actual_dist = float('nan')
        self.last_rel_velocity = float('nan')
        self.last_lead_speed = float('nan')
        self.last_distance_error = float('nan')
        self.last_target_speed = float('nan')
        self.prev_target_speed = float('nan')
        self.last_lead_exists = 0

        # [추가] CSV 버퍼
        self._pending_rows = []

        # -------------------------
        # Publisher
        # -------------------------
        self.target_speed_pub = rospy.Publisher("/scc/target_speed", Float32, queue_size=10)
        self.target_vehicle_pub = rospy.Publisher("/scc/target_vehicle", Marker, queue_size=10)
        self.roi_pub = rospy.Publisher("/scc/roi_debug", PointCloud2, queue_size=1)

        # -------------------------
        # Subscribers
        # -------------------------
        rospy.Subscriber("/sensors/lidar/points", PointCloud2, self.callback_lidar)
        rospy.Subscriber("/sensors/radar/points", PointCloud2, self.callback_radar)
        rospy.Subscriber("/vehicle/state", VehicleState, self.callback_vehicle_state)

        # -------------------------
        # 로깅 CSV 초기화
        # -------------------------
        self._init_log_path()

        # -------------------------
        # 타이머 (20Hz)
        # -------------------------
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.callback_timer)
        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo("SCC node initialized")

    # =========================================================================
    # CSV Logging
    # =========================================================================
    def _init_log_path(self):
        """log_csv_path가 비어있으면 dcas_perception_sim/logs/scc_metrics.csv 사용"""
        try:
            if self.log_csv_path and len(self.log_csv_path) > 0:
                out_path = self.log_csv_path
            else:
                rospack = rospkg.RosPack()
                pkg_path = rospack.get_path('dcas_perception_sim')
                logs_dir = os.path.join(pkg_path, 'logs')
                out_path = os.path.join(logs_dir, 'scc_metrics.csv')

            out_dir = os.path.dirname(out_path)
            if out_dir and (not os.path.exists(out_dir)):
                os.makedirs(out_dir, exist_ok=True)

            self.log_csv_path = out_path

            if not os.path.exists(self.log_csv_path):
                self._write_log_header()

            rospy.loginfo(f"[SCC] Logging to: {self.log_csv_path}")

        except Exception as e:
            rospy.logerr(f"[SCC] Failed to init log path: {e}")
            home_ros = os.path.join(os.path.expanduser("~"), ".ros")
            os.makedirs(home_ros, exist_ok=True)
            self.log_csv_path = os.path.join(home_ros, "scc_metrics.csv")
            if not os.path.exists(self.log_csv_path):
                self._write_log_header()

    def _write_log_header(self):
        header = [
            "t_sec",
            # lead existence / geometry
            "lead_exists",
            "actual_dist_m",
            "safe_distance_m",
            "distance_error_m",
            "rel_velocity_mps",
            "lead_speed_est_mps",
            # SCC command
            "target_speed_cmd_mps",
            "target_speed_cmd_rate_mps2",
            # 추천 평가 지표
            "time_headway_s",         # dist / v_ego
            "time_gap_error_s",       # time_headway - time_gap
            "ttc_s",                  # dist / (-rel_vel) when closing
            # ego state (가능한 것들)
            "ego_x_m",
            "ego_y_m",
            "ego_v_mps",
            "ego_steer_deg",
        ]
        try:
            with open(self.log_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        except Exception as e:
            rospy.logerr(f"[SCC] Failed to write log header: {e}")

    def _append_log_row(self):
        """현재 상태를 한 줄로 버퍼에 추가"""
        t_sec = rospy.Time.now().to_sec()

        # ego state 안전 추출
        ego_x = float('nan')
        ego_y = float('nan')
        ego_v = float('nan')
        ego_steer = float('nan')

        if self.vehicle_state is not None:
            try:
                ego_x = float(self.vehicle_state.pose.position.x)
                ego_y = float(self.vehicle_state.pose.position.y)
            except Exception:
                pass
            try:
                ego_v = float(self.vehicle_state.twist.linear.x)
            except Exception:
                pass
            try:
                ego_steer = float(self.vehicle_state.steering_wheel_angle_deg)
            except Exception:
                pass

        # 추천 평가 지표 계산
        # Time headway
        if np.isfinite(self.last_actual_dist) and np.isfinite(ego_v) and ego_v > 0.2:
            th = float(self.last_actual_dist / ego_v)
        else:
            th = float('nan')

        # time gap error
        tgap_err = float(th - self.time_gap) if np.isfinite(th) else float('nan')

        # TTC (closing일 때만)
        if np.isfinite(self.last_actual_dist) and np.isfinite(self.last_rel_velocity) and self.last_rel_velocity < -0.1:
            ttc = float(self.last_actual_dist / (-self.last_rel_velocity))
        else:
            ttc = float('inf')  # 닫히는 중이 아니면 inf로 표기(원하면 nan으로 바꿔도 됨)

        # command rate (smoothness proxy)
        if np.isfinite(self.prev_target_speed) and np.isfinite(self.last_target_speed) and self.dt > 1e-6:
            cmd_rate = float((self.last_target_speed - self.prev_target_speed) / self.dt)
        else:
            cmd_rate = float('nan')

        row = [
            float(t_sec),
            int(self.last_lead_exists),
            float(self.last_actual_dist),
            float(self.last_safe_distance),
            float(self.last_distance_error),
            float(self.last_rel_velocity),
            float(self.last_lead_speed),
            float(self.last_target_speed),
            float(cmd_rate),
            float(th),
            float(tgap_err),
            float(ttc),
            float(ego_x),
            float(ego_y),
            float(ego_v),
            float(ego_steer),
        ]

        self._pending_rows.append(row)

        if len(self._pending_rows) >= max(1, self.log_flush_every):
            self._flush_logs()

    def _flush_logs(self):
        if len(self._pending_rows) == 0:
            return
        try:
            with open(self.log_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self._pending_rows)
            self._pending_rows = []
        except Exception as e:
            rospy.logerr(f"[SCC] Failed to flush logs: {e}")

    def on_shutdown(self):
        self._flush_logs()
        rospy.loginfo("[SCC] Shutdown: logs flushed")

    # =========================================================================
    # Callbacks
    # =========================================================================
    def callback_lidar(self, msg):
        self.lidar_points = self.parse_pointcloud2(msg)

    def callback_radar(self, msg):
        self.radar_points = self.parse_pointcloud2(msg)

    def callback_vehicle_state(self, msg):
        self.vehicle_state = msg

    # =========================================================================
    # PointCloud parsing
    # =========================================================================
    def parse_pointcloud2(self, cloud_msg):
        if cloud_msg is None:
            return None

        available_fields = [f.name for f in cloud_msg.fields]
        rospy.loginfo_throttle(5.0, f"[SCC] Incoming Fields: {available_fields}")

        fields_to_read = ['x', 'y', 'z']

        if 'doppler' in available_fields:
            fields_to_read.append('doppler')
        elif 'velocity' in available_fields:
            fields_to_read.append('velocity')
        elif 'intensity' in available_fields:
            fields_to_read.append('intensity')

        rospy.logdebug_throttle(5.0, f"[SCC] Selected Fields to read: {fields_to_read}")

        try:
            point_generator = pc2.read_points(cloud_msg, field_names=fields_to_read, skip_nans=True)
            points_list = list(point_generator)
        except Exception as e:
            rospy.logerr(f"[SCC] Error parsing PointCloud: {e}")
            return None

        if not points_list:
            rospy.logwarn_throttle(2.0, "[SCC] Parsed PointCloud is empty!")
            return np.empty((0, len(fields_to_read)), dtype=np.float32)

        points_np = np.array(points_list, dtype=np.float32)
        rospy.loginfo_throttle(1.0, f"[SCC] Parsed Shape: {points_np.shape}")

        return points_np

    # =========================================================================
    # Detection
    # =========================================================================
    def detect_leading_vehicle(self):
        """전방 선행 차량 검출"""
        if self.lidar_points is None:
            return None
        if self.vehicle_state is None:
            return None

        total_points = len(self.lidar_points)
        rospy.logdebug_throttle(1.0, f"[SCC] Total LiDAR points: {total_points}")

        # --- ROI 필터링 (조향 반영) ---
        STEERING_RATIO = 1  # steering_wheel_angle_deg는 이미 바퀴각이라고 가정
        wheel_angle_rad = np.radians(self.vehicle_state.steering_wheel_angle_deg / STEERING_RATIO)

        theta = -wheel_angle_rad * 1.4
        c, s = np.cos(theta), np.sin(theta)

        x_raw = self.lidar_points[:, 0]
        y_raw = self.lidar_points[:, 1]

        x_rot = x_raw * c - y_raw * s
        y_rot = x_raw * s + y_raw * c

        half_width = self.detection_width / 2.0
        mask = (x_rot > 0.5) & (x_rot < self.detection_range) & (np.abs(y_rot) < half_width)

        roi_points = self.lidar_points[mask]

        # ROI debug publish
        if len(roi_points) > 0:
            header = rospy.Header()
            header.frame_id = "lidar"
            header.stamp = rospy.Time.now()
            roi_cloud = pc2.create_cloud_xyz32(header, roi_points[:, :3])
            self.roi_pub.publish(roi_cloud)

        if len(roi_points) < 3:
            rospy.logdebug_throttle(1.0, f"[SCC] ROI filtered out all points. Left: {len(roi_points)}/{total_points}")
            return None

        rospy.logdebug_throttle(1.0, f"[SCC] Points in ROI: {len(roi_points)}")

        # --- DBSCAN clustering ---
        try:
            model = DBSCAN(eps=1.0, min_samples=3)
            labels = model.fit_predict(roi_points[:, :3])
        except Exception as e:
            rospy.logerr(f"[SCC] Clustering error: {e}")
            return None

        detected_objects = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                continue

            cluster_points = roi_points[labels == label]

            cx = float(np.mean(cluster_points[:, 0]))
            cy = float(np.mean(cluster_points[:, 1]))
            distance = float(np.sqrt(cx**2 + cy**2))

            rel_velocity = 0.0
            is_matched = False

            if self.radar_points is not None and len(self.radar_points) > 0:
                radar_pos = self.radar_points[:, :2]
                dists = np.sqrt(np.sum((radar_pos - np.array([cx, cy], dtype=np.float32))**2, axis=1))
                min_idx = int(np.argmin(dists))
                min_dist = float(dists[min_idx])

                if min_dist < 2.5:
                    if self.radar_points.shape[1] >= 4:
                        rel_velocity = float(self.radar_points[min_idx, 3])
                        is_matched = True

            if is_matched:
                detected_objects.append({
                    'x': cx,
                    'y': cy,
                    'distance': distance,
                    'relative_velocity': rel_velocity,
                    'matched': is_matched
                })

        if not detected_objects:
            rospy.logdebug_throttle(1.0, "[SCC] Clusters found but ignored (noise?)")
            return None

        detected_objects.sort(key=lambda obj: obj['distance'])
        target = detected_objects[0]

        rospy.loginfo_throttle(0.5,
            f"[SCC] TARGET: Dist={target['distance']:.1f}m, "
            f"Vel={target['relative_velocity']:.1f}m/s, "
            f"Radar={'O' if target['matched'] else 'X'}"
        )
        return target

    # =========================================================================
    # Control
    # =========================================================================
    def calculate_safe_distance(self, ego_velocity):
        safe_distance = ego_velocity * self.time_gap
        return max(safe_distance, self.min_distance)

    def calculate_target_speed(self):
        """SCC 목표 속도 계산 + (추가) 로깅용 내부 변수 업데이트"""

        if self.vehicle_state is None:
            # 로깅 상태 업데이트 (lead 없음)
            self.last_lead_exists = 0
            self.last_safe_distance = float('nan')
            self.last_actual_dist = float('nan')
            self.last_rel_velocity = float('nan')
            self.last_lead_speed = float('nan')
            self.last_distance_error = float('nan')
            return self.desired_speed

        ego_velocity = float(self.vehicle_state.twist.linear.x)

        # 1) 선행차량 탐지
        target = self.detect_leading_vehicle()
        self.target_vehicle = target

        if target is None:
            # lead 없음 상태 로깅
            self.last_lead_exists = 0
            self.last_safe_distance = self.calculate_safe_distance(ego_velocity)  # 그래도 참고용으로 남김
            self.last_actual_dist = float('nan')
            self.last_rel_velocity = float('nan')
            self.last_lead_speed = float('nan')
            self.last_distance_error = float('nan')
            return self.desired_speed

        # lead 있음
        self.last_lead_exists = 1

        dist_actual = float(target['distance'])
        rel_velocity = float(target['relative_velocity'])
        lead_vehicle_speed = float(ego_velocity + rel_velocity)

        safe_distance = float(self.min_distance + (ego_velocity * self.time_gap))
        distance_error = float(dist_actual - safe_distance)

        k_p = 0.7
        calculated_speed = float(lead_vehicle_speed + (k_p * distance_error))

        target_speed = min(calculated_speed, self.desired_speed)
        target_speed = max(target_speed, 0.0)

        if dist_actual < 2.0:
            target_speed = 0.0

        rospy.logdebug_throttle(0.5,
            f"[SCC] Dist: {dist_actual:.1f}m (Safe: {safe_distance:.1f}m) | "
            f"Err: {distance_error:.1f} | "
            f"LeadSpd: {lead_vehicle_speed:.1f} | "
            f"Cmd: {target_speed:.1f}"
        )

        # [추가] 로깅용 상태 업데이트
        self.last_safe_distance = safe_distance
        self.last_actual_dist = dist_actual
        self.last_rel_velocity = rel_velocity
        self.last_lead_speed = lead_vehicle_speed
        self.last_distance_error = distance_error

        return float(target_speed)

    # =========================================================================
    # Visualization
    # =========================================================================
    def visualize_target_vehicle(self):
        if self.target_vehicle is None:
            return

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "scc_target"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = self.target_vehicle['x']
        marker.pose.position.y = self.target_vehicle['y']
        marker.pose.position.z = 1.0

        marker.scale.x = 4.5
        marker.scale.y = 2.0
        marker.scale.z = 1.5

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.7

        self.target_vehicle_pub.publish(marker)

    # =========================================================================
    # Timer
    # =========================================================================
    def callback_timer(self, event):
        if self.vehicle_state is None:
            return

        # 이전 cmd 저장(명령 변화율 계산용)
        self.prev_target_speed = self.last_target_speed

        # 목표 속도 계산
        target_speed = self.calculate_target_speed()
        self.last_target_speed = float(target_speed)

        # publish
        self.target_speed_pub.publish(Float32(target_speed))
        self.visualize_target_vehicle()

        # CSV 로깅 (timer와 같은 주기에서 기록)
        self._append_log_row()


def main():
    rospy.init_node("scc_node")
    try:
        node = SCCNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
