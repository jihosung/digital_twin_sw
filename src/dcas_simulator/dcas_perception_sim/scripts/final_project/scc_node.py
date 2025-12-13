#!/usr/bin/env python
"""
Smart Cruise Control (SCC) Node

이 노드는 LiDAR와 Radar 센서 데이터를 사용하여
전방 차량을 추적하고 안전 거리를 유지하는 SCC를 구현합니다.

입력:
- /sensors/lidar/points_noise: LiDAR 포인트 클라우드
- /sensors/radar/points_noise: Radar 포인트 클라우드
- /vehicle/state: 차량 상태

출력:
- /scc/target_speed: SCC에서 계산된 목표 속도
- /scc/target_vehicle: 추적 중인 전방 차량 정보 (시각화용)
"""

import rospy
import numpy as np
import math
import struct
from sensor_msgs.msg import PointCloud2
from dcas_msgs.msg import VehicleState
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN


class SCCNode:
    """Smart Cruise Control 노드

    LiDAR와 Radar 데이터를 융합하여 전방 차량을 검출하고
    안전 거리를 유지하며 속도를 제어합니다.
    """

    def __init__(self):
        """노드 초기화"""

        # ROS 파라미터
        self.desired_speed = rospy.get_param("~desired_speed", 10.0)  # 순항 속도 (m/s)
        self.time_gap = rospy.get_param("~time_gap", 1.5)  # 시간 간격 (초)
        self.min_distance = rospy.get_param("~min_distance", 5.0)  # 최소 안전 거리 (m)
        self.detection_width = rospy.get_param("~detection_width", 3.5)  # 차선 폭 (m) default: 2.0
        self.detection_range = rospy.get_param("~detection_range", 50.0)  # 최대 검출 거리 (m)

        # 상태 변수
        self.lidar_points = None
        self.radar_points = None
        self.vehicle_state = None
        self.target_vehicle = None  # (x, y, distance, relative_velocity)

        # Publisher
        self.target_speed_pub = rospy.Publisher(
            "/scc/target_speed",
            Float32,
            queue_size=10
        )
        self.target_vehicle_pub = rospy.Publisher(
            "/scc/target_vehicle",
            Marker,
            queue_size=10
        )
        # [추가됨] ROI 필터링된 점들을 Rviz에서 보기 위한 디버깅용 Publisher
        self.roi_pub = rospy.Publisher("/scc/roi_debug", PointCloud2, queue_size=1)

        # Subscribers
        rospy.Subscriber(
            "/sensors/lidar/points",
            PointCloud2,
            self.callback_lidar
        )
        rospy.Subscriber(
            "/sensors/radar/points",
            PointCloud2,
            self.callback_radar
        )
        rospy.Subscriber(
            "/vehicle/state",
            VehicleState,
            self.callback_vehicle_state
        )

        # 타이머 (20Hz)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.callback_timer)

        rospy.loginfo("SCC node initialized")

    def callback_lidar(self, msg):
        """LiDAR 포인트 클라우드 콜백"""
        self.lidar_points = self.parse_pointcloud2(msg)

    def callback_radar(self, msg):
        """Radar 포인트 클라우드 콜백"""
        self.radar_points = self.parse_pointcloud2(msg)

    def callback_vehicle_state(self, msg):
        """차량 상태 콜백"""
        self.vehicle_state = msg

    def parse_pointcloud2(self, cloud_msg):
        """PointCloud2 메시지를 numpy 배열로 변환"""
        if cloud_msg is None:
            return None

        available_fields = [f.name for f in cloud_msg.fields]
        
        # [로그] 현재 들어온 필드 (5초 주기)
        rospy.loginfo_throttle(5.0, f"[SCC] Incoming Fields: {available_fields}")

        fields_to_read = ['x', 'y', 'z']
        
        # Radar/LiDAR 필드 선택 로직
        if 'doppler' in available_fields:
            fields_to_read.append('doppler')
        elif 'velocity' in available_fields:
            fields_to_read.append('velocity')
        elif 'intensity' in available_fields:
            fields_to_read.append('intensity')
        
        # [로그] 실제 읽으려는 필드 확인 (디버깅용)
        # 만약 이 로그가 안 뜨면 위쪽 코드에서 멈춘 것임
        rospy.logdebug_throttle(5.0, f"[SCC] Selected Fields to read: {fields_to_read}")

        try:
            point_generator = pc2.read_points(cloud_msg, field_names=fields_to_read, skip_nans=True)
            points_list = list(point_generator)
        except Exception as e:
            rospy.logerr(f"[SCC] Error parsing PointCloud: {e}")
            return None

        if not points_list:
            # 포인트가 0개면 여기서 경고가 떠야 함
            rospy.logwarn_throttle(2.0, "[SCC] Parsed PointCloud is empty!")
            return np.empty((0, len(fields_to_read)), dtype=np.float32)

        points_np = np.array(points_list, dtype=np.float32)

        # [로그] 파싱 성공 여부 (debug 레벨 -> info 레벨로 잠시 격상해서 확인)
        rospy.loginfo_throttle(1.0, f"[SCC] Parsed Shape: {points_np.shape}")

        return points_np

    def detect_leading_vehicle(self):
        """전방 선행 차량 검출 (디버깅 강화 및 파라미터 완화)"""
        
        # 1. 데이터 존재 여부 확인
        if self.lidar_points is None:
            return None
            
        # [로그] 전체 포인트 개수 확인 (너무 자주 뜨지 않게 throttle)
        total_points = len(self.lidar_points)
        rospy.logdebug_throttle(1.0, f"[SCC] Total LiDAR points: {total_points}")

        # --- 2. ROI (Region of Interest) 필터링 ---
        # 조건: 전방(x > 0), 최대 거리 이내, 차선 폭 이내
        # x축: 전방, y축: 좌우
        
        # 팁: detection_width가 너무 좁으면 차량의 측면이 잘릴 수 있음.
        # 디버깅을 위해 width를 살짝 여유 있게 잡는 것도 방법.
        # half_width = self.detection_width / 2.0
        
        # mask = (self.lidar_points[:, 0] > 0.5) & \
        #        (self.lidar_points[:, 0] < self.detection_range) & \
        #        (np.abs(self.lidar_points[:, 1]) < half_width)
        
        # roi_points = self.lidar_points[mask]

        # --- 2. ROI (Region of Interest) 필터링 (회전 반영) ---
        # [설정] 조향비 (Steering Ratio)
        # 일반적인 승용차는 13~16 사이입니다. 차량 제원에 맞춰 수정하세요.
        STEERING_RATIO = 1 # steering wheel angle (deg)는 이미 바퀴각임
        
        # 1. 핸들 각도를 타이어 회전각(Radian)으로 변환
        # ROS 좌표계 표준: 좌회전(+), 우회전(-) 
        # 만약 핸들 데이터 부호가 반대라면 -를 붙여주세요.
        wheel_angle_rad = np.radians(self.vehicle_state.steering_wheel_angle_deg / STEERING_RATIO)

        # 2. 회전 행렬 계산 (Points를 핸들 반대 방향으로 돌려서 정렬)
        # 우리가 원하는 건 "기울어진 박스 안의 점"을 찾는 것인데,
        # 수식적으로는 "점을 반대로 기울여서 똑바로 선 박스에 넣는 것"이 훨씬 빠릅니다.
        theta = -wheel_angle_rad * 1.2 # steering보다 20% 더 돌림
        c, s = np.cos(theta), np.sin(theta)

        # LiDAR 원본 좌표 가져오기
        x_raw = self.lidar_points[:, 0]
        y_raw = self.lidar_points[:, 1]

        # 3. 회전 변환 적용 (Rotation Matrix)
        # x' = x cos θ - y sin θ
        # y' = x sin θ + y cos θ
        x_rot = x_raw * c - y_raw * s
        y_rot = x_raw * s + y_raw * c

        # 4. 마스크 생성 (회전된 좌표계 기준으로 필터링)
        half_width = self.detection_width / 2.0
        
        mask = (x_rot > 0.5) & \
               (x_rot < self.detection_range) & \
               (np.abs(y_rot) < half_width)
        
        # [중요] 실제 roi_points에는 '원본 좌표'를 담아야 함 (회전된 좌표 아님)
        roi_points = self.lidar_points[mask]
        
        # [디버그용 Publisher] ROI 내부 점들을 Rviz로 발행
        if len(roi_points) > 0:
            header = rospy.Header()
            header.frame_id = "lidar" # 또는 sensor_frame 이름
            header.stamp = rospy.Time.now()
            # pc2.create_cloud_xyz32 함수 사용 (import 필요)
            roi_cloud = pc2.create_cloud_xyz32(header, roi_points[:, :3])
            self.roi_pub.publish(roi_cloud)

        # [중요 로그] ROI 통과 개수 확인
        # 여기서 0개가 나오면 detection_width나 range 파라미터를 확인해야 함
        if len(roi_points) < 3:
            rospy.logdebug_throttle(1.0, f"[SCC] ROI filtered out all points. Left: {len(roi_points)}/{total_points}")
            return None

        rospy.logdebug_throttle(1.0, f"[SCC] Points in ROI: {len(roi_points)}")

        # --- 3. LiDAR 클러스터링 (DBSCAN) ---
        # 파라미터 완화: 
        # eps: 0.5 -> 1.0 (점 사이 거리가 1m 이내면 같은 물체로 간주)
        # min_samples: 5 -> 3 (점이 3개만 모여도 물체로 인정)
        try:
            model = DBSCAN(eps=1.0, min_samples=3)
            labels = model.fit_predict(roi_points[:, :2])
        except Exception as e:
            rospy.logerr(f"[SCC] Clustering error: {e}")
            return None

        # --- 4. 객체 분석 및 Radar 융합 ---
        detected_objects = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1: continue # 노이즈

            cluster_mask = (labels == label)
            cluster_points = roi_points[cluster_mask]

            cx = np.mean(cluster_points[:, 0])
            cy = np.mean(cluster_points[:, 1])
            distance = np.sqrt(cx**2 + cy**2)

            # Radar 융합 (가장 가까운 레이더 포인트 찾기)
            rel_velocity = 0.0
            is_matched = False
            
            if self.radar_points is not None and len(self.radar_points) > 0:
                radar_pos = self.radar_points[:, :2]
                # 현재 클러스터 중심과 모든 레이더 포인트 간 거리 계산
                dists = np.sqrt(np.sum((radar_pos - np.array([cx, cy]))**2, axis=1))
                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]

                # 매칭 임계값 (2.5m 이내면 같은 물체로 간주)
                if min_dist < 2.5:
                    # parse_pointcloud2에서 doppler를 4번째(index 3)로 넣었음
                    if self.radar_points.shape[1] >= 4:
                        rel_velocity = self.radar_points[min_idx, 3]
                        is_matched = True

            if is_matched:
                detected_objects.append({
                    'x': cx,
                    'y': cy,
                    'distance': distance,
                    'relative_velocity': rel_velocity,
                    'matched': is_matched
                })

        # --- 5. 타겟 선정 ---
        if not detected_objects:
            # DBSCAN은 돌았는데 유의미한 클러스터가 안 나온 경우
            rospy.logdebug_throttle(1.0, "[SCC] Clusters found but ignored (noise?)")
            return None

        # 거리순 정렬
        detected_objects.sort(key=lambda obj: obj['distance'])
        target = detected_objects[0]

        # [성공 로그]
        rospy.loginfo_throttle(0.5, 
            f"[SCC] TARGET: Dist={target['distance']:.1f}m, "
            f"Vel={target['relative_velocity']:.1f}m/s, "
            f"Radar={'O' if target['matched'] else 'X'}"
        )

        return target

    def calculate_safe_distance(self, ego_velocity):
        """안전 거리 계산

        Args:
            ego_velocity: 자차 속도 (m/s)

        Returns:
            float: 안전 거리 (m)
        """
        # Time gap 기반 안전 거리 계산
        safe_distance = ego_velocity * self.time_gap

        # 최소 안전 거리 적용
        return max(safe_distance, self.min_distance)

    def calculate_target_speed(self):
        """SCC 목표 속도 계산 (Constant Time Gap 로직 적용)

        제어 전략:
        1. 선행 차량 속도 추정 (V_lead = V_ego + V_rel)
        2. 안전 거리 계산 (D_safe = D_min + V_ego * T_gap)
        3. P-Control: 거리 오차에 비례하여 속도 가감
        """
        # 차량 상태가 없으면 기본 속도 반환
        if self.vehicle_state is None:
            return self.desired_speed

        ego_velocity = self.vehicle_state.twist.linear.x

        # 1. 전방 차량 검출 및 업데이트
        target = self.detect_leading_vehicle()
        self.target_vehicle = target

        # 2. 전방 차량이 없는 경우: 설정된 순항 속도로 주행
        if target is None:
            return self.desired_speed

        # 3. 데이터 추출
        dist_actual = target['distance']          # 실제 거리
        rel_velocity = target['relative_velocity'] # 상대 속도 (음수면 가까워짐)

        # 4. 선행 차량의 절대 속도 추정
        # 내 속도 + 상대 속도 = 앞차 속도
        lead_vehicle_speed = ego_velocity + rel_velocity

        # 5. 안전 거리 계산 (Time Gap 적용)
        # 안전 거리 = 최소 거리 + (내 속도 * 시간 간격)
        # 속도가 빠를수록 더 먼 거리에서 멈추도록 함
        safe_distance = self.min_distance + (ego_velocity * self.time_gap)

        # 6. 속도 제어량 계산 (P-Control)
        # 거리 오차: (현재 거리 - 안전 거리)
        # 양수면 너무 멂(가속 필요), 음수면 너무 가까움(감속 필요)
        distance_error = dist_actual - safe_distance

        # 제어 이득(Gain): 거리 오차 1m당 속도를 얼마나 조절할지 (튜닝 파라미터)
        k_p = 0.7

        # 목표 속도 = 앞차 속도 + (거리 오차 * 이득)
        # 앞차랑 속도를 맞추되, 거리가 멀면 더 빨리, 가까우면 더 느리게
        calculated_speed = lead_vehicle_speed + (k_p * distance_error)

        # 7. 목표 속도 클램핑 (안전 제한)
        # 최대: 사용자가 설정한 순항 속도(desired_speed)를 넘지 않음
        # 최소: 0.0 (후진 금지)
        target_speed = min(calculated_speed, self.desired_speed)
        target_speed = max(target_speed, 0.0)

        # [안전] 거리가 너무 가까우면(예: 2m 이내) 강제 정지 명령
        if dist_actual < 2.0:
            target_speed = 0.0

        # [로그] 제어 상황 모니터링 (0.5초 주기)
        rospy.logdebug_throttle(0.5, 
            f"[SCC] Dist: {dist_actual:.1f}m (Safe: {safe_distance:.1f}m) | "
            f"Err: {distance_error:.1f} | "
            f"LeadSpd: {lead_vehicle_speed:.1f} | "
            f"Cmd: {target_speed:.1f}"
        )

        return target_speed

    def visualize_target_vehicle(self):
        """추적 중인 전방 차량 시각화"""
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

    def callback_timer(self, event):
        """타이머 콜백 - 주기적으로 SCC 수행"""

        if self.vehicle_state is None:
            return

        # 목표 속도 계산
        target_speed = self.calculate_target_speed()

        # 목표 속도 발행
        self.target_speed_pub.publish(Float32(target_speed))

        # 전방 차량 시각화
        self.visualize_target_vehicle()


def main():
    """메인 함수"""
    rospy.init_node("scc_node")

    try:
        node = SCCNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
