#!/usr/bin/env python
"""
Lane Detection Node

이 노드는 카메라 이미지와 lane.csv 파일로부터 차선을 검출합니다.

입력:
- /sensors/camera/image_raw: 카메라 이미지
- /vehicle/state: 차량 상태

출력:
- /perception/lanes: 검출된 차선 정보
"""

import os
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dcas_msgs.msg import VehicleState, LaneArray, Lane
from geometry_msgs.msg import Point
import rospkg


class LaneDetectionNode:
    """차선 검출 노드

    카메라 이미지와 ground truth lane 데이터를 사용하여 차선을 검출합니다.
    """

    def __init__(self):
        """노드 초기화"""

        # ROS 파라미터
        self.detection_mode = rospy.get_param("~detection_mode", "image")  # "image" or "csv"
        self.detection_range = rospy.get_param("~detection_range", 50.0)  # 검출 범위 (m)
        self.target_center_lane = rospy.get_param("~target_center_lane", 1)  # 1: lane 0-1 사이, 2: lane 1-2 사이

        # CvBridge
        self.bridge = CvBridge()

        # 상태 변수
        self.current_image = None
        self.vehicle_state = None
        self.ground_truth_lanes = None

        # Ground truth lane 데이터 로드
        self._load_lane_csv()

        # Publisher
        self.lane_pub = rospy.Publisher(
            "/perception/lanes",
            LaneArray,
            queue_size=10
        )

        # Subscribers
        rospy.Subscriber(
            "/sensors/camera/image_raw",
            Image,
            self.callback_image
        )
        rospy.Subscriber(
            "/vehicle/state",
            VehicleState,
            self.callback_vehicle_state
        )
        rospy.Subscriber(
            "/env/lanes",
            LaneArray,
            self.callback_ground_truth_lanes
        )

        # 타이머 (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.callback_timer)

        rospy.loginfo("Lane detection node initialized")

    def _load_lane_csv(self):
        """lane.csv 파일 로드 및 중앙 차선 계산"""
        try:
            # ROS 패키지 경로 찾기
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('dcas_perception_sim')
            csv_path = os.path.join(pkg_path, 'maps', 'lanes.csv')

            if not os.path.exists(csv_path):
                rospy.logwarn(f"Lane CSV file not found: {csv_path}")
                self.lane_csv_data = {}
                self.center_lanes = {}
                return

            # CSV 파일 읽기
            import csv
            self.lane_csv_data = {}

            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lane_id = int(row['lane_id'])
                    if lane_id not in self.lane_csv_data:
                        self.lane_csv_data[lane_id] = {
                            'points': [],
                            'lane_type': int(row['lane_type'])
                        }

                    point = {
                        'x': float(row['x']),
                        'y': float(row['y']),
                        'z': float(row['z'])
                    }
                    self.lane_csv_data[lane_id]['points'].append(point)

            rospy.loginfo(f"Loaded {len(self.lane_csv_data)} lanes from CSV")

            # 중앙 차선들 계산
            self._compute_center_lanes()

        except Exception as e:
            rospy.logerr(f"Failed to load lane CSV: {e}")
            self.lane_csv_data = {}
            self.center_lanes = {}

    def _compute_center_lanes(self):
        """차선 중앙선들 계산

        3개의 차선(lane_id 0, 1, 2)이 있을 때:
        - center_lane 1: lane 0과 1의 중점
        - center_lane 2: lane 1과 2의 중점
        """
        self.center_lanes = {}

        # lane_id 0, 1, 2가 모두 있는지 확인
        if 0 not in self.lane_csv_data or 1 not in self.lane_csv_data or 2 not in self.lane_csv_data:
            rospy.logwarn("Lane ID 0, 1, or 2 not found. Cannot compute center lanes.")
            return

        lane0_points = self.lane_csv_data[0]['points']
        lane1_points = self.lane_csv_data[1]['points']
        lane2_points = self.lane_csv_data[2]['points']

        # Center lane 1: lane 0과 1 사이
        self.center_lanes[1] = []
        min_len_01 = min(len(lane0_points), len(lane1_points))
        for i in range(min_len_01):
            p0 = lane0_points[i]
            p1 = lane1_points[i]
            center_point = {
                'x': (p0['x'] + p1['x']) / 2.0,
                'y': (p0['y'] + p1['y']) / 2.0,
                'z': (p0['z'] + p1['z']) / 2.0
            }
            self.center_lanes[1].append(center_point)

        # Center lane 2: lane 1과 2 사이
        self.center_lanes[2] = []
        min_len_12 = min(len(lane1_points), len(lane2_points))
        for i in range(min_len_12):
            p1 = lane1_points[i]
            p2 = lane2_points[i]
            center_point = {
                'x': (p1['x'] + p2['x']) / 2.0,
                'y': (p1['y'] + p2['y']) / 2.0,
                'z': (p1['z'] + p2['z']) / 2.0
            }
            self.center_lanes[2].append(center_point)

        rospy.loginfo(
            f"Computed center lanes: "
            f"center_lane_1 ({len(self.center_lanes[1])} points), "
            f"center_lane_2 ({len(self.center_lanes[2])} points)"
        )

    def callback_image(self, msg):
        """카메라 이미지 콜백"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def callback_vehicle_state(self, msg):
        """차량 상태 콜백"""
        self.vehicle_state = msg

    def callback_ground_truth_lanes(self, msg):
        """Ground truth 차선 콜백"""
        self.ground_truth_lanes = msg

    def detect_lanes_from_csv(self):
        """CSV 데이터로부터 차선 검출

        미리 계산된 중앙 차선(center lane)을 사용하여
        안정적인 차선 추종이 가능하도록 합니다.
        파라미터로 지정된 center lane(1 또는 2)을 발행합니다.
        """
        if self.vehicle_state is None or not self.center_lanes:
            return None

        # 선택된 중앙 차선 확인
        if self.target_center_lane not in self.center_lanes:
            rospy.logwarn_throttle(
                5.0,
                f"Target center lane {self.target_center_lane} not found. "
                f"Available: {list(self.center_lanes.keys())}"
            )
            return None

        vehicle_x = self.vehicle_state.pose.position.x
        vehicle_y = self.vehicle_state.pose.position.y

        # 차량의 yaw 각도 계산
        quat = self.vehicle_state.pose.orientation
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        vehicle_yaw = np.arctan2(siny_cosp, cosy_cosp)

        detected_lanes = LaneArray()
        detected_lanes.header.stamp = rospy.Time.now()
        detected_lanes.header.frame_id = "map"

        # 선택된 중앙 차선을 단일 Lane으로 발행
        lane_msg = Lane()
        lane_msg.id = 900 + self.target_center_lane  # 901 또는 902
        lane_msg.lane_type = 1  # solid line
        lane_msg.header.stamp = rospy.Time.now()
        lane_msg.header.frame_id = "map"

        # 차량 전방의 중앙 차선 포인트만 선택
        selected_center_lane = self.center_lanes[self.target_center_lane]
        for point_data in selected_center_lane:
            dx = point_data['x'] - vehicle_x
            dy = point_data['y'] - vehicle_y
            distance = np.sqrt(dx*dx + dy*dy)

            # 차량 좌표계로 변환
            local_x = dx * np.cos(vehicle_yaw) + dy * np.sin(vehicle_yaw)

            # 전방 포인트만 포함 (약간의 후방 포함)
            if distance < self.detection_range and local_x > -5.0:
                point = Point()
                point.x = point_data['x']
                point.y = point_data['y']
                point.z = point_data['z']
                lane_msg.lane_lines.append(point)

        # 포인트가 있으면 추가
        if len(lane_msg.lane_lines) > 0:
            detected_lanes.lanes.append(lane_msg)

        return detected_lanes

    def detect_lanes_from_image(self):
        """이미지로부터 차선 검출

        이미지 처리를 통해 차선을 검출합니다.
        TODO: 실제 이미지 처리 알고리즘 구현 필요
        - 이미지 전처리 (ROI, 그레이스케일, 가우시안 블러)
        - 엣지 검출 (Canny)
        - 차선 검출 (Hough Transform 또는 색상 기반)
        - 차선 피팅 (다항식)
        - LaneArray 메시지 생성
        """
        if self.current_image is None:
            return None

        # TODO: 이미지 처리 구현 필요
        # 현재는 빈 LaneArray 반환 (구현 전까지 차선이 검출되지 않음)
        detected_lanes = LaneArray()
        detected_lanes.header.stamp = rospy.Time.now()
        detected_lanes.header.frame_id = "map"

        # 이미지 처리 로직을 여기에 구현하세요:
        # 1. self.current_image를 사용하여 이미지 전처리
        # 2. 차선 검출 알고리즘 적용
        # 3. 검출된 차선을 detected_lanes에 추가
        
        lane_msg = Lane()
        lane_msg.id = 900
        lane_msg.lane_type = 1  # solid line
        lane_msg.header.stamp = rospy.Time.now()
        lane_msg.header.frame_id = "map"

        return detected_lanes

    def callback_timer(self, event):
        """타이머 콜백 - 주기적으로 차선 검출 수행"""

        if self.vehicle_state is None:
            return

        # 검출 모드에 따라 차선 검출
        if self.detection_mode == "csv":
            detected_lanes = self.detect_lanes_from_csv()
        else:  # "image"
            detected_lanes = self.detect_lanes_from_image()

        # 검출된 차선 발행
        if detected_lanes is not None:
            self.lane_pub.publish(detected_lanes)


def main():
    """메인 함수"""
    rospy.init_node("lane_detection_node")

    try:
        node = LaneDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
