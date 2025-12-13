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
        self.detection_width = rospy.get_param("~detection_width", 2.0)  # 차선 폭 (m)
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
        """PointCloud2 메시지를 numpy 배열로 변환

        TODO: 실제 PointCloud2 파싱 구현 필요
        - cloud_msg.data를 파싱하여 (x, y, z, intensity/velocity) 추출
        - sensor_frame에서의 좌표를 반환
        """
        # TODO: PointCloud2 파싱 구현
        # 현재는 None 반환 (구현 전까지 동작하지 않음)
        return None

    def detect_leading_vehicle(self):
        """전방 선행 차량 검출

        LiDAR와 Radar 데이터를 융합하여 전방 차량을 검출합니다.

        TODO: 실제 차량 검출 알고리즘 구현 필요
        - LiDAR 포인트 클러스터링
        - Radar 타겟과 LiDAR 클러스터 매칭
        - 차량 좌표계 기준으로 전방 차선 내 차량만 선택
        - 가장 가까운 차량 선택

        Returns:
            dict or None: {'x': float, 'y': float, 'distance': float, 'relative_velocity': float}
        """
        if self.vehicle_state is None:
            return None

        # TODO: 차량 검출 알고리즘 구현
        # 1. LiDAR 포인트 클러스터링 (DBSCAN 등)
        # 2. Radar 타겟 추출
        # 3. LiDAR-Radar 융합 (거리 기반 매칭)
        # 4. 전방 차선 내 타겟 필터링
        # 5. 가장 가까운 타겟 선택

        return None

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
        """SCC 목표 속도 계산

        TODO: 실제 속도 제어 로직 구현 필요
        - 전방 차량이 없으면 desired_speed 반환
        - 전방 차량이 있으면 안전 거리 기반 속도 계산
        - PID 또는 MPC 기반 제어

        Returns:
            float: 목표 속도 (m/s)
        """
        if self.vehicle_state is None:
            return self.desired_speed

        ego_velocity = self.vehicle_state.twist.linear.x

        # 전방 차량 검출
        target = self.detect_leading_vehicle()
        self.target_vehicle = target

        if target is None:
            # 전방 차량이 없으면 순항 속도로 주행
            return self.desired_speed

        # TODO: 속도 제어 로직 구현
        # 1. 현재 거리와 안전 거리 비교
        # 2. 상대 속도 고려
        # 3. 목표 속도 계산 (PID 또는 adaptive cruise control)

        # 간단한 예시 (실제 구현 필요):
        # distance = target['distance']
        # relative_velocity = target['relative_velocity']
        # safe_distance = self.calculate_safe_distance(ego_velocity)
        #
        # if distance < safe_distance:
        #     # 거리가 너무 가까우면 감속
        #     target_speed = ego_velocity - 1.0
        # else:
        #     # 안전 거리 유지하며 순항
        #     target_speed = self.desired_speed

        return self.desired_speed

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
