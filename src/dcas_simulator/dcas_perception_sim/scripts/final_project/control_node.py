#!/usr/bin/env python
"""
Vehicle Control Node

이 노드는 검출된 차선 정보를 기반으로 차량을 제어합니다.

입력:
- /perception/lanes: 검출된 차선 정보
- /vehicle/state: 차량 상태

출력:
- /vehicle/cmd/steer: 조향 명령
- /vehicle/cmd/accel: 가속 명령
"""

import rospy
import numpy as np
import math
from dcas_msgs.msg import LaneArray, VehicleState
from std_msgs.msg import Float32, ColorRGBA
from geometry_msgs.msg import Quaternion, Point
from visualization_msgs.msg import Marker


class ControlNode:
    """차량 제어 노드

    차선 정보를 기반으로 차량을 제어합니다.
    """

    def __init__(self):
        """노드 초기화"""

        # ROS 파라미터
        self.target_speed = rospy.get_param("~target_speed", 10.0)  # 목표 속도 (m/s)
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 10.0)  # Pure Pursuit lookahead (m)
        self.steering_gain = rospy.get_param("~steering_gain", 1.0)  # 조향 게인
        self.speed_gain_p = rospy.get_param("~speed_gain_p", 0.5)  # 속도 P 게인
        self.wheelbase = rospy.get_param("~wheelbase", 2.7)  # 축간거리 (m)
        self.max_steer_angle = rospy.get_param("~max_steer_angle", 30.0)  # 최대 조향각 (deg)

        # 상태 변수
        self.detected_lanes = None
        self.vehicle_state = None

        # Publisher
        self.steer_pub = rospy.Publisher(
            "/vehicle/cmd/steer",
            Float32,
            queue_size=10
        )
        self.accel_pub = rospy.Publisher(
            "/vehicle/cmd/accel",
            Float32,
            queue_size=10
        )
        self.viz_path_pub = rospy.Publisher(
            "/viz/center_path",
            Marker,
            queue_size=10
        )
        self.viz_target_pub = rospy.Publisher(
            "/viz/target_point",
            Marker,
            queue_size=10
        )

        # Subscribers
        rospy.Subscriber(
            "/perception/lanes",
            LaneArray,
            self.callback_lanes
        )
        rospy.Subscriber(
            "/vehicle/state",
            VehicleState,
            self.callback_vehicle_state
        )
        rospy.Subscriber(
            "/scc/target_speed",
            Float32,
            self.callback_target_speed
        )

        # 타이머 (50Hz)
        self.timer = rospy.Timer(rospy.Duration(0.02), self.callback_timer)

        rospy.loginfo("Control node initialized")

    def callback_lanes(self, msg):
        """차선 정보 콜백"""
        self.detected_lanes = msg

    def callback_vehicle_state(self, msg):
        """차량 상태 콜백"""
        self.vehicle_state = msg

    def callback_target_speed(self, msg):
        """목표 속도 콜백"""
        self.target_speed = msg.data

    def quaternion_to_yaw(self, quat):
        """쿼터니언을 yaw 각도로 변환"""
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def find_center_lane(self):
        """중앙 차선 경로 찾기

        lane_detection_node에서 이미 계산된 중앙 차선을 사용하거나,
        왼쪽/오른쪽 차선이 있으면 중점을 계산합니다.
        """
        if self.detected_lanes is None or len(self.detected_lanes.lanes) == 0:
            return None

        if self.vehicle_state is None:
            return None

        vehicle_x = self.vehicle_state.pose.position.x
        vehicle_y = self.vehicle_state.pose.position.y
        vehicle_yaw = self.quaternion_to_yaw(self.vehicle_state.pose.orientation)

        # 중앙 차선이 이미 제공된 경우 (lane_id >= 900)
        for lane in self.detected_lanes.lanes:
            if lane.id >= 900 and len(lane.lane_lines) > 0:
                # 중앙 차선을 그대로 사용
                center_points = []
                for point in lane.lane_lines:
                    center_points.append((point.x, point.y))
                return center_points

        return None

    def _offset_lane(self, lane, offset_distance, vehicle_yaw):
        """차선에 오프셋을 적용하여 새로운 경로 생성"""
        offset_points = []

        for i in range(len(lane.lane_lines) - 1):
            p1 = lane.lane_lines[i]
            p2 = lane.lane_lines[i + 1]

            # 차선 방향 벡터
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            length = math.sqrt(dx*dx + dy*dy)

            if length < 0.001:
                continue

            # 수직 벡터 (왼쪽 방향)
            perp_x = -dy / length
            perp_y = dx / length

            # 오프셋 적용
            offset_x = p1.x + perp_x * offset_distance
            offset_y = p1.y + perp_y * offset_distance

            offset_points.append((offset_x, offset_y))

        return offset_points

    def pure_pursuit_control(self, path):
        """Pure Pursuit 알고리즘으로 조향각 계산"""
        if path is None or len(path) == 0:
            return 0.0, None

        if self.vehicle_state is None:
            return 0.0, None

        vehicle_x = self.vehicle_state.pose.position.x
        vehicle_y = self.vehicle_state.pose.position.y
        vehicle_yaw = self.quaternion_to_yaw(self.vehicle_state.pose.orientation)

        # Lookahead 거리 내의 목표 포인트 찾기
        target_point = None
        min_dist_diff = float('inf')

        for (px, py) in path:
            dx = px - vehicle_x
            dy = py - vehicle_y
            distance = math.sqrt(dx*dx + dy*dy)

            # Lookahead 거리와 가장 가까운 포인트 선택
            dist_diff = abs(distance - self.lookahead_distance)
            if dist_diff < min_dist_diff and distance > self.lookahead_distance * 0.5:
                min_dist_diff = dist_diff
                target_point = (px, py)

        if target_point is None:
            # 경로의 마지막 포인트 사용
            target_point = path[-1]

        # 차량 좌표계로 변환
        dx = target_point[0] - vehicle_x
        dy = target_point[1] - vehicle_y

        cos_yaw = math.cos(vehicle_yaw)
        sin_yaw = math.sin(vehicle_yaw)
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw

        # Pure Pursuit 조향각 계산
        ld = math.sqrt(local_x*local_x + local_y*local_y)
        if ld < 0.1:
            return 0.0, target_point

        # 곡률 계산
        curvature = 2.0 * local_y / (ld * ld)

        # 조향각 계산 (Ackermann 기하)
        steer_angle_rad = math.atan(curvature * self.wheelbase)
        steer_angle_deg = math.degrees(steer_angle_rad)

        # 게인 적용 및 제한
        steer_angle_deg *= self.steering_gain
        steer_angle_deg = np.clip(steer_angle_deg, -self.max_steer_angle, self.max_steer_angle)

        return steer_angle_deg, target_point

    def speed_control(self):
        """속도 제어 (간단한 P 제어)"""
        if self.vehicle_state is None:
            return 0.0

        current_speed = self.vehicle_state.twist.linear.x
        speed_error = self.target_speed - current_speed

        # P 제어
        accel_cmd = self.speed_gain_p * speed_error

        # 제한
        accel_cmd = np.clip(accel_cmd, -3.0, 3.0)

        return accel_cmd

    def visualize_path(self, path):
        """중앙 경로 시각화"""
        if path is None or len(path) == 0:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "center_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # 경로 포인트 추가
        for (px, py) in path:
            point = Point()
            point.x = px
            point.y = py
            point.z = 0.1  # 지면보다 약간 위
            marker.points.append(point)

        # 선 스타일
        marker.scale.x = 0.3  # 선 두께
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.viz_path_pub.publish(marker)

    def visualize_target_point(self, target_point):
        """목표 포인트 시각화"""
        if target_point is None:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "target_point"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = target_point[0]
        marker.pose.position.y = target_point[1]
        marker.pose.position.z = 0.5

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.viz_target_pub.publish(marker)

    def callback_timer(self, event):
        """타이머 콜백 - 주기적으로 제어 수행"""

        if self.vehicle_state is None:
            return

        # 중앙 경로 찾기
        center_path = self.find_center_lane()

        # 조향 제어
        target_point = None
        if center_path is not None and len(center_path) > 0:
            steer_cmd, target_point = self.pure_pursuit_control(center_path)
            # 경로 시각화
            self.visualize_path(center_path)
            self.visualize_target_point(target_point)
        else:
            steer_cmd = 0.0

        # 속도 제어
        accel_cmd = self.speed_control()

        # 명령 발행
        self.steer_pub.publish(Float32(steer_cmd))
        self.accel_pub.publish(Float32(accel_cmd))


def main():
    """메인 함수"""
    rospy.init_node("control_node")

    try:
        node = ControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
