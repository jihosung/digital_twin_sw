#!/usr/bin/env python
"""
Evaluation Node

이 노드는 차량의 주행 성능을 평가합니다.
lanes.csv 파일을 읽어와 차량의 현재 위치와 비교하여
Cross Track Error(CTE)를 계산합니다.

입력:
- /vehicle/state: 차량 상태
- maps/lanes.csv: ground truth 차선 정보

출력:
- /evaluation/cte: Cross Track Error (m)
- /evaluation/metrics: 평가 지표 (평균 CTE, 최대 CTE 등)
"""

import os
import rospy
import numpy as np
import math
import csv
from dcas_msgs.msg import VehicleState
from std_msgs.msg import Float32
from geometry_msgs.msg import Quaternion
import rospkg


class EvaluationNode:
    """주행 평가 노드

    차량의 CTE(Cross Track Error)를 계산하고 주행 성능을 평가합니다.
    """

    def __init__(self):
        """노드 초기화"""

        # ROS 파라미터
        self.evaluation_rate = rospy.get_param("~evaluation_rate", 10.0)  # 평가 주기 (Hz)

        # 상태 변수
        self.vehicle_state = None
        self.lane_data = {}
        self.center_lanes = {}
        self.cte_history = []
        self.max_cte = 0.0

        # lanes.csv 파일 로드 및 중앙 차선 계산
        self._load_lane_csv()

        # Publisher
        self.cte_pub = rospy.Publisher(
            "/evaluation/cte",
            Float32,
            queue_size=10
        )

        # Subscriber
        rospy.Subscriber(
            "/vehicle/state",
            VehicleState,
            self.callback_vehicle_state
        )

        # 타이머
        timer_period = 1.0 / self.evaluation_rate
        self.timer = rospy.Timer(rospy.Duration(timer_period), self.callback_timer)

        rospy.loginfo("Evaluation node initialized")

    def _load_lane_csv(self):
        """lanes.csv 파일 로드 및 중앙 차선 계산"""
        try:
            # ROS 패키지 경로 찾기
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('dcas_perception_sim')
            csv_path = os.path.join(pkg_path, 'maps', 'lanes.csv')

            if not os.path.exists(csv_path):
                rospy.logerr(f"Lane CSV file not found: {csv_path}")
                return

            # CSV 파일 읽기
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lane_id = int(row['lane_id'])
                    if lane_id not in self.lane_data:
                        self.lane_data[lane_id] = {
                            'points': [],
                            'lane_type': int(row['lane_type'])
                        }

                    point = np.array([
                        float(row['x']),
                        float(row['y']),
                        float(row['z'])
                    ])
                    self.lane_data[lane_id]['points'].append(point)

            # numpy 배열로 변환
            for lane_id in self.lane_data:
                self.lane_data[lane_id]['points'] = np.array(
                    self.lane_data[lane_id]['points']
                )

            rospy.loginfo(f"Loaded {len(self.lane_data)} lanes from CSV")

            # 중앙 차선들 계산
            self._compute_center_lanes()

        except Exception as e:
            rospy.logerr(f"Failed to load lane CSV: {e}")

    def _compute_center_lanes(self):
        """차선 중앙선들 계산

        3개의 차선(lane_id 0, 1, 2)이 있을 때:
        - center_lane 1: lane 0과 1의 중점
        - center_lane 2: lane 1과 2의 중점
        """
        self.center_lanes = {}

        # lane_id 0, 1, 2가 모두 있는지 확인
        if 0 not in self.lane_data or 1 not in self.lane_data or 2 not in self.lane_data:
            rospy.logwarn("Lane ID 0, 1, or 2 not found. Cannot compute center lanes.")
            return

        lane0_points = self.lane_data[0]['points']
        lane1_points = self.lane_data[1]['points']
        lane2_points = self.lane_data[2]['points']

        # Center lane 1: lane 0과 1 사이
        min_len_01 = min(len(lane0_points), len(lane1_points))
        center_lane_1 = []
        for i in range(min_len_01):
            p0 = lane0_points[i]
            p1 = lane1_points[i]
            center_point = (p0 + p1) / 2.0
            center_lane_1.append(center_point)
        self.center_lanes[1] = np.array(center_lane_1)

        # Center lane 2: lane 1과 2 사이
        min_len_12 = min(len(lane1_points), len(lane2_points))
        center_lane_2 = []
        for i in range(min_len_12):
            p1 = lane1_points[i]
            p2 = lane2_points[i]
            center_point = (p1 + p2) / 2.0
            center_lane_2.append(center_point)
        self.center_lanes[2] = np.array(center_lane_2)

        rospy.loginfo(
            f"Computed center lanes: "
            f"center_lane_1 ({len(self.center_lanes[1])} points), "
            f"center_lane_2 ({len(self.center_lanes[2])} points)"
        )

    def callback_vehicle_state(self, msg):
        """차량 상태 콜백"""
        self.vehicle_state = msg

    def quaternion_to_yaw(self, quat):
        """쿼터니언을 yaw 각도로 변환"""
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def find_closest_point_on_lane(self, vehicle_pos, lane_points):
        """차선에서 차량과 가장 가까운 점 찾기

        선분(두 포인트 사이)까지의 최단거리를 계산하여 정확한 CTE를 구합니다.

        Args:
            vehicle_pos: 차량 위치 (numpy array [x, y])
            lane_points: 차선 포인트들 (numpy array [N, 3])

        Returns:
            tuple: (closest_point, closest_index, distance)
        """
        if len(lane_points) == 0:
            return None, None, float('inf')

        min_distance = float('inf')
        closest_point = None
        closest_segment_idx = 0

        vehicle_xy = vehicle_pos[:2]

        # 각 선분에 대해 최단거리 계산
        for i in range(len(lane_points) - 1):
            p1 = lane_points[i][:2]
            p2 = lane_points[i + 1][:2]

            # 선분 p1-p2에 대한 차량의 투영점 계산
            segment = p2 - p1
            segment_length_sq = np.dot(segment, segment)

            if segment_length_sq < 1e-6:
                # 선분이 점에 가까운 경우
                projection_point = p1
            else:
                # 매개변수 t: 0이면 p1, 1이면 p2
                t = np.clip(np.dot(vehicle_xy - p1, segment) / segment_length_sq, 0.0, 1.0)
                projection_point = p1 + t * segment

            # 투영점까지의 거리
            distance = np.linalg.norm(vehicle_xy - projection_point)

            if distance < min_distance:
                min_distance = distance
                closest_point = np.array([projection_point[0], projection_point[1], 0.0])
                closest_segment_idx = i

        return closest_point, closest_segment_idx, min_distance

    def calculate_cte(self):
        """Cross Track Error 계산

        차량의 현재 위치와 가장 가까운 중앙 차선 간의 수직 거리를 계산합니다.
        두 중앙 차선 중 가까운 쪽을 자동으로 선택합니다.

        Returns:
            float or None: CTE (m), 계산 불가능한 경우 None
        """
        if self.vehicle_state is None:
            return None

        if not self.center_lanes:
            rospy.logwarn_throttle(5.0, "Center lanes not available")
            return None

        # 차량 위치
        vehicle_x = self.vehicle_state.pose.position.x
        vehicle_y = self.vehicle_state.pose.position.y
        vehicle_pos = np.array([vehicle_x, vehicle_y, 0.0])

        # 두 중앙 차선 중 가까운 쪽 찾기
        min_cte = None
        min_distance = float('inf')

        for center_lane_id, center_lane_points in self.center_lanes.items():
            # 가장 가까운 선분 찾기 (선분까지의 최단거리)
            closest_point, closest_segment_idx, distance = self.find_closest_point_on_lane(
                vehicle_pos, center_lane_points
            )

            if closest_point is None:
                continue

            # 이 중앙 차선이 더 가까운 경우
            if distance < min_distance:
                min_distance = distance

                # CTE 계산 (부호 포함)
                # 가장 가까운 선분의 방향 벡터 사용
                if closest_segment_idx < len(center_lane_points) - 1:
                    p1 = center_lane_points[closest_segment_idx][:2]
                    p2 = center_lane_points[closest_segment_idx + 1][:2]
                    lane_vector = p2 - p1
                elif closest_segment_idx > 0:
                    # 마지막 선분
                    p1 = center_lane_points[closest_segment_idx - 1][:2]
                    p2 = center_lane_points[closest_segment_idx][:2]
                    lane_vector = p2 - p1
                else:
                    # 단일 점인 경우 부호 없는 거리
                    min_cte = distance
                    continue

                # 차선 방향 벡터 정규화
                lane_vector_norm = np.linalg.norm(lane_vector)
                if lane_vector_norm < 1e-6:
                    min_cte = distance
                    continue

                lane_direction = lane_vector / lane_vector_norm

                # 차량과 투영점 간의 벡터
                vehicle_to_lane = vehicle_pos[:2] - closest_point[:2]

                # 외적을 사용하여 좌우 판단 (2D)
                # lane_direction과 vehicle_to_lane의 외적의 z 성분
                cross_product = lane_direction[0] * vehicle_to_lane[1] - lane_direction[1] * vehicle_to_lane[0]

                # 부호 결정: 양수면 왼쪽, 음수면 오른쪽
                min_cte = math.copysign(distance, cross_product)

        return min_cte

    def update_metrics(self, cte):
        """평가 지표 업데이트

        Args:
            cte: Cross Track Error (m)
        """
        if cte is None:
            return

        # CTE 히스토리 저장
        self.cte_history.append(abs(cte))

        # 최대 CTE 업데이트
        if abs(cte) > self.max_cte:
            self.max_cte = abs(cte)

        # 주기적으로 통계 출력 (10초마다)
        if len(self.cte_history) % 100 == 0:
            mean_cte = np.mean(self.cte_history)
            std_cte = np.std(self.cte_history)
            rospy.loginfo(
                f"CTE Statistics - Mean: {mean_cte:.3f}m, "
                f"Std: {std_cte:.3f}m, Max: {self.max_cte:.3f}m"
            )

    def callback_timer(self, event):
        """타이머 콜백 - 주기적으로 CTE 계산"""

        if self.vehicle_state is None:
            return

        # CTE 계산
        cte = self.calculate_cte()

        if cte is not None:
            # CTE 발행
            self.cte_pub.publish(Float32(cte))

            # 평가 지표 업데이트
            self.update_metrics(cte)


def main():
    """메인 함수"""
    rospy.init_node("evaluation_node")

    try:
        node = EvaluationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
