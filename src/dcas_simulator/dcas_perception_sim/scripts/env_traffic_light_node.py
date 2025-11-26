#!/usr/bin/env python
"""
환경: 신호등 퍼블리셔

- CSV에서 신호등 위치/상태를 읽어 /env/traffic_lights 로 퍼블리시합니다.
- RViz에는 SPHERE 마커로 시각화하여 색으로 상태(적/황/녹)를 표시합니다.
"""
import rospy
import os
import csv
from dcas_msgs.msg import TrafficLight, TrafficLightArray
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray


class EnvTrafficLightPublisher:
    """신호등 퍼블리셔 클래스
    - 주기적으로 CSV를 읽어 TrafficLightArray와 마커를 퍼블리시합니다.
    """
    def __init__(self) -> None:
        # Parameters
        self.pub = rospy.Publisher("/env/traffic_lights", TrafficLightArray, queue_size=10)
        self.csv_path = rospy.get_param("/map/traffic_lights_csv", "traffic_lights.csv")
        period = float(rospy.get_param("/env_traffic_light_node/period_s", 0.5))

        # Publishers
        self.viz_pub = rospy.Publisher("/viz/env/traffic_lights", MarkerArray, queue_size=10)

        # Timer
        self.duration = period
        self.timer = rospy.Timer(rospy.Duration(period), self.CallbackTimer)

    # 주기 콜백
    # - CSV를 읽어 신호등 배열과 마커 퍼블리시
    def CallbackTimer(self, _evt) -> None:
        """주기적으로 CSV를 읽어 신호등 배열과 마커를 퍼블리시합니다."""
        # Create a TrafficLightArray message
        arr = TrafficLightArray()
        arr.header.stamp = rospy.Time.now()

        # Read the traffic lights from the CSV file
        loaded = self.ReadTrafficLightFromCsv(self.csv_path, arr)
        if not loaded:
            rospy.logwarn("env_traffic_light_node: CSV not found or empty: %s", self.csv_path)

        # Publish the TrafficLightArray message
        self.pub.publish(arr)

        # Publish the markers
        marker_arr = self.TrafficLightsToMarkers(arr)
        self.viz_pub.publish(marker_arr)

    # CSV에서 신호등 목록을 읽어 TrafficLightArray에 채움
    # - 스키마: x, y, (optional z), (optional state)
    def ReadTrafficLightFromCsv(self, path: str, arr: TrafficLightArray) -> bool:
        """CSV에서 신호등 목록을 읽어 TrafficLightArray에 채웁니다.

        인자:
            path: CSV 파일 경로
            arr: 결과를 채워 넣을 TrafficLightArray
        """
        full = path
        if not os.path.isabs(full):
            pkg_dir = os.path.dirname(os.path.dirname(__file__))
            base_dir = os.path.abspath(os.path.join(pkg_dir, "maps"))
            full = os.path.join(base_dir, path)
        if not os.path.exists(full):
            return False
        
        # Read the CSV file
        with open(full, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # ============================ #
                    # ====== TODO: .csv 파싱 ====== #
                    # ============================ #
                    x = float(row.get('x'))
                    y = float(row.get('y'))
                    z = float(row.get('z'))
                    state = int(row.get('state'))
                    # ============================ #
                    # ============================ #
                    # ============================ #

                except Exception:
                    continue
                # Create a TrafficLight message
                tl = TrafficLight()
                tl.header = arr.header

                # ====================================== #
                # ====== TODO: TrafficLight 채우기 ====== #
                # ====================================== #

                # yaw 회전 --> Quaternion변환
                # Roll: phi, Pitch: theta, Yaw: psi
                # q_x = sin(phi/2)cos(theta/2)cos(psi/2) - cos(phi/2)sin(theta/2)sin(yaw/2)
                # q_y = cos(phi/2)sin(theta/2)cos(psi/2) + sin(phi/2)cos(theta/2)sin(yaw/2)
                # q_z = cos(phi/2)cos(theta/2)sin(psi/2) + sin(phi/2)sin(theta/2)cos(yaw/2)
                # q_w = cos(phi/2)cos(theta/2)cos(psi/2) + sin(phi/2)sin(theta/2)sin(yaw/2)

                tl.pose = Pose(position=Point(x=x, y=y, z=z), orientation=Quaternion(0, 0, 0, 1))
                tl.state = state
                
                # ====================================== #
                # ====================================== #
                # ====================================== #
                arr.lights.append(tl)
        return True

    # TrafficLightArray를 RViz MarkerArray로 변환 (상태별 색상 적용)
    def TrafficLightsToMarkers(self, msg: TrafficLightArray) -> MarkerArray:
        """TrafficLightArray를 RViz MarkerArray로 변환합니다.

        인자:
            msg: 신호등 배열(TrafficLightArray)
        """
        arr = MarkerArray()
        for idx, tl in enumerate(msg.lights):
            m = self.MakeMarker(ns="traffic_lights", mid=idx, mtype=Marker.SPHERE)
            m.pose = tl.pose
            m.scale.x = m.scale.y = m.scale.z = 0.6
            if tl.state == 0:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0
            elif tl.state == 1:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.0, 1.0
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
            arr.markers.append(m)
        return arr

    # 신호등 마커(SPHERE) 생성 유틸리티
    def MakeMarker(self, ns: str, mid: int, mtype: int, frame: str = "map"):
        """신호등 시각화를 위한 마커를 생성합니다.

        인자:
            ns: RViz 네임스페이스
            mid: 마커 ID
            mtype: 마커 타입
            frame: 기준 프레임
        """
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.lifetime = rospy.Duration(self.duration)
        return m


def Main() -> None:
    # 노드 초기화
    rospy.init_node("env_traffic_light_node")

    # 퍼블리셔 인스턴스 생성
    EnvTrafficLightPublisher()

    # 시작 로그 출력
    rospy.loginfo("env_traffic_light_node started: publishing /env/traffic_lights")
    rospy.spin()


if __name__ == "__main__":
    Main()


