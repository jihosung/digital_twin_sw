#!/usr/bin/env python
"""
환경: 교통 표지판 퍼블리셔

- CSV에서 표지판 위치/종류를 읽어 /env/traffic_signs 로 퍼블리시합니다.
- RViz에는 표지판 크기의 CUBE 마커로 시각화합니다.
"""
import math
import rospy
import os
import csv
from dcas_msgs.msg import TrafficSign, TrafficSignArray
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray


class EnvSignPublisher:
    """교통 표지판 퍼블리셔 클래스
    - 주기적으로 CSV를 읽어 TrafficSignArray와 시각화 마커를 퍼블리시합니다.
    """
    def __init__(self) -> None:
        # Parameters
        self.pub = rospy.Publisher("/env/traffic_signs", TrafficSignArray, queue_size=10)
        period = float(rospy.get_param("/env_sign_node/period_s", 1.0))

        # Publishers
        self.viz_pub = rospy.Publisher("/viz/env/signs", MarkerArray, queue_size=10)
        self.csv_path = rospy.get_param("/map/signs_csv", "signs.csv")

        # Timer
        self.duration = period
        self.timer = rospy.Timer(rospy.Duration(period), self.CallbackTimer)

    # 주기 콜백
    # - CSV를 읽어 TrafficSignArray 퍼블리시 및 마커 퍼블리시
    def CallbackTimer(self, _evt) -> None:
        """주기적으로 CSV를 읽어 표지판 메시지/마커를 퍼블리시합니다."""
        # Create a TrafficSignArray message
        arr = TrafficSignArray()
        arr.header.stamp = rospy.Time.now()
        
        # Read the signs from the CSV file
        signs = self.ReadSignsFromCsv(self.csv_path)
        if signs:
            arr.signs = signs
        else:
            rospy.logwarn("env_sign_node: CSV not found or empty: %s", self.csv_path)

        # Publish the TrafficSignArray message
        self.pub.publish(arr)

        # Publish the markers
        marker_arr = self.SignsToMarkers(arr)
        self.viz_pub.publish(marker_arr)

    # CSV에서 표지판 목록을 읽어 리스트로 반환
    # - 스키마: x, y, (optional z), (optional yaw), (optional sign_class)
    def ReadSignsFromCsv(self, path: str) -> list:
        """CSV에서 표지판 목록을 읽어 리스트로 반환합니다.

        인자:
            path: CSV 파일 경로
        반환:
            TrafficSign 메시지 리스트
        """
        # Check if the path is absolute
        full = path
        if not os.path.isabs(full):
            pkg_dir = os.path.dirname(os.path.dirname(__file__))
            base_dir = os.path.abspath(os.path.join(pkg_dir, "maps"))
            full = os.path.join(base_dir, path)
        
        # Check if the file exists
        if not os.path.exists(full):
            return []
        
        signs = []
        # Read the CSV file
        with open(full, newline='') as f:
            reader = csv.DictReader(f)
            idx = 0
            for row in reader:
                try:
                    # ============================ #
                    # ====== TODO: .csv 파싱 ====== #
                    # ============================ #
                    x = float(row.get('x', 0.0))
                    y = float(row.get('y', 0.0))
                    z = float(row.get('z', 0.0)) # default : 2.0
                    yaw = float(row.get('yaw', 0.0))
                    sign_class = int(row.get('sign_class', 0))
                    # ============================ #
                    # ============================ #
                    # ============================ #
                except Exception:
                    continue
                
                # Create a TrafficSign for the sign
                sign = TrafficSign()

                # ===================================== #
                # ====== TODO: TrafficSign 채우기 ====== #
                # ===================================== #
                sign.id = idx
                idx += 1
                sign.sign_class = sign_class
                # Convert yaw to quaternion
                yaw_rad = math.radians(yaw)

                # yaw 회전 --> Quaternion변환
                # Roll: phi, Pitch: theta, Yaw: psi
                # q_x = sin(phi/2)cos(theta/2)cos(psi/2) - cos(phi/2)sin(theta/2)sin(yaw/2)
                # q_y = cos(phi/2)sin(theta/2)cos(psi/2) + sin(phi/2)cos(theta/2)sin(yaw/2)
                # q_z = cos(phi/2)cos(theta/2)sin(psi/2) + sin(phi/2)sin(theta/2)cos(yaw/2)
                # q_w = cos(phi/2)cos(theta/2)cos(psi/2) + sin(phi/2)sin(theta/2)sin(yaw/2)
                quat = Quaternion(0, 0, math.sin(yaw_rad * 0.5), math.cos(yaw_rad * 0.5))
                sign.pose = Pose(position=Point(x=x, y=y, z=z), orientation=quat)

                # ===================================== #
                # ===================================== #
                # ===================================== #
                
                signs.append(sign)
        return signs

    # TrafficSignArray를 RViz MarkerArray로 변환 (표지판 종류별 색상)
    def SignsToMarkers(self, msg: TrafficSignArray) -> MarkerArray:
        """TrafficSignArray를 RViz MarkerArray로 변환합니다.

        인자:
            msg: 표지판 배열(TrafficSignArray)
        """
        # Create a MarkerArray
        arr = MarkerArray()
        for sign in msg.signs:
            m = self.MakeMarker(ns="signs", mid=sign.id, mtype=Marker.CUBE)
            m.pose = sign.pose
            m.scale.x, m.scale.y, m.scale.z = 0.1, 0.6, 0.6  # Standard sign size
            
            # Set color based on sign class
            if sign.sign_class == TrafficSign.CLASS_STOP:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0  # red
            elif sign.sign_class == TrafficSign.CLASS_YIELD:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.0, 1.0  # yellow
            elif sign.sign_class in [TrafficSign.CLASS_SPEED_LIMIT_30, TrafficSign.CLASS_SPEED_LIMIT_50, 
                                   TrafficSign.CLASS_SPEED_LIMIT_60, TrafficSign.CLASS_SPEED_LIMIT_80]:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 1.0  # white
            elif sign.sign_class == TrafficSign.CLASS_NO_ENTRY:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0  # red
            elif sign.sign_class == TrafficSign.CLASS_PEDESTRIAN_CROSSING:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.0, 1.0, 1.0  # blue
            elif sign.sign_class == TrafficSign.CLASS_SCHOOL_ZONE:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0  # orange
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.6, 1.0, 1.0  # default blue
            
            arr.markers.append(m)
        return arr

    # 표지판 마커(CUBE)를 생성하는 유틸리티
    def MakeMarker(self, ns: str, mid: int, mtype: int, frame: str = "map"):
        """표지판 시각화를 위한 마커를 생성합니다.

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



def main() -> None:
    # 노드 초기화
    rospy.init_node("env_sign_node")

    # 퍼블리셔 인스턴스 생성
    EnvSignPublisher()

    # 시작 로그 출력
    rospy.loginfo("env_sign_node started: publishing /env/traffic_signs")
    rospy.spin()


if __name__ == "__main__":
    main()


