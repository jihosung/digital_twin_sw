#!/usr/bin/env python
"""
환경: 가드레일 퍼블리셔

- CSV에서 가드레일 지상진실(ObjectArray)을 읽어 /env/guardrails 로 퍼블리시합니다.
- RViz에는 CUBE 마커로 간단히 시각화합니다.
- yaw(도)를 내부에서 쿼터니언으로 변환해 Pose를 구성합니다.
"""
import rospy
import os
import csv
import math
from dcas_msgs.msg import Object, ObjectArray
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Twist
from visualization_msgs.msg import Marker, MarkerArray


class EnvGuardrailPublisher:
    """가드레일 퍼블리셔 클래스
    - CSV를 읽어 ObjectArray를 구성하고, 주기적으로 메시지/마커를 퍼블리시합니다.
    - 마커 수명은 period_s로 설정하여 RViz 갱신에 유리합니다.
    """
    def __init__(self) -> None:
        # Parameters
        self.csv_path = rospy.get_param("/map/guardrails_csv", "guardrails.csv")
        period = float(rospy.get_param("/env_guardrail_node/period_s", 1.0))

        # Publishers
        self.pub = rospy.Publisher("/env/guardrails", ObjectArray, queue_size=10)
        self.viz_pub = rospy.Publisher("/viz/env/guardrails", MarkerArray, queue_size=10)

        # Timer
        self.duration = period
        self.timer = rospy.Timer(rospy.Duration(period), self.CallbackTimer)

    # 주기 콜백
    # - CSV에서 가드레일을 읽어 퍼블리시 및 시각화 마커 퍼블리시
    def CallbackTimer(self, _evt) -> None:
        """주기적으로 CSV를 읽어 가드레일 메시지/마커를 퍼블리시합니다.

        - 매 주기마다 CSV를 파싱하여 ObjectArray를 갱신하고 토픽으로 발행합니다.
        - 동일한 내용의 RViz 마커도 함께 발행합니다.
        """
        object_array = ObjectArray()
        object_array.header.stamp = rospy.Time.now()
        loaded = self.ReadGuardrailsFromCsv(self.csv_path, object_array)
        if not loaded:
            rospy.logwarn("env_guardrail_node: CSV not found or empty: %s", self.csv_path)
        self.pub.publish(object_array)
        marker_arr = self.ObjectsToMarkers(object_array, group="guardrails")
        self.viz_pub.publish(marker_arr)

    # CSV에서 가드레일 목록을 읽어 ObjectArray로 변환
    # - 경로가 상대면 패키지 maps/ 하위로 해석합니다.
    # - 각 행에서 중심(cx,cy), 길이, yaw_deg을 읽어 Pose/Size를 구성합니다.
    def ReadGuardrailsFromCsv(self, path: str, object_array: ObjectArray) -> bool:
        """CSV 파일에서 가드레일 정보를 읽어 ObjectArray에 채웁니다.

        인자:
            path: CSV 파일 경로(상대 경로면 패키지 maps/ 기준으로 해석)
            object_array: 결과를 채워 넣을 ObjectArray

        반환:
            하나 이상 로드되면 True, 없으면 False
        """
        # Check if the path is absolute
        full = path
        if not os.path.isabs(full):
            pkg_dir = os.path.dirname(os.path.dirname(__file__))
            base_dir = os.path.abspath(os.path.join(pkg_dir,"maps"))
            full = os.path.join(base_dir, path)
        if not os.path.exists(full):
            return False
        # Convert yaw to quaternion
        def yaw_to_quat(yaw_deg: float) -> Quaternion:
            half = math.radians(yaw_deg) * 0.5
            return Quaternion(0.0, 0.0, math.sin(half), math.cos(half))
        # Read the CSV file
        with open(full, newline='') as f:
            reader = csv.DictReader(f)
            idx = 0
            for row in reader:
                try:
                    #===================================================#
                    #============== TODO: .csv 파일 파싱 ================#
                    #===================================================#
                    cx = float(row.get('cx', 0.0))
                    cy = float(row.get('cy', 0.0))
                    yaw_deg = float(row.get('yaw_deg', 0.0))    
                    length = float(row.get('length', 0.0))     # set default 10.0
                    #===================================================#
                    #===================================================#
                    #===================================================#

                except Exception:
                    continue
                # Create an object for the guardrail
                obj = Object()
                obj.header = object_array.header # Header

                #==============================================================#
                #==== TODO: object msg의 빈부분 채우기(geom_type은 채울필요 X) ====#
                #==============================================================#
                obj.id = idx
                idx += 1
                obj.type = Object.LABEL_GUARDRAIL
                obj.pose = Pose(position=Point(x=cx, y=cy, z=0.0), orientation=yaw_to_quat(yaw_deg)) # Pose
                obj.size = Vector3(length, 0.2, 0.8) # Size
                obj.twist = Twist() # Zero twist
                #==============================================================#
                #==== TODO: object msg의 빈부분 채우기(geom_type은 채울필요 X) ====#
                #==============================================================#
                
                object_array.objects.append(obj)
        return True

    # ObjectArray(가드레일)를 MarkerArray로 변환
    # - 각 가드레일을 CUBE 마커로 표현 (position/size 반영)
    def ObjectsToMarkers(self, msg: ObjectArray, group: str) -> MarkerArray:
        """가드레일 ObjectArray를 RViz MarkerArray로 변환합니다.

        인자:
            msg: 가드레일 목록(ObjectArray)
            group: 마커 그룹 이름(RViz 네임스페이스 구성에 사용)
        """
        markers = MarkerArray()
        for obj in msg.objects:
            # Create a marker for the guardrail
            m = self.MakeMarker(ns=f"{group}", mid=obj.id, mtype=Marker.CUBE)
            m.pose = obj.pose
            m.scale.x, m.scale.y, m.scale.z = obj.size.x, obj.size.y, obj.size.z
            # Set the color of the marker
            if group == "guardrails":
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 1.0
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.2, 1.0
            markers.markers.append(m)
        return markers
    
    # RViz에서 사용할 가드레일 Marker를 생성하는 유틸리티
    # - 입력: 네임스페이스, 마커 ID, 타입, 프레임
    # - 출력: 공통 속성이 세팅된 Marker
    def MakeMarker(self, ns: str, mid: int, mtype: int, frame: str = "map"):
        """가드레일 시각화를 위한 마커를 생성합니다.

        인자:
            ns: RViz 네임스페이스(관련 마커를 묶는 용도)
            mid: 마커 ID(네임스페이스 내에서 고유)
            mtype: 마커 타입(예: CUBE 등)
            frame: 기준 프레임(기본: map)
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
    # Initialize the node
    rospy.init_node("env_guardrail_node")

    # Create an instance of the EnvGuardrailPublisher class
    EnvGuardrailPublisher()

    # Log the start of the node
    rospy.loginfo("env_guardrail_node started: publishing /env/guardrails")
    rospy.spin()


if __name__ == "__main__":
    Main()


