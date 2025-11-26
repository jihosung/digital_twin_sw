#!/usr/bin/env python
"""
Environment: Surrounding vehicles (dynamic actors)

This node publishes surrounding vehicle ground-truth as dcas_msgs/ObjectArray
and their simple visualization markers.

Input CSV (polylines per vehicle):
- /map/surrounding_vehicles_csv (default: surrounding_vehicles.csv)
- Schema: id,seq,x,y,speed_mps
  * id: vehicle identifier
  * seq: waypoint order for the given id (0,1,2,...)
  * x,y: waypoint position in map frame
  * speed_mps: constant speed along the polyline for the vehicle

Behavior:
- Each vehicle moves along its waypoint polyline at constant speed.
- Heading (yaw) is aligned with current segment direction.
- File changes are auto-detected via mtime and reloaded at runtime.
"""
# CSV에 정의된 차량 경로(폴리라인)를 따라 일정 속도로 이동하는 차량들을 퍼블리시합니다.
# 각 차량의 헤딩(yaw)은 현재 진행 중인 세그먼트 방향으로 맞춥니다.
# CSV 파일이 갱신되면 자동으로 재로드합니다.
import os
import csv
import math
import rospy
from dcas_msgs.msg import Object, ObjectArray
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Twist
from visualization_msgs.msg import Marker, MarkerArray


class EnvSurroundingVehiclePublisher:
    # 마주오는 차량 퍼블리셔 클래스
    # - 타이머 주기로 차량 상태를 진전시키고 ObjectArray/MarkerArray를 퍼블리시합니다.
    """Publishes surrounding vehicles driven by CSV waypoints."""
    def __init__(self) -> None:
        """퍼블리셔/타이머 초기화 및 초기 CSV 로드."""
        # Parameters
        self.csv_path = rospy.get_param("/map/surrounding_vehicles_csv", "surrounding_vehicles.csv")
        period = float(rospy.get_param("/env_surrounding_vehicle_node/period_s", 0.2))

        # Publishers
        self.pub = rospy.Publisher("/env/surrounding_vehicles", ObjectArray, queue_size=10)
        self.viz_pub = rospy.Publisher("/viz/env/vehicles", MarkerArray, queue_size=10)

        # Timer
        self.duration = period
        self.timer = rospy.Timer(rospy.Duration(period), self.CallbackTimer)

        # State
        self.vehicles = {}
        self.t = 0.0

        # CSV file path and mtime for incremental reload
        self.csv_full = self.ResolveMapPath(self.csv_path)
        self.csv_mtime = None
        if not self.ReadSurroundingFromCsv(self.csv_path):
            rospy.logwarn("env_surrounding_vehicle_node: CSV not found or empty: %s", self.csv_path)

    # 주기 콜백
    # - dt만큼 각 차량을 폴리라인 상에서 전진시키고, 포즈/속도를 계산하여 퍼블리시합니다.
    # - CSV 변경을 감지하여 런타임에 재로드합니다.
    def CallbackTimer(self, evt) -> None:
        """각 차량을 폴리라인을 따라 전진시키고 상태/마커를 퍼블리시."""
        dt = (evt.current_real.to_sec() - evt.last_real.to_sec()) if evt.last_real else self.duration
        self.t += dt
        # Reload CSV if file updated
        try:
            full = self.ResolveMapPath(self.csv_path)
            mtime = os.path.getmtime(full)
            if self.csv_mtime is None or mtime > self.csv_mtime:
                if self.ReadSurroundingFromCsv(self.csv_path):
                    self.csv_full = full
                    self.csv_mtime = mtime
                    rospy.loginfo("env_surrounding_vehicle_node: reloaded surrounding CSV (%s)", full)
        except Exception:
            rospy.logerr("env_surrounding_vehicle_node: error in CallbackTimer")
            pass

        # Create an ObjectArray message
        objects = ObjectArray()
        objects.header.stamp = rospy.Time.now()

        # Advance each vehicle along its polyline
        for vehicle_id, st in self.vehicles.items():
            trajectory = st['trajectory'] # [[x, y, speed], ...]
            if len(trajectory) < 2:
                continue
            # Get the current segment index and progress
            seg_idx = st['seg_idx'] 
            seg_s = st['seg_s']
            # Get the current speed
            speed = float(trajectory[seg_idx][2])
            # Compute the distance to advance along the polyline
            move = speed * dt # Distance to advance along polyline
            # Advance along polyline
            if len(trajectory) > 2:
                p0 = trajectory[seg_idx][:2] # Current segment start point
                p1 = trajectory[(seg_idx + 1) % len(trajectory)][:2] # Next segment end point
                dx = p1[0] - p0[0] # Segment length in x direction
                dy = p1[1] - p0[1] # Segment length in y direction
                seg_len = max(1e-6, math.hypot(dx, dy))
                remain = seg_len - seg_s # Remaining distance to next segment

                # Check if the vehicle can advance along the current segment
                if move < remain:
                    seg_s += move
                    move = 0.0
                else:
                    move -= remain
                    seg_idx = (seg_idx + 1) % len(trajectory) # Advance to next segment or return to start
                    seg_s = 0.0

            # If the vehicle has reached the end of the polyline, reset the segment index and progress
            if seg_idx == len(trajectory) - 1:
                seg_idx = 0
                seg_s = 0.0

            # Compute current pose on segment
            p0 = trajectory[seg_idx][:2]
            p1 = trajectory[(seg_idx + 1) % len(trajectory)][:2]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            seg_len = max(1e-6, math.hypot(dx, dy))
            nx = dx / seg_len
            ny = dy / seg_len

            point_x = p0[0] + nx * seg_s
            point_y = p0[1] + ny * seg_s
            yaw = math.atan2(dy, dx)

            # Update state
            st['seg_idx'] = seg_idx
            st['seg_s'] = seg_s
            
            # Create object
            obj = Object()
            obj.header = objects.header
            twist = Twist() # Zero twist
            # ================================= #
            # ====== TODO: obj msg 채우기 ====== #
            # ================================= #
            obj.id = vehicle_id
            obj.type = Object.LABEL_VEHICLE
            obj.pose = Pose(position=Point(x=point_x, y=point_y, z=0.0), orientation=self.YawToQuaternion(yaw))
            obj.size = Vector3(4.5, 1.9, 1.6) # Default size for vehicle (4.5, 1.9, 1.6)
            twist.linear.x = speed * math.cos(yaw) # Linear velocity in x direction
            twist.linear.y = speed * math.sin(yaw) # Linear velocity in y direction
            # ================================= #
            # ================================= #
            # ================================= #
            obj.twist = twist
            objects.objects.append(obj)

        # Publish the ObjectArray message
        self.pub.publish(objects)

        # Publish the markers
        marker_arr = self.ObjectsToMarkers(objects, group="vehicles")
        self.viz_pub.publish(marker_arr)

    # CSV에서 차량 경로(waypoints)를 읽어 내부 상태(self.vehicles)에 저장
    # - 스키마: id, seq, x, y, speed_mps
    # - 반환: 유효 차량(>=2개 지점)이 하나 이상이면 True
    def ReadSurroundingFromCsv(self, path: str) -> bool:
        """CSV에서 폴리라인을 읽어 차량 상태를 구성. 최소 2개 지점을 가진 차량 존재 시 True."""
        full = self.ResolveMapPath(path)
        if not os.path.exists(full):
            return False
        id_to_trajectory = {} # Dictionary to store waypoints for each vehicle
        with open(full, newline='') as f:
            reader = csv.DictReader(f)
            rows = [] # List to store waypoints for each vehicle
            for row in reader:
                try:
                    
                    # ============================ #
                    # ====== TODO: .csv 파싱 ====== #
                    # ============================ #
                    vehicle_id = int(row.get('id')) # Vehicle ID
                    seq = int(row.get('seq')) # Waypoint order, default = 0
                    x = float(row.get('x')) # X coordinate
                    y = float(row.get('y')) # Y coordinate
                    speed_mps = float(row.get('speed_mps')) # Speed
                    # ============================ #
                    # ============================ #
                    # ============================ #
                except Exception:
                    continue
                rows.append((vehicle_id, seq, x, y, speed_mps)) # Add waypoint to vehicle ID
        if not rows:
            return False
        # Sort by vehicle ID and sequence order
        rows.sort(key=lambda r: (r[0], r[1]))
        for vehicle_id, seq, x, y, speed in rows:
            if vehicle_id not in id_to_trajectory:
                id_to_trajectory[vehicle_id] = []
            id_to_trajectory[vehicle_id].append((x, y, speed)) # Add waypoint to vehicle ID

        # Build state
        self.vehicles = {} # Dictionary to store state for each vehicle
        for vehicle_id, trajectory in id_to_trajectory.items():
            if len(trajectory) < 2:
                continue
            self.vehicles[vehicle_id] = {'trajectory': trajectory, 'seg_idx': 0, 'seg_s': 0.0}
        return len(self.vehicles) > 0

    # 패키지 maps/ 기준 상대 경로를 절대 경로로 변환
    def ResolveMapPath(self, path: str) -> str:
        """패키지 maps/ 기준 상대 경로를 절대 경로로 변환."""
        full = path
        if not os.path.isabs(full):
            pkg_dir = os.path.dirname(os.path.dirname(__file__))
            base_dir = os.path.abspath(os.path.join(pkg_dir, "maps"))
            full = os.path.join(base_dir, path)
        return full

    # 평면 yaw(rad)를 z-회전만 가진 쿼터니언으로 변환
    def YawToQuaternion(self, yaw: float) -> Quaternion:
        """평면 yaw(rad)를 z축 회전만 가진 쿼터니언으로 변환."""
        half = yaw * 0.5
        return Quaternion(0.0, 0.0, math.sin(half), math.cos(half))

    # 차량 ObjectArray를 단순 CUBE MarkerArray로 변환
    def ObjectsToMarkers(self, msg: ObjectArray, group: str) -> MarkerArray:
        """차량 ObjectArray를 간단한 CUBE MarkerArray로 변환."""
        arr = MarkerArray()
        for obj in msg.objects:
            m = self.MakeMarker(ns=f"{group}", mid=obj.id, mtype=Marker.CUBE)
            m.pose = obj.pose
            m.scale.x, m.scale.y, m.scale.z = obj.size.x, obj.size.y, obj.size.z
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.2, 1.0 # Red
            arr.markers.append(m)
        return arr

    # 차량 시각화를 위한 Marker (CUBE) 생성 유틸리티
    def MakeMarker(self, ns: str, mid: int, mtype: int, frame: str = "map"):
        """차량 시각화를 위한 기본 Marker 생성."""
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
    rospy.init_node("env_surrounding_vehicle_node")

    # Create an instance of the EnvSurroundingVehiclePublisher class
    EnvSurroundingVehiclePublisher()

    # Log the start of the node
    rospy.loginfo("env_surrounding_vehicle_node started: publishing /env/surrounding_vehicles")
    rospy.spin()


if __name__ == "__main__":
    Main()
