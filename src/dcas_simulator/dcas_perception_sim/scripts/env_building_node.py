#!/usr/bin/env python
"""
Environment: Buildings (static)

This node publishes building ground-truth as dcas_msgs/ObjectArray and
their visualization as visualization_msgs/MarkerArray.

Input:
- CSV file specified by /map/buildings_csv, relative to package maps/ by default

Behavior:
- Preferred CSV schema: polygon per building (id,x,y[,z][,h])
- Legacy CSV schema: rectangle per row (cx,cy,w,d,h)

Visualization:
- For polygon buildings, a single TRIANGLE_LIST marker is created per object
  by extruding the 2D footprint vertically to a given height (walls + roof).
- For legacy rectangular records, a CUBE marker is used as fallback.

Performance notes:
- Markers use finite lifetime = period_s to keep RViz updated on reload.
- Keep headers/frame_id stable to avoid RViz flicker.
"""
import rospy
import os
import csv
import math
from dcas_msgs.msg import Object, ObjectArray
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Twist, Point32
from visualization_msgs.msg import Marker, MarkerArray


class EnvBuildingPublisher:
    """환경: 건물 퍼블리셔
    - CSV에서 건물 지상진실(ObjectArray)을 읽어 /env/buildings 로 퍼블리시합니다.
    - RViz 시각화를 위해 TRIANGLE_LIST(벽/지붕) 마커를 생성합니다.
    - 폴리곤 스키마 우선, 레거시 사각형 스키마는 폴백 처리합니다.
    """
    def __init__(self) -> None:
        # Parameters
        self.csv_path = rospy.get_param("/map/buildings_csv", "buildings.csv")
        period        = float(rospy.get_param("/env_building_node/period_s", 2.0))
        self.duration = period

        # Publishers
        self.publisher_building_info = rospy.Publisher("/env/buildings", ObjectArray, queue_size=10)
        self.publisher_building_viz  = rospy.Publisher("/viz/env/buildings", MarkerArray, queue_size=10)

        # Timer
        self.timer = rospy.Timer(rospy.Duration(period), self.CallbackTimer)

    # 주기 콜백
    # - CSV에서 건물 목록을 읽어 ObjectArray로 퍼블리시
    # - 동시에 RViz 마커도 갱신 퍼블리시
    def CallbackTimer(self, _evt) -> None:
        """주기마다 CSV를 읽어 건물 ObjectArray와 마커를 퍼블리시합니다.

        CSV가 비어 있거나 없으면 경고를 남기고, 빈 배열을 퍼블리시하여 구독자 측의 안정성을 보장합니다.
        """
        building_object_array                 = ObjectArray()
        building_object_array.header.stamp    = rospy.Time.now()
        building_object_array.header.frame_id = "map"

        # .csv에서 값을 읽어와 building_object_array에 값을 채움
        # b_is_loaded: bool type의 loading 완료 flag
        b_is_loaded = self.ReadBuildingsFromCsv(self.csv_path, building_object_array)

        if not b_is_loaded:
            rospy.logwarn("env_building_node: CSV file read failed: %s", self.csv_path)

        self.publisher_building_info.publish(building_object_array)
        building_marker_array = self.ObjectsToMarkers(building_object_array, group="buildings")
        self.publisher_building_viz.publish(building_marker_array)

    # CSV로부터 건물 정보를 읽어 ObjectArray에 채움
    # - 지원 스키마: (1) 폴리곤: id,x,y[,z][,h]  (2) 레거시 사각형: cx,cy,w,d,h
    # - 반환: 하나 이상 로드 시 True
    def ReadBuildingsFromCsv(self, csv_path: str, building_object_array: ObjectArray) -> bool:
        """CSV에서 건물 정보를 읽어 ObjectArray에 채웁니다.

        반환:
            하나 이상 로드되면 True.
        지원 스키마:
          1) 폴리곤: id,x,y[,z][,h]
          2) 레거시 사각형: cx,cy,w,d,h
        """
        # CSV 파일 경로 처리
        full_csv_path = csv_path
        
        # 상대 경로인 경우 패키지의 maps/ 폴더 기준으로 절대 경로 생성
        if not os.path.isabs(full_csv_path):
            pkg_dir       = os.path.dirname(os.path.dirname(__file__))
            base_dir      = os.path.abspath(os.path.join(pkg_dir, "maps"))
            full_csv_path = os.path.join(base_dir, csv_path)
        
        # 파일이 존재하지 않으면 False 반환
        if not os.path.exists(full_csv_path):
            return False

        # CSV 파일을 읽어서 건물 객체들을 생성합니다
        with open(full_csv_path, newline='') as f:
            reader = csv.DictReader(f)
            headers = [h.strip().lower() for h in (reader.fieldnames or [])]
            
            # 폴리곤 스키마: id,x,y,(선택사항 z),(선택사항 h)
            # 필수 컬럼인 id, x, y가 있는지 확인
            if {'id', 'x', 'y'}.issubset(set(headers)):
                id_to_points = {}    # 각 건물 ID별로 점들을 저장할 딕셔너리
                id_to_height = {}    # 각 건물 ID별로 높이를 저장할 딕셔너리
                
                # CSV의 모든 행을 읽어서 처리
                for row in reader:
                    try:
                        #===================================================#
                        #========= TODO: building 변수 채우기 =========#
                        #===================================================#
                        
                        building_id      = int(row.get('id'))          # 건물 ID
                        building_x_point = float(row.get('x'))         # X 좌표
                        building_y_point = float(row.get('y'))         # Y 좌표
                        building_z_point = float(row.get('z', 0.0))   # Z 좌표 (기본값: 0.0)
                        building_height  = float(row.get('h'))   # 높이 (기본값: 10.0)

                        #===================================================#
                        #===================================================#
                        #===================================================#


                    except Exception:
                        # 데이터 파싱 실패시 해당 행은 건너뜀
                        continue
                    
                    # 같은 ID의 건물에 점 추가 (폴리곤 구성을 위해)
                    id_to_points.setdefault(building_id, []).append(Point32(x=building_x_point, y=building_y_point, z=building_z_point))
                    
                    # 해당 건물 ID에 대한 높이 정보가 없으면 추가
                    if building_id not in id_to_height:
                        id_to_height[building_id] = building_height
                
                # 수집된 데이터로부터 각 건물에 대한 Object 생성
                for building_id in sorted(id_to_points.keys()):
                    building_points = id_to_points[building_id]
                    
                    # 폴리곤을 구성하려면 최소 3개의 점이 필요
                    if len(building_points) < 3:
                        continue
                    
                    # Object 메시지 생성 및 기본 정보 설정
                    building_obj = Object()
                    building_obj.header = building_object_array.header
                    
                    # 건물 point가 항상 시계 방향으로 정렬되어 있음을 보장
                    # 폴리곤의 중심점을 계산한 후 각 점의 atan2 각도를 기준으로 정렬,
                    # atan2는 반시계(CW 반대) 방향으로 증가하므로 reverse=True로 정렬하면 시계 방향 순서로 정렬
                    try:
                        cx_sort = sum(point.x for point in building_points) / len(building_points)
                        cy_sort = sum(point.y for point in building_points) / len(building_points)
                        building_points.sort(key=lambda p: math.atan2(p.y - cy_sort, p.x - cx_sort), reverse=True)
                    except Exception:
                        # 계산 실패시 원래 순서 유지
                        pass

                    # 폴리곤 점들로부터 건물 중심점(centroid) 계산
                    centroid_x = sum(building_point.x for building_point in building_points) / len(building_points)
                    centroid_y = sum(building_point.y for building_point in building_points) / len(building_points)
                    centroid_z = sum(building_point.z for building_point in building_points) / len(building_points)
                    
                    # 계산된 중심점으로 Pose 설정
                    building_obj.pose = Pose(
                        position=Point(x=centroid_x, y=centroid_y, z=centroid_z), 
                        orientation=Quaternion(0, 0, 0, 1)
                    )
                    
                    # 폴리곤 경계로부터 건물 크기(bounding box) 계산
                    min_x = min(building_point.x for building_point in building_points)
                    max_x = max(building_point.x for building_point in building_points)
                    min_y = min(building_point.y for building_point in building_points)
                    max_y = max(building_point.y for building_point in building_points)
                    building_width = max_x - min_x   # X 방향 너비
                    building_length = max_y - min_y  # Y 방향 길이
                    
                    # 계산된 크기 정보로 Vector3 설정
                    
                    #===================================================#
                    #========= TODO: object msg의 빈부분 채워넣기 =========#
                    #===================================================#
                    building_obj.geom_type = Object.GEOM_POLYGON
                    building_obj.id = building_id
                    building_obj.type = Object.LABEL_BUILDING
                    building_obj.footprint = building_points
                    building_obj.height = building_height
                    building_obj.size = Vector3(building_width, building_length, building_obj.height)
                    building_obj.twist = Twist()  # 정적 객체이므로 속도는 0
                    
                    #===================================================#
                    #===================================================#
                    #===================================================#
                    
                    # 완성된 건물 객체를 배열에 추가
                    building_object_array.objects.append(building_obj)
                
                # 하나 이상의 건물이 로드되었으면 True 반환
                return len(building_object_array.objects) > 0
            else:
                # 지원하지 않는 CSV 스키마인 경우 경고 출력
                rospy.logwarn("env_building_node: Unsupported CSV schema: %s", csv_path)
                return False

    # 빌딩 ObjectArray를 RViz MarkerArray로 변환
    # - 폴리곤 건물: 수직 압출된 TRIANGLE_LIST 생성
    # - 레거시 사각형: CUBE 폴백(간단 표현)
    def ObjectsToMarkers(self, building_object_array: ObjectArray, group: str) -> MarkerArray:
        """건물 ObjectArray를 RViz MarkerArray로 변환합니다.

        - 폴리곤 건물: 수직 압출된 TRIANGLE_LIST 생성
        - 레거시 사각형: CUBE 폴백
        """
        arr = MarkerArray()
        for building_object in building_object_array.objects:
            # Polygon buildings -> TRIANGLE_LIST with vertical extrusion
            h = max(0.1, getattr(building_object, 'height', building_object.size.z))
            tri = self.MakeMarker(ns=f"{group}_mesh", mid=building_object.id, mtype=Marker.TRIANGLE_LIST)
            tri.color.r, tri.color.g, tri.color.b, tri.color.a = 0.6, 0.6, 0.6, 1.0 # Color: Gray
            tri.scale.x = tri.scale.y = tri.scale.z = 1.0
            pts = [Point(x=p.x, y=p.y, z=p.z) for p in building_object.footprint]
            tri.points = self.ExtrudePolygonToTriangles(pts, h)
            arr.markers.append(tri)
            
        return arr

    # RViz 마커 공통 설정을 생성하는 유틸리티
    # - 입력: 네임스페이스(ns), 마커 ID(mid), 마커 타입(mtype), 프레임(frame)
    # - 출력: 기본 설정이 채워진 Marker 객체
    def MakeMarker(self, ns: str, mid: int, mtype: int, frame: str = "map"):
        """RViz 마커 기본 설정을 채운 Marker를 생성합니다.

        인자:
            ns: RViz 네임스페이스(관련 마커 묶음)
            mid: 네임스페이스 내에서 고유한 ID
            mtype: 마커 타입(TRIANGLE_LIST, CUBE 등)
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

    # 2D 폴리곤 외곽선(points)을 수직으로 높이(height)만큼 압출하여
    # 벽과 지붕을 이루는 TRIANGLE_LIST(삼각형 목록)로 변환
    def ExtrudePolygonToTriangles(self, points, height: float):
        """2D 폴리곤을 수직으로 압출하여 벽과 지붕을 TRIANGLE_LIST로 생성합니다.

        RViz에서 일관된 면 방향을 유지하도록 삼각형 정점 순서를 일정하게 구성합니다.
        """
        tri_points = []
        n = len(points)
        if n < 3:
            return tri_points

        def pt(p: Point, z: float) -> Point:
            return Point(x=p.x, y=p.y, z=z)

        # Walls: for each edge, build two triangles (quad split)
        for i in range(n):
            p0 = points[i]
            p1 = points[(i + 1) % n]
            b0 = pt(p0, 0.0)
            b1 = pt(p1, 0.0)
            t0 = pt(p0, height)
            t1 = pt(p1, height)
            tri_points.extend([b1, b0, t1])  # bottom edge -> upper triangle
            tri_points.extend([t0, t1, b0])  # bottom edge -> lower triangle

        # Roof: triangle fan around centroid to cover top face
        cx = sum(p.x for p in points) / n
        cy = sum(p.y for p in points) / n
        c_top = Point(x=cx, y=cy, z=height)
        for i in range(0, n - 1):
            tri_points.extend([c_top, pt(points[i+1], height), pt(points[i], height)])
        tri_points.extend([c_top, pt(points[0], height), pt(points[n - 1], height)])
        return tri_points


def Main() -> None:
    # Initialize the node
    rospy.init_node("env_building_node")

    # Create an instance of the EnvBuildingPublisher class
    EnvBuildingPublisher()

    # Log the start of the node
    rospy.loginfo("env_building_node started: publishing /env/buildings")
    rospy.spin()


if __name__ == "__main__":
    Main()


