#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point
from std_msgs.msg import ColorRGBA, Header
import lanelet2
from lanelet2.core import BasicPoint2d, GPSPoint
from lanelet2 import geometry as geom
import math
import os

class LaneletRvizRouting:
    def __init__(self):
        rospy.init_node("lanelet_rviz_routing")

        # ---------------- Params ----------------
        file_path = os.path.dirname(__file__)
        self.map_path = rospy.get_param("~map_path", os.path.join(file_path, "../lanelet2_maps", "Town02.osm"))  # (필수) Lanelet2 .osm 경로
        origin_lat = rospy.get_param("~origin_lat", -0.0018001081) # 타원 표면을 평면으로 근사하는 중심점이기 때문에 잘못 설정하면 m 오차 발생
        origin_lon = rospy.get_param("~origin_lon", -0.00008482343)
        self.fixed_frame = rospy.get_param("~frame_id", "map")
        self.draw_boundaries = rospy.get_param("~draw_boundaries", True)

        # RViz (0,0)을 origin으로: UTM + 오프셋=origin
        self.use_origin_as_zero = rospy.get_param("~use_origin_as_zero", True)
        self.auto_offset = rospy.get_param("~auto_offset", False)
        self.offset_x = rospy.get_param("~offset_x", 0.0)
        self.offset_y = rospy.get_param("~offset_y", 0.0)

        # 엔드포인트 마커 크기(기본 2.0m 지름 느낌)
        self.endpoint_scale = rospy.get_param("~endpoint_scale", 2.0)

        # --------------- Load Map ---------------
        self.projector = lanelet2.projection.UtmProjector( # Utm projector 사용한것
            lanelet2.io.Origin(origin_lat, origin_lon)
        )
        self.map_lanelet2 = lanelet2.io.load(self.map_path, self.projector)

        # --------------- Offset 선택 ------------
        if self.use_origin_as_zero:
            o = self.projector.forward(GPSPoint(origin_lat, origin_lon, 0.0))
            self.offset_x, self.offset_y = o.x, o.y
            rospy.loginfo("Offset = origin (%.6f, %.6f) -> (%.3f, %.3f)",
                          origin_lat, origin_lon, self.offset_x, self.offset_y)
        elif self.auto_offset:
            self.compute_auto_offset()
            rospy.loginfo("Auto offset (bbox center): (%.3f, %.3f)", self.offset_x, self.offset_y)
        else:
            rospy.loginfo("Manual offset: (%.3f, %.3f)", self.offset_x, self.offset_y)

        # ------------- Routing Graph ------------
        rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle
        )
        self.graph = lanelet2.routing.RoutingGraph(self.map_lanelet2, rules)

        # ----------------  Pub/Sub --------------
        self.publisher_map       = rospy.Publisher("lanelet_map_markers", MarkerArray, queue_size=1, latch=True)
        self.publisher_route     = rospy.Publisher("lanelet_route_markers", MarkerArray, queue_size=1)
        self.publisher_endpoint  = rospy.Publisher("lanelet_route_endpoints", MarkerArray, queue_size=1)
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.callback_initialpose)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_goalpoint)

        self.start_point_local = None  # RViz(로컬 좌표)
        self.goal_point_local  = None

        # 맵 1회 표시
        self.publish_map_markers()

        rospy.loginfo("Lanelet map loaded. In RViz: 2D Pose Estimate(start), 2D Nav Goal(goal).")
        rospy.spin()

    # ----------------- Helpers -----------------
    def compute_auto_offset(self):
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
        for lanelet in self.map_lanelet2.laneletLayer:
            for point in lanelet.centerline:
                if point.x < min_x: 
                    min_x = point.x
                if point.y < min_y: 
                    min_y = point.y
                if point.x > max_x: 
                    max_x = point.x
                if point.y > max_y: 
                    max_y = point.y
        if math.isfinite(min_x) and math.isfinite(min_y) and math.isfinite(max_x) and math.isfinite(max_y):
            self.offset_x = 0.5 * (min_x + max_x)
            self.offset_y = 0.5 * (min_y + max_y)
        else:
            self.offset_x, self.offset_y = 0.0, 0.0

    def to_local(self, x, y):
        """전역(투영좌표) -> 로컬(RViz 좌표)"""
        return x - self.offset_x, y - self.offset_y

    def to_global(self, x, y):
        """로컬(RViz 좌표) -> 전역(투영좌표)"""
        return x + self.offset_x, y + self.offset_y

    # ----------------- Callbacks ----------------
    def callback_initialpose(self, msg: PoseWithCovarianceStamped):
        initial_point = msg.pose.pose.position
        self.start_point_local = BasicPoint2d(initial_point.x, initial_point.y)
        self.publish_endpoints()     # ★ 즉시 엔드포인트 마커 표시
        self.routing()

    def callback_goalpoint(self, msg: PoseStamped):
        goal_point = msg.pose.position
        self.goal_point_local = BasicPoint2d(goal_point.x, goal_point.y)
        self.publish_endpoints()     # ★ 즉시 엔드포인트 마커 표시
        self.routing()

    # ------------------ Routing -----------------
    def routing(self):
        if self.start_point_local is None or self.goal_point_local is None:
            return

        global_start_point = BasicPoint2d(*self.to_global(self.start_point_local.x, self.start_point_local.y))
        global_goal_point  = BasicPoint2d(*self.to_global(self.goal_point_local.x,  self.goal_point_local.y))

        try:
            start_lanelet = geom.findNearest(self.map_lanelet2.laneletLayer, global_start_point, 1)[0][1]
            goal_lanelet  = geom.findNearest(self.map_lanelet2.laneletLayer, global_goal_point,  1)[0][1]
        except Exception as e:
            rospy.logwarn("Nearest lanelet failed: %s", e)
            self.publisher_route.publish(MarkerArray())
            return

        shortest_path = self.graph.shortestPath(start_lanelet, goal_lanelet)
        if shortest_path is None:
            rospy.logwarn("No path found between selected lanelets.")
            self.publisher_route.publish(MarkerArray())  # clear
            return

        # 경로 centerline(전역) → 로컬로 변환해 표시
        local_polyline = []
        for lanelet in shortest_path:
            for point in lanelet.centerline:
                x, y = self.to_local(point.x, point.y)
                local_polyline.append((x, y))

        self.publish_route_markers(local_polyline)
        self.publish_endpoints()  # 경로와 함께 엔드포인트도 갱신

    # --------------- Visualization --------------
    def publish_map_markers(self):
        marker_array = MarkerArray()
        header = Header(frame_id=self.fixed_frame, stamp=rospy.Time.now())

        # 1) 전체 centerline (연회색)
        marker = Marker(type=Marker.LINE_LIST, action=Marker.ADD)
        marker.header = header
        marker.ns = "lanelet_map"
        marker.id = 0
        marker.scale.x = 0.2
        marker.color = ColorRGBA(0.7, 0.7, 0.7, 0.9)

        for lanelet in self.map_lanelet2.laneletLayer:
            centerline = list(lanelet.centerline)
            for i in range(len(centerline) - 1):
                x1, y1 = self.to_local(centerline[i].x,  centerline[i].y)
                x2, y2 = self.to_local(centerline[i+1].x,centerline[i+1].y)
                marker.points.append(Point(x1, y1, 0.0))
                marker.points.append(Point(x2, y2, 0.0))
        marker_array.markers.append(marker)

        # 2) (옵션) 좌/우 경계
        if self.draw_boundaries:
            boundary_marker         = Marker(type=Marker.LINE_LIST, action=Marker.ADD)
            boundary_marker.header  = header
            boundary_marker.ns      = "lanelet_bounds"
            boundary_marker.id      = 1
            boundary_marker.scale.x = 0.15
            boundary_marker.color   = ColorRGBA(0.4, 0.4, 0.9, 0.6)

            for lanelet in self.map_lanelet2.laneletLayer:
                left_boundary  = list(lanelet.leftBound)
                right_boundary = list(lanelet.rightBound)
                for total_boundary_points in (left_boundary, right_boundary):
                    for i in range(len(total_boundary_points) - 1):
                        x1, y1 = self.to_local(total_boundary_points[i].x,   total_boundary_points[i].y)
                        x2, y2 = self.to_local(total_boundary_points[i+1].x, total_boundary_points[i+1].y)
                        boundary_marker.points.append(Point(x1, y1, 0.0))
                        boundary_marker.points.append(Point(x2, y2, 0.0))
            marker_array.markers.append(boundary_marker)

        self.publisher_map.publish(marker_array)

    def publish_route_markers(self, list_xy_local_route):
        marker_array = MarkerArray()
        header = Header(frame_id=self.fixed_frame, stamp=rospy.Time.now())

        # 경로 (초록 LINE_STRIP)
        route_maker = Marker(type=Marker.LINE_STRIP, action=Marker.ADD)
        route_maker.header = header
        route_maker.ns = "lanelet_route"
        route_maker.id = 100
        route_maker.scale.x = 0.35
        route_maker.color = ColorRGBA(0.0, 0.85, 0.25, 0.95)


        sample_step = 3
        for i in range(0, len(list_xy_local_route), sample_step):
            x, y = list_xy_local_route[i]
            route_maker.points.append(Point(x, y, 0.05))
        # 마지막 점이 누락되지 않게 추가
        if list_xy_local_route:
            x, y = list_xy_local_route[-1]
            route_maker.points.append(Point(x, y, 0.05))
        
        marker_array.markers.append(route_maker)
        self.publisher_route.publish(marker_array)

    def publish_endpoints(self):
        """시작/목표 마커를 즉시 갱신해서 표출"""
        marker_array = MarkerArray()
        header = Header(frame_id=self.fixed_frame, stamp=rospy.Time.now())

        # 시작/목표 포인트
        endpoints = [
            ("start", self.start_point_local, ColorRGBA(0.95, 0.25, 0.25, 0.98)),  # 빨강
            ("goal",  self.goal_point_local,  ColorRGBA(0.25, 0.45, 0.95, 0.98)),  # 파랑
        ]
        for idx, (name, point, color) in enumerate(endpoints):
            if point is None:
                continue
            endpoint_marker               = Marker(type=Marker.SPHERE, action=Marker.ADD)
            endpoint_marker.header        = header
            endpoint_marker.ns            = "route_endpoints"
            endpoint_marker.id            = 200 + idx
            endpoint_marker.scale.x       = endpoint_marker.scale.y = endpoint_marker.scale.z = self.endpoint_scale   # ★ 크게
            endpoint_marker.color         = color
            endpoint_marker.pose.position = Point(point.x, point.y, self.endpoint_scale * 0.1)
            marker_array.markers.append(endpoint_marker)

        self.publisher_endpoint.publish(marker_array)

if __name__ == "__main__":
    LaneletRvizRouting()
