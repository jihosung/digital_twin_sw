#!/usr/bin/env python
import random
import struct
import numpy as np
import time
from typing import List, Tuple
import math

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from dcas_msgs.msg import ObjectArray, LaneArray, VehicleState, TrafficSignArray

"""
LiDAR 센서 시뮬레이터 메인 함수

알고리즘 프로세스 흐름:
1. ROS 노드 초기화 (rospy.init_node)
2. LidarEmu 클래스 인스턴스 생성 (__init__ 호출)
    - 파라미터 로드 및 설정
    - ROS 퍼블리셔/구독자 설정
    - 타이머 생성 (주기적 CallbackTimer 실행)
3. rospy.spin()으로 이벤트 루프 시작

타이머 콜백(CallbackTimer)에서 실행되는 주요 프로세스:
1. GenerateRingBasedScan: 링 기반 라이다 스캔 생성
   - 각 링(고도각)과 방위각 조합으로 레이 캐스팅 수행
   - CastRay: 개별 광선에 대한 교차점 계산
   - ApplyNoise: 가우시안 노이즈 적용
   - GenerateClutterPoints: 클러터(허위 양성) 포인트 생성
2. CreateXyziPointcloud2: PointCloud2 메시지 생성 및 퍼블리시

각 프로세스에서 공통적으로 수행되는 작업:
- 좌표계 변환 (MapToSensor): 월드 프레임 → 센서 프레임
- 레이 캐스팅: 지면, 건물, 차량, 가드레일, 표지판, 차선과의 교차점 계산
- 강도 계산 (CalculateIntensityForPoint): 물체 타입과 거리 기반 반사 강도 계산
- 노이즈 적용 (ApplyNoise): 구면좌표계에서 가우시안 노이즈 추가
- 클러터 생성: 허위 양성 감지 포인트 시뮬레이션
"""

# PointCloud2 변환 - LiDAR 포인트 리스트를 ROS PointCloud2 메시지로 변환
def CreateXyziPointcloud2(points: List[Tuple[float, float, float, float]], frame_id: str = "lidar") -> PointCloud2:
    header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
    ]
    data = b''.join([struct.pack('ffff', p[0], p[1], p[2], p[3]) for p in points])
    return PointCloud2(
        header=header,
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=16,
        row_step=16 * len(points),
        is_dense=True,
        data=data,
    )

class LidarEmu:
    """Simple LiDAR emulator that samples points from environment topics.

    Subscribes to lanes and generic objects and produces a synthetic PointCloud2.
    This uses coarse sampling (no true ray casting) for educational wiring.
    """

    # ===============================
    # 1. 초기화 및 콜백 함수들
    # ===============================

    # LiDAR 에뮬레이터 초기화 - 파라미터 로드, 퍼블리셔/구독자 설정, 타이머 생성
    def __init__(self) -> None:
        # Check if sensor is enabled
        self.enabled = rospy.get_param("/sensor_lidar_node/enabled", True)
        if not self.enabled:
            rospy.loginfo("LiDAR sensor is disabled")
            return
        
        # Publishers
        self.pub = rospy.Publisher("/sensors/lidar/points", PointCloud2, queue_size=10)  # GT points
        self.pub_noise = rospy.Publisher("/sensors/lidar/points_noise", PointCloud2, queue_size=10)  # Noisy points
        
        # Subscribers
        rospy.Subscriber("/env/lanes", LaneArray, self.CallbackLanes)
        rospy.Subscriber("/env/buildings", ObjectArray, self.CallbackBuildings)
        rospy.Subscriber("/env/guardrails", ObjectArray, self.CallbackGuardrails)
        rospy.Subscriber("/env/surrounding_vehicles", ObjectArray, self.CallbackVehicles)
        rospy.Subscriber("/env/traffic_signs", TrafficSignArray, self.CallbackSigns)
        rospy.Subscriber("/vehicle/state", VehicleState, lambda m: setattr(self, 'state', m))
        
        # Inputs
        self.lanes: LaneArray = None  # type: ignore
        self.objects: ObjectArray = None  # type: ignore
        self.buildings: ObjectArray = None  # type: ignore
        self.guardrails: ObjectArray = None  # type: ignore
        self.vehicles: ObjectArray = None  # type: ignore
        self.signs: TrafficSignArray = None  # type: ignore
        self.state: VehicleState = None  # type: ignore
        
        # Parameters
        period = float(rospy.get_param("/sensor_lidar_node/period_s", 0.1))
        
        # LiDAR extrinsics and ROI
        self.sensor_tx = rospy.get_param("/sensor_lidar_node/offset_x", 1.2)
        self.sensor_ty = rospy.get_param("/sensor_lidar_node/offset_y", 0.0)
        self.sensor_tz = rospy.get_param("/sensor_lidar_node/offset_z", 1.5)
        self.sensor_roll_deg = rospy.get_param("/sensor_lidar_node/roll_deg", 0.0)
        self.sensor_pitch_deg = rospy.get_param("/sensor_lidar_node/pitch_deg", 0.0)
        self.sensor_yaw_deg = rospy.get_param("/sensor_lidar_node/yaw_deg", 0.0)
        self.roi_radius = rospy.get_param("/sensor_lidar_node/radius", 50.0)
        
        # Ring-based LiDAR parameters
        self.num_rings = rospy.get_param("/sensor_lidar_node/num_rings", 16)  # Number of vertical rings (vertical Channel)
        self.horizontal_resolution = rospy.get_param("/sensor_lidar_node/horizontal_resolution", 2.0)  # degrees per ray (더 sparse하게)
        self.vertical_fov_min = rospy.get_param("/sensor_lidar_node/vertical_fov_min", -15.0)  # degrees
        self.vertical_fov_max = rospy.get_param("/sensor_lidar_node/vertical_fov_max", 15.0)   # degrees
        self.rotation_frequency = rospy.get_param("/sensor_lidar_node/rotation_frequency", 10.0)  # Hz
        
        # Calculate ring elevation angles
        self.ring_elevations = np.linspace(self.vertical_fov_min, self.vertical_fov_max, self.num_rings)
        
        # Rotation state
        self.current_angle = 0.0  # Current rotation angle in degrees
        self.angle_increment = 360.0 * self.rotation_frequency * period  # degrees per timer callback
        
        # Gaussian noise parameters
        self.range_noise_std = rospy.get_param("/sensor_lidar_node/range_noise_std", 0.02)  # meters
        self.azimuth_noise_std = rospy.get_param("/sensor_lidar_node/azimuth_noise_std", 0.1)  # degrees
        self.elevation_noise_std = rospy.get_param("/sensor_lidar_node/elevation_noise_std", 0.1)  # degrees
        self.enable_noise = rospy.get_param("/sensor_lidar_node/enable_noise", True)
        
        # Clutter noise parameters
        self.enable_clutter = rospy.get_param("/sensor_lidar_node/enable_clutter", True)
        self.clutter_probability = rospy.get_param("/sensor_lidar_node/clutter_probability", 0.1)
        self.clutter_density = rospy.get_param("/sensor_lidar_node/clutter_density", 5.0)
        self.clutter_range_min = rospy.get_param("/sensor_lidar_node/clutter_range_min", 2.0)
        self.clutter_range_max = rospy.get_param("/sensor_lidar_node/clutter_range_max", 40.0)
        
        # Load shared sensor parameters (LiDAR needs intensity and reflectivity for power calculation)
        self.object_props = {
            'vehicle': {
                'base_intensity': rospy.get_param('/sensor_shared_params/object_properties/vehicle/base_intensity', 160),
                'reflectivity': rospy.get_param('/sensor_shared_params/object_properties/vehicle/reflectivity', 0.3)
            },
            'building': {
                'base_intensity': rospy.get_param('/sensor_shared_params/object_properties/building/base_intensity', 180),
                'reflectivity': rospy.get_param('/sensor_shared_params/object_properties/building/reflectivity', 0.8)
            },
            'guardrail': {
                'base_intensity': rospy.get_param('/sensor_shared_params/object_properties/guardrail/base_intensity', 190),
                'reflectivity': rospy.get_param('/sensor_shared_params/object_properties/guardrail/reflectivity', 0.9)
            },
            'traffic_sign': {
                'base_intensity': rospy.get_param('/sensor_shared_params/object_properties/traffic_sign/base_intensity', 230),
                'reflectivity': rospy.get_param('/sensor_shared_params/object_properties/traffic_sign/reflectivity', 0.7)
            },
            'ground': {
                'base_intensity': rospy.get_param('/sensor_shared_params/object_properties/ground/base_intensity', 70),
                'reflectivity': rospy.get_param('/sensor_shared_params/object_properties/ground/reflectivity', 0.1)
            },
            'lane_marking': {
                'base_intensity': rospy.get_param('/sensor_shared_params/object_properties/lane_marking/base_intensity', 120),
                'reflectivity': rospy.get_param('/sensor_shared_params/object_properties/lane_marking/reflectivity', 0.8)
            }
        }
        
        # LiDAR power calculation constants
        self.laser_power_watts = 0.001  # 1 mW typical laser power
        self.alpha_constant = 1e10      # Scaling constant for intensity calculation (tuned for realistic range)
        
        # Clutter parameters
        self.clutter_intensity_min = rospy.get_param('/sensor_shared_params/clutter/intensity_min', 10)
        self.clutter_intensity_max = rospy.get_param('/sensor_shared_params/clutter/intensity_max', 50)
        
        # Random number generator for reproducible noise
        noise_seed = rospy.get_param("/sensor_lidar_node/noise_seed", None)
        if noise_seed is not None:
            self.rng = np.random.RandomState(noise_seed)
        else:
            self.rng = np.random.RandomState(int(time.time() * 1000) % 2**32)

        
        # Timer (only create if sensor is enabled)
        if self.enabled:
            self.timer = rospy.Timer(rospy.Duration(period), self.CallbackTimer)
        
        rospy.loginfo(f"Ring-based LiDAR initialized: {self.num_rings} rings, elevations: {self.vertical_fov_min}° to {self.vertical_fov_max}°")

    # 차선 정보 수신 콜백 - 차선 데이터 저장
    def CallbackLanes(self, msg: LaneArray) -> None:
        self.lanes = msg

    # 건물 정보 수신 콜백 - 건물 데이터 저장
    def CallbackBuildings(self, msg: ObjectArray) -> None:
        self.buildings = msg

    # 가드레일 정보 수신 콜백 - 가드레일 데이터 저장
    def CallbackGuardrails(self, msg: ObjectArray) -> None:
        self.guardrails = msg

    # 주변 차량 정보 수신 콜백 - 차량 데이터 저장
    def CallbackVehicles(self, msg: ObjectArray) -> None:
        self.vehicles = msg

    # 교통표지판 정보 수신 콜백 - 표지판 데이터 저장
    def CallbackSigns(self, msg: TrafficSignArray) -> None:
        self.signs = msg

    # ===============================
    # 2. 메인 타이머 콜백 및 스캔 생성 함수들
    # ===============================

    # 메인 타이머 콜백 - LiDAR 데이터 처리 및 퍼블리시 (주기적 실행)
    def CallbackTimer(self, _) -> None:
        """Generate ring-based LiDAR point cloud using ray casting."""
        if not self.enabled:
            return
        
        # Generate points using ring-based ray casting
        points_gt, points_noise = self.GenerateRingBasedScan()
        
        # Publish both GT and noisy point clouds
        if points_gt:
            self.pub.publish(CreateXyziPointcloud2(points_gt))
        if points_noise:
            self.pub_noise.publish(CreateXyziPointcloud2(points_noise, frame_id="lidar"))
        
        # Update rotation angle for next scan
        self.current_angle = (self.current_angle + self.angle_increment) % 360.0

    # 링 기반 스캔 생성 - 각 링과 방위각 조합으로 레이 캐스팅 수행
    # TODO: 레이 캐스팅 - 레이 캐스팅 실습
    def GenerateRingBasedScan(self) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]]:
        """Generate ring-based LiDAR scan using ray casting for each ring."""
        points_gt = []
        points_noise = []
        
        # Calculate angular range for this scan
        start_angle = self.current_angle
        end_angle = (self.current_angle + self.angle_increment) % 360.0

        #############################################################
        # Generate azimuth angles for this scan sector
        if end_angle > start_angle:
            azimuth_angles = np.arange(start_angle, end_angle, self.horizontal_resolution)
        else:  # Handle wrap-around case
            azimuth_angles = np.concatenate([
                np.arange(start_angle, 360.0, self.horizontal_resolution),
                np.arange(0.0, end_angle, self.horizontal_resolution)
            ])
        #############################################################

        # For each ring (elevation angle)
        for _, elevation_deg in enumerate(self.ring_elevations):
            # For each azimuth angle in this scan
            for azimuth_deg in azimuth_angles:
                #############################################################
                # TODO: 레이 캐스팅 - 레이 캐스팅 실습
                # Cast ray for this (azimuth, elevation) combination
                # Hint: CastRay 함수를 호출하여 광선이 충돌한 점 찾기
                hit_point = self.CastRay(azimuth_deg, elevation_deg)  # 실습: self.CastRay(입력인자 1, 입력인자 2) 호출
                #############################################################

                if hit_point is not None:
                    points_gt.append(hit_point)

                    #############################################################
                    # TODO: 노이즈 적용 - 노이즈 적용 실습
                    # Add noisy point with same intensity (noise doesn't affect material properties)
                    # Hint: ApplyNoise 함수를 사용하여 x, y, z 좌표에 노이즈 적용
                    noisy_point = self.ApplyNoise(hit_point[0], hit_point[1], hit_point[2])  # TODO: ApplyNoise 함수 구현 ,실습: ApplyNoise 함수 호출
                    points_noise.append(noisy_point + (hit_point[3],))  # Keep original intensity
                    #############################################################

        # Add clutter noise to both GT and noise point clouds
        if self.enable_clutter:
            clutter_points = self.GenerateClutterPoints()
            # Add random intensity to clutter points
            clutter_with_intensity = []
            for pt in clutter_points:
                random_intensity = int(self.rng.uniform(self.clutter_intensity_min, self.clutter_intensity_max))
                clutter_with_intensity.append(pt + (random_intensity,))
            points_gt.extend(clutter_with_intensity)
            
            # Add noise to clutter points for noise point cloud
            clutter_noise_points = []
            for pt in clutter_points:
                random_intensity = int(self.rng.uniform(self.clutter_intensity_min, self.clutter_intensity_max))
                clutter_noise_points.append(self.ApplyNoise(*pt) + (random_intensity,))
            points_noise.extend(clutter_noise_points)
        
        return points_gt, points_noise

    # ===============================
    # 3. 레이 캐스팅 및 교차점 계산 함수들
    # ===============================

    # 단일 레이 캐스팅 - 주어진 방위각과 고도각으로 광선을 투사하여 가장 가까운 교차점 찾기
    # TODO: 레이 캐스팅 - 레이 캐스팅 실습
    def CastRay(self, azimuth_deg: float, elevation_deg: float) -> Tuple[float, float, float, int]:
        """Cast a single ray and find the closest intersection with performance optimizations.
        
        Returns:
            Tuple of (x, y, z, intensity) where intensity is calculated based on hit object type
        """
        # Convert angles to radians
        azimuth_rad = math.radians(azimuth_deg)
        elevation_rad = math.radians(elevation_deg)
        
        # Ray direction in sensor coordinate system
        # x: forward, y: left, z: up
        cos_azimuth = math.cos(azimuth_rad)
        sin_azimuth = math.sin(azimuth_rad)
        cos_elevation = math.cos(elevation_rad)
        sin_elevation = math.sin(elevation_rad)
        
        # Ray direction vector (normalized)
        ray_direction_x = cos_elevation * cos_azimuth  # forward component
        ray_direction_y = cos_elevation * sin_azimuth  # left component  
        ray_direction_z = sin_elevation                # up component
        
        # Find closest intersection with early termination for performance
        closest_distance = self.roi_radius  # Use ROI as max distance
        closest_point = None
        hit_object_type = 'ground'  # Default object type
        
        #############################################################
        # TODO: 레이 캐스팅 - 레이 캐스팅 실습
        # Check intersection with ground plane
        # Hint: 지면과의 교점 계산
        ground_hit = self.IntersectRayWithGround(ray_direction_x, ray_direction_y, ray_direction_z)  # 실습: IntersectRayWithGround 함수 호출
        #############################################################

        if ground_hit is not None:
            hit_distance = math.sqrt(sum(p**2 for p in ground_hit))
            if hit_distance <= self.roi_radius and hit_distance < closest_distance:
                closest_distance = hit_distance
                closest_point = ground_hit
                hit_object_type = 'ground'
        
        # Early termination: if ground is very close, skip other objects for performance
        ground_proximity_threshold = 2.0  # meters
        if closest_distance < ground_proximity_threshold:
            if closest_point is not None:
                intensity = self.CalculateIntensityForPoint(closest_point[0], closest_point[1], closest_point[2], azimuth_deg, elevation_deg, hit_object_type)
                return (closest_point[0], closest_point[1], closest_point[2], intensity)
            return None
        
        # Check intersections with buildings (usually largest objects, check first)
        if self.buildings:
            for building in self.buildings.objects:
                building_hit = self.IntersectRayWithBuilding(ray_direction_x, ray_direction_y, ray_direction_z, building)
                if building_hit is not None:
                    hit_distance = math.sqrt(sum(p**2 for p in building_hit))
                    if hit_distance <= self.roi_radius and hit_distance < closest_distance:
                        closest_distance = hit_distance
                        closest_point = building_hit
                        hit_object_type = 'building'
                        # Option: Early termination if very close building found
                        close_building_threshold = 5.0  # meters
                        if hit_distance < close_building_threshold:
                            break

        # Check intersections with vehicles (only if no close building found)
        vehicle_check_threshold = 3.0  # meters
        if self.vehicles and closest_distance > vehicle_check_threshold:
            #############################################################
            # TODO: 레이 캐스팅 - 레이 캐스팅 실습
            # Check intersections with vehicles
            # Hint: 각 차량에 대해 광선-박스 교점 계산, building_hit과 유사
            # 실습: 차량들과의 교점 검사 루프 구현
            if self.vehicles and closest_distance > vehicle_check_threshold:
                for vehicle in self.vehicles.objects:
                    # 실습: IntersectRayWithBox 함수 호출하여 교점 계산
                    vehicle_hit = self.IntersectRayWithBox(ray_direction_x, ray_direction_y, ray_direction_z, vehicle)
                    if vehicle_hit is not None:
                        hit_distance = math.sqrt(sum(p**2 for p in vehicle_hit))
                        if hit_distance <= self.roi_radius and hit_distance < closest_distance:
                            closest_distance = hit_distance
                            closest_point = vehicle_hit
                            hit_object_type = 'vehicle'
            #############################################################
        
        # Check intersections with guardrails (only if no close objects found)
        guardrail_check_threshold = 2.0  # meters
        if self.guardrails and closest_distance > guardrail_check_threshold:
            #############################################################
            # TODO: 레이 캐스팅 - 레이 캐스팅 실습
            # Check intersections with guardrails
            # Hint: 각 가드레일에 대해 광선-박스 교점 계산
            # 실습: 가드레일들과의 교점 검사 루프 구현
            if self.guardrails and closest_distance > guardrail_check_threshold:
                for guardrail in self.guardrails.objects:
                    # 실습: IntersectRayWithBox 함수 호출하여 교점 계산
                    guardrail_hit = self.IntersectRayWithBox(ray_direction_x, ray_direction_y, ray_direction_z, guardrail)
                    if guardrail_hit is not None:
                        hit_distance = math.sqrt(sum(p**2 for p in guardrail_hit))
                        if hit_distance <= self.roi_radius and hit_distance < closest_distance:
                            closest_distance = hit_distance
                            closest_point = guardrail_hit
                            hit_object_type = 'guardrail'
            #############################################################

        # Check intersections with signs (only if no close objects found)
        sign_check_threshold = 2.0  # meters
        if self.signs and closest_distance > sign_check_threshold:
            #############################################################
            # TODO: 레이 캐스팅 - 레이 캐스팅 실습
            # Check intersections with traffic signs
            # Hint: 각 교통표지판에 대해 광선-표지판 교점 계산
            # 실습: 교통표지판들과의 교점 검사 루프 구현
            if self.signs and closest_distance > sign_check_threshold:
                for sign in self.signs.signs:
                    # 실습: IntersectRayWithSign 함수 호출하여 교점 계산
                    sign_hit = self.IntersectRayWithSign(ray_direction_x, ray_direction_y, ray_direction_z, sign)
                    if sign_hit is not None:
                        hit_distance = math.sqrt(sum(p**2 for p in sign_hit))
                        if hit_distance <= self.roi_radius and hit_distance < closest_distance:
                            closest_distance = hit_distance
                            closest_point = sign_hit
                            hit_object_type = 'traffic_sign'
            #############################################################

        # Check intersections with lane markings (only if no close objects found)
        lane_check_threshold = 1.0  # meters
        if self.lanes and closest_distance > lane_check_threshold:
            #############################################################
            # TODO: 레이 캐스팅 - 레이 캐스팅 실습
            # Check intersections with lane markings
            # Hint: 차선과의 교점 계산
            # 실습: IntersectRayWithLanes 함수 호출 및 교점 검사
            if self.lanes and closest_distance > lane_check_threshold:
                # 실습: IntersectRayWithLanes 함수 호출하여 교점 계산
                lanes_hit = self.IntersectRayWithLanes(ray_direction_x, ray_direction_y, ray_direction_z)
                if lanes_hit is not None:
                    hit_distance = math.sqrt(sum(p**2 for p in lanes_hit))
                    if hit_distance <= self.roi_radius and hit_distance < closest_distance:
                        closest_distance = hit_distance
                        closest_point = lanes_hit
                        hit_object_type = 'lane_marking'
            #############################################################

        if closest_point is not None:
            intensity = self.CalculateIntensityForPoint(closest_point[0], closest_point[1], closest_point[2], azimuth_deg, elevation_deg, hit_object_type)
            return (closest_point[0], closest_point[1], closest_point[2], intensity)
        return None

    # 폴리곤 내부 점 검사 - 레이 캐스팅 알고리즘을 사용한 점-폴리곤 포함 관계 검사
    def PointInPolygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    # 지면 교차점 계산 - 광선과 지면 평면(z=0)의 교차점 찾기
    def IntersectRayWithGround(self, ray_direction_x: float, ray_direction_y: float, ray_direction_z: float) -> Tuple[float, float, float]:
        """Find intersection of ray with ground plane (z=0 in world coordinates)."""
        # Ray origin in sensor coordinates is (0, 0, 0)
        # Ground plane is at z = -sensor_tz in sensor coordinates
        
        if abs(ray_direction_z) < 1e-6:  # Ray is parallel to ground
            return None
        
        # Distance along ray to hit ground plane
        ray_parameter = -self.sensor_tz / ray_direction_z
        
        if ray_parameter <= 0:  # Intersection is behind sensor
            return None
        
        # Intersection point in sensor coordinates
        intersection_x = ray_parameter * ray_direction_x
        intersection_y = ray_parameter * ray_direction_y
        intersection_z = ray_parameter * ray_direction_z
        
        return (intersection_x, intersection_y, intersection_z)

    # 건물 교차점 계산 - 광선과 건물의 교차점 찾기 (AABB 및 폴리곤 지원)
    def IntersectRayWithBuilding(self, ray_direction_x: float, ray_direction_y: float, ray_direction_z: float, building) -> Tuple[float, float, float]:
        """Find intersection of ray with building."""
        if self.state:
            building_world_pos = (building.pose.position.x, building.pose.position.y)
            sensor_world_pos = (self.state.pose.position.x + self.sensor_tx, 
                              self.state.pose.position.y + self.sensor_ty)
            building_distance = math.sqrt((building_world_pos[0] - sensor_world_pos[0])**2 + 
                                        (building_world_pos[1] - sensor_world_pos[1])**2)
            # Skip buildings that are too far (beyond sensor range + building size)
            max_building_size = max(building.size.x, building.size.y)
            if building_distance > self.roi_radius + max_building_size:
                return None
        
        # Use AABB test first for quick rejection
        aabb_hit = self.IntersectRayWithBox(ray_direction_x, ray_direction_y, ray_direction_z, building)
        if aabb_hit is None:
            return None  # No intersection with bounding box
        
        # For polygon buildings, only do detailed intersection if AABB hit exists
        if hasattr(building, 'footprint') and len(building.footprint) >= 3:
            # Use polygon intersection
            return self.IntersectRayWithPolygonBuilding(ray_direction_x, ray_direction_y, ray_direction_z, building)
        else:
            # Simple box buildings - AABB result is sufficient
            return aabb_hit

    # 폴리곤 건물 교차점 계산 - 복잡한 폴리곤 건물과 광선의 교차점 계산 (벽면 및 지붕)
    def IntersectRayWithPolygonBuilding(self, ray_direction_x: float, ray_direction_y: float, ray_direction_z: float, building) -> Tuple[float, float, float]:
        """Polygon building intersection - fast 2D approach."""
        # Get building footprint and height
        footprint_world = [(p.x, p.y) for p in building.footprint]
        base_z = building.pose.position.z
        height = getattr(building, 'height', building.size.z)
        top_z = base_z + height
        
        closest_intersection = None
        closest_distance = float('inf')
        
        # Method 1: 2D wall intersection check
        for i in range(len(footprint_world)):
            p1_world = footprint_world[i]
            p2_world = footprint_world[(i + 1) % len(footprint_world)]
            
            # Wall direction check - skip walls facing away from sensor
            wall_direction_x = p2_world[0] - p1_world[0]
            wall_direction_y = p2_world[1] - p1_world[1]
            wall_center_x = (p1_world[0] + p2_world[0]) * 0.5
            wall_center_y = (p1_world[1] + p2_world[1]) * 0.5
            
            if self.state:
                sensor_world_x = self.state.pose.position.x + self.sensor_tx
                sensor_world_y = self.state.pose.position.y + self.sensor_ty
                to_sensor_x = sensor_world_x - wall_center_x
                to_sensor_y = sensor_world_y - wall_center_y
                
                # Wall normal (perpendicular to wall, pointing outward)
                wall_normal_x = -wall_direction_y  # Perpendicular vector
                wall_normal_y = wall_direction_x
                
                # Check if wall faces sensor (dot product > 0)
                if (wall_normal_x * to_sensor_x + wall_normal_y * to_sensor_y) <= 0:
                    continue  # Wall faces away from sensor, skip
            
            # 2D line intersection
            wall_hit = self.IntersectRayWithVerticalWall2D(ray_direction_x, ray_direction_y, ray_direction_z, p1_world, p2_world, base_z, top_z)
            if wall_hit is not None:
                wall_distance = math.sqrt(sum(p**2 for p in wall_hit))
                if wall_distance < closest_distance:
                    closest_distance = wall_distance
                    closest_intersection = wall_hit
        
        # Method 2: Check roof intersection (only if ray points upward)
        if ray_direction_z > 0:
            roof_hit = self.IntersectRayWithPolygonRoof(ray_direction_x, ray_direction_y, ray_direction_z, footprint_world, top_z)
            if roof_hit is not None:
                roof_distance = math.sqrt(sum(p**2 for p in roof_hit))
                if roof_distance < closest_distance:
                    closest_distance = roof_distance
                    closest_intersection = roof_hit
        
        return closest_intersection

    # 수직 벽면 교차점 계산 - 광선과 수직 벽면의 2D 교차점 계산 (센서 좌표계 기준)
    def IntersectRayWithVerticalWall2D(self, ray_direction_x: float, ray_direction_y: float, ray_direction_z: float,
                                     p1_world: Tuple[float, float], p2_world: Tuple[float, float],
                                     base_z: float, top_z: float) -> Tuple[float, float, float]:
        """Sensor coordinate system wall intersection for accuracy."""
        if self.state is None:
            return None
        
        # Transform wall endpoints to sensor coordinates
        p1_sensor = self.MapToSensor(p1_world[0], p1_world[1], base_z)
        p2_sensor = self.MapToSensor(p2_world[0], p2_world[1], base_z)
        base_z_sensor = self.MapToSensor(0, 0, base_z)[2]
        top_z_sensor = self.MapToSensor(0, 0, top_z)[2]
        
        # Wall vector in sensor coordinates (2D projection)
        wall_direction_x = p2_sensor[0] - p1_sensor[0]
        wall_direction_y = p2_sensor[1] - p1_sensor[1]
        
        # Ray origin is (0, 0) in sensor coordinates
        # 2D line-line intersection in sensor coordinate system
        denominator = ray_direction_x * wall_direction_y - ray_direction_y * wall_direction_x
        if abs(denominator) < 1e-6:  # Parallel lines
            return None
        
        # Vector from ray origin (0,0) to wall start
        to_wall_x = p1_sensor[0]
        to_wall_y = p1_sensor[1]
        
        # Calculate intersection parameters
        ray_parameter = (to_wall_x * wall_direction_y - to_wall_y * wall_direction_x) / denominator
        wall_parameter = (to_wall_x * ray_direction_y - to_wall_y * ray_direction_x) / denominator
        
        # Check if intersection is valid
        if ray_parameter <= 0 or wall_parameter < 0 or wall_parameter > 1:  # Behind sensor or outside wall segment
            return None
        
        # Calculate intersection point in sensor coordinates
        intersection_x = ray_parameter * ray_direction_x
        intersection_y = ray_parameter * ray_direction_y
        intersection_z = ray_parameter * ray_direction_z
        
        # Check height bounds in sensor coordinates
        if intersection_z < base_z_sensor or intersection_z > top_z_sensor:
            return None
        
        return (intersection_x, intersection_y, intersection_z)

    # 폴리곤 지붕 교차점 계산 - 광선과 폴리곤 지붕의 교차점 계산 (2D 점-폴리곤 테스트 사용)
    def IntersectRayWithPolygonRoof(self, ray_direction_x: float, ray_direction_y: float, ray_direction_z: float,
                                   footprint_world: List[Tuple[float, float]], z_world: float) -> Tuple[float, float, float]:
        """Fast roof intersection using 2D point-in-polygon test."""
        if abs(ray_direction_z) < 1e-6:  # Ray parallel to horizontal plane
            return None
        
        # Calculate intersection with horizontal plane
        z_sensor = self.MapToSensor(0, 0, z_world)[2]
        ray_parameter = z_sensor / ray_direction_z
        
        if ray_parameter <= 0:  # Intersection behind sensor
            return None
        
        # Intersection point in sensor coordinates
        intersection_x_sensor = ray_parameter * ray_direction_x
        intersection_y_sensor = ray_parameter * ray_direction_y
        intersection_z_sensor = ray_parameter * ray_direction_z
        
        # Convert to world coordinates for polygon test
        world_hit = self.SensorToWorld(intersection_x_sensor, intersection_y_sensor, intersection_z_sensor)
        
        # Fast point-in-polygon test
        if self.PointInPolygon(world_hit[0], world_hit[1], footprint_world):
            return (intersection_x_sensor, intersection_y_sensor, intersection_z_sensor)
        
        return None

    # AABB 교차점 계산 - 광선과 축 정렬 바운딩 박스(AABB)의 교차점 찾기
    def IntersectRayWithBox(self, ray_direction_x: float, ray_direction_y: float, ray_direction_z: float, obj) -> Tuple[float, float, float]:
        """Find intersection of ray with axis-aligned bounding box."""
        # Transform object center to sensor coordinates
        obj_center_world = (obj.pose.position.x, obj.pose.position.y, obj.pose.position.z)
        obj_center_sensor = self.MapToSensor(*obj_center_world)
        
        # Object dimensions
        half_size_x = obj.size.x * 0.5
        half_size_y = obj.size.y * 0.5
        half_size_z = obj.size.z * 0.5
        
        # Bounding box in sensor coordinates (simplified as axis-aligned)
        box_min_x = obj_center_sensor[0] - half_size_x
        box_max_x = obj_center_sensor[0] + half_size_x
        box_min_y = obj_center_sensor[1] - half_size_y
        box_max_y = obj_center_sensor[1] + half_size_y
        box_min_z = obj_center_sensor[2] - half_size_z
        box_max_z = obj_center_sensor[2] + half_size_z
        
        # Ray-box intersection (AABB)
        ray_param_min = float('-inf')
        ray_param_max = float('inf')
        
        # X slab
        if abs(ray_direction_x) > 1e-6:
            ray_param_1 = box_min_x / ray_direction_x
            ray_param_2 = box_max_x / ray_direction_x
            ray_param_min = max(ray_param_min, min(ray_param_1, ray_param_2))
            ray_param_max = min(ray_param_max, max(ray_param_1, ray_param_2))
        elif 0 < box_min_x or 0 > box_max_x:
            return None
        
        # Y slab
        if abs(ray_direction_y) > 1e-6:
            ray_param_1 = box_min_y / ray_direction_y
            ray_param_2 = box_max_y / ray_direction_y
            ray_param_min = max(ray_param_min, min(ray_param_1, ray_param_2))
            ray_param_max = min(ray_param_max, max(ray_param_1, ray_param_2))
        elif 0 < box_min_y or 0 > box_max_y:
            return None
        
        # Z slab
        if abs(ray_direction_z) > 1e-6:
            ray_param_1 = box_min_z / ray_direction_z
            ray_param_2 = box_max_z / ray_direction_z
            ray_param_min = max(ray_param_min, min(ray_param_1, ray_param_2))
            ray_param_max = min(ray_param_max, max(ray_param_1, ray_param_2))
        elif 0 < box_min_z or 0 > box_max_z:
            return None
        
        # Check if intersection exists and is in front of sensor
        if ray_param_min <= ray_param_max and ray_param_min > 0:
            # Use ray_param_min for closest intersection
            intersection_x = ray_param_min * ray_direction_x
            intersection_y = ray_param_min * ray_direction_y
            intersection_z = ray_param_min * ray_direction_z
            return (intersection_x, intersection_y, intersection_z)
        
        return None

    # 차선 교차점 계산 - 광선과 차선 마킹의 교차점 찾기 (지면상의 점으로 처리)
    def IntersectRayWithLanes(self, ray_direction_x: float, ray_direction_y: float, ray_direction_z: float) -> Tuple[float, float, float]:
        """Find intersection of ray with lane markings (treated as points on ground)."""
        # First, find intersection with ground
        ground_hit = self.IntersectRayWithGround(ray_direction_x, ray_direction_y, ray_direction_z)
        if ground_hit is None:
            return None
        
        # Convert hit point back to world coordinates to check against lanes
        world_hit = self.SensorToWorld(*ground_hit)
        
        # Check if hit point is close to any lane marking
        lane_tolerance = 0.3  # meters - tolerance for lane marking detection
        for lane in self.lanes.lanes:
            for lane_point in lane.lane_lines:
                distance_to_lane = math.sqrt((world_hit[0] - lane_point.x)**2 + 
                                           (world_hit[1] - lane_point.y)**2)
                if distance_to_lane < lane_tolerance:
                    return ground_hit
        
        return None

    # 교통표지판 교차점 계산 - 광선과 교통표지판(작은 박스로 처리)의 교차점 찾기    
    def IntersectRayWithSign(self, ray_direction_x: float, ray_direction_y: float, ray_direction_z: float, sign) -> Tuple[float, float, float]:
        """Find intersection of ray with traffic sign (treated as a small box)."""
        # Standard traffic sign dimensions
        standard_sign_width = 0.6   # meters
        standard_sign_height = 0.6  # meters  
        standard_sign_depth = 0.1   # meters (thin sign)
        
        # Transform sign center to sensor coordinates
        sign_center_world = (sign.pose.position.x, sign.pose.position.y, sign.pose.position.z)
        sign_center_sensor = self.MapToSensor(*sign_center_world)
        
        # Sign dimensions
        half_size_x = standard_sign_width * 0.5
        half_size_y = standard_sign_depth * 0.5  # depth becomes y in sensor frame
        half_size_z = standard_sign_height * 0.5
        
        # Bounding box in sensor coordinates (simplified as axis-aligned)
        box_min_x = sign_center_sensor[0] - half_size_x
        box_max_x = sign_center_sensor[0] + half_size_x
        box_min_y = sign_center_sensor[1] - half_size_y
        box_max_y = sign_center_sensor[1] + half_size_y
        box_min_z = sign_center_sensor[2] - half_size_z
        box_max_z = sign_center_sensor[2] + half_size_z
        
        # Ray-box intersection (AABB)
        ray_param_min = float('-inf')
        ray_param_max = float('inf')
        
        # X slab
        if abs(ray_direction_x) > 1e-6:
            ray_param_1 = box_min_x / ray_direction_x
            ray_param_2 = box_max_x / ray_direction_x
            ray_param_min = max(ray_param_min, min(ray_param_1, ray_param_2))
            ray_param_max = min(ray_param_max, max(ray_param_1, ray_param_2))
        elif 0 < box_min_x or 0 > box_max_x:
            return None
        
        # Y slab
        if abs(ray_direction_y) > 1e-6:
            ray_param_1 = box_min_y / ray_direction_y
            ray_param_2 = box_max_y / ray_direction_y
            ray_param_min = max(ray_param_min, min(ray_param_1, ray_param_2))
            ray_param_max = min(ray_param_max, max(ray_param_1, ray_param_2))
        elif 0 < box_min_y or 0 > box_max_y:
            return None
        
        # Z slab
        if abs(ray_direction_z) > 1e-6:
            ray_param_1 = box_min_z / ray_direction_z
            ray_param_2 = box_max_z / ray_direction_z
            ray_param_min = max(ray_param_min, min(ray_param_1, ray_param_2))
            ray_param_max = min(ray_param_max, max(ray_param_1, ray_param_2))
        elif 0 < box_min_z or 0 > box_max_z:
            return None
        
        # Check if intersection exists and is in front of sensor
        if ray_param_min <= ray_param_max and ray_param_min > 0:
            # Use ray_param_min for closest intersection
            intersection_x = ray_param_min * ray_direction_x
            intersection_y = ray_param_min * ray_direction_y
            intersection_z = ray_param_min * ray_direction_z
            return (intersection_x, intersection_y, intersection_z)
        
        return None

    # 강도 계산 - LiDAR 전력 기반 모델을 사용한 반사 강도 계산
    # TODO: 강도 계산 - 강도 계산 실습
    def CalculateIntensityForPoint(self, x: float, y: float, z: float, azimuth_deg: float, elevation_deg: float, object_type: str) -> int:
        """Calculate LiDAR intensity using power-based model.
        
        LiDAR Intensity Formula:
        intensity = α × (Power_receive) / (range²)
        
        Where:
        - α: Scaling constant for intensity normalization
        - Power_receive: Received optical power from target
        - range: Distance to target (meters)
        
        Power_receive calculation:
        Power_receive = P_laser × reflectivity × cos(θ) × roughness / range²
        
        Args:
            x, y, z: Point coordinates in sensor frame (meters)
            azimuth_deg: Horizontal angle in degrees (not used in current implementation)
            elevation_deg: Vertical angle for incident angle calculation (degrees)
            object_type: Known object type string for config lookup
            
        Returns:
            Intensity value between 0 and 255
        """
        # Physical constants
        min_distance = 1e-6  # Avoid division by zero
        roughness_min = 0.8  # Minimum surface roughness factor
        roughness_max = 1.2  # Maximum surface roughness factor
        min_intensity = 0  # Minimum intensity value
        max_intensity = 255  # Maximum intensity value
        
        # Calculate range to target
        range_m = math.sqrt(x*x + y*y + z*z)
        if range_m < min_distance:
            range_m = min_distance
        
        # Get material reflectivity
        reflectivity = self.object_props[object_type]['reflectivity']
        
        # Surface angle factor (Lambert's cosine law)
        # Assume surface normal points towards sensor for simplicity
        incident_angle_rad = math.radians(abs(elevation_deg))
        angle_factor = math.cos(incident_angle_rad)
        
        # Surface roughness variation (random component)
        roughness_factor = roughness_min + (roughness_max - roughness_min) * self.rng.random()
        
        # Calculate received power using simplified LiDAR equation
        # Power_receive = P_laser × reflectivity × cos(θ) × roughness / range²
        # TODO: 수식 완성, Hint: 레이저 출력, 반사율, 각도 요인, 거리에 따른 감쇠 고려, self.laser_power_watts 사용
        power_receive = self.laser_power_watts * reflectivity * math.cos(incident_angle_rad) * roughness_factor / (range_m**2)
        
        # Apply new intensity formula: intensity = α × (Power_receive) / (range²)
        # TODO: 수식 완성, Hint: self.alpha_constant 사용
        intensity_float = self.alpha_constant * power_receive / (range_m**2)
        
        # Convert to integer and clamp to valid range [0, 255]
        intensity_int = int(round(intensity_float))
        return max(min_intensity, min(max_intensity, intensity_int))
    
    # ===============================
    # 4. 좌표계 변환 함수들
    # ===============================

    # 센서→월드 좌표 변환 - 센서 좌표계를 월드 좌표계로 변환
    def SensorToWorld(self, x_sensor: float, y_sensor: float, z_sensor: float) -> Tuple[float, float, float]:
        """Convert sensor coordinates back to world coordinates."""
        if self.state is None:
            return x_sensor, y_sensor, z_sensor
        
        # Reverse transformation of MapToSensor
        ego_x = self.state.pose.position.x
        ego_y = self.state.pose.position.y
        ego_z = self.state.pose.position.z
        qw = self.state.pose.orientation.w
        qz = self.state.pose.orientation.z
        ego_yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
        
        # Reverse sensor rotation
        cos_sensor_yaw = math.cos(math.radians(self.sensor_yaw_deg))
        sin_sensor_yaw = math.sin(math.radians(self.sensor_yaw_deg))
        x_base = cos_sensor_yaw * x_sensor - sin_sensor_yaw * y_sensor + self.sensor_tx
        y_base = sin_sensor_yaw * x_sensor + cos_sensor_yaw * y_sensor + self.sensor_ty
        z_base = z_sensor + self.sensor_tz
        
        # Reverse vehicle rotation
        cos_ego_yaw = math.cos(ego_yaw)
        sin_ego_yaw = math.sin(ego_yaw)
        x_world = cos_ego_yaw * x_base - sin_ego_yaw * y_base + ego_x
        y_world = sin_ego_yaw * x_base + cos_ego_yaw * y_base + ego_y
        z_world = z_base + ego_z
        
        return x_world, y_world, z_world

    # 월드→센서 좌표 변환 - 월드 좌표계를 센서 좌표계로 변환
    def MapToSensor(self, x_map: float, y_map: float, z_map: float) -> Tuple[float, float, float]:
        if self.state is None:
            return x_map, y_map, z_map
        ego_x = self.state.pose.position.x
        ego_y = self.state.pose.position.y
        ego_z = self.state.pose.position.z  # Vehicle's z position (usually ground level)
        qw = self.state.pose.orientation.w
        qz = self.state.pose.orientation.z
        ego_yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
        delta_x = x_map - ego_x
        delta_y = y_map - ego_y
        delta_z = z_map - ego_z  # World z relative to vehicle base
        cos_ego_yaw = math.cos(-ego_yaw)
        sin_ego_yaw = math.sin(-ego_yaw)
        x_base = cos_ego_yaw * delta_x - sin_ego_yaw * delta_y
        y_base = sin_ego_yaw * delta_x + cos_ego_yaw * delta_y
        z_base = delta_z  # Keep relative z coordinate
        cos_sensor_yaw = math.cos(-math.radians(self.sensor_yaw_deg))
        sin_sensor_yaw = math.sin(-math.radians(self.sensor_yaw_deg))
        x_sensor = cos_sensor_yaw * (x_base - self.sensor_tx) - sin_sensor_yaw * (y_base - self.sensor_ty)
        y_sensor = sin_sensor_yaw * (x_base - self.sensor_tx) + cos_sensor_yaw * (y_base - self.sensor_ty)
        z_sensor = z_base - self.sensor_tz  # Subtract sensor height from ground level
        return x_sensor, y_sensor, z_sensor

    # ===============================
    # 5. 노이즈 처리 함수들
    # ===============================

    # 노이즈 적용 - LiDAR 측정값에 구면좌표계에서 가우시안 노이즈 추가
    # TODO: 노이즈 적용 - 노이즈 적용 실습
    def ApplyNoise(self, x_sensor: float, y_sensor: float, z_sensor: float) -> Tuple[float, float, float]:
        """Apply Gaussian noise to LiDAR measurements in polar coordinates."""
        if not self.enable_noise:
            return x_sensor, y_sensor, z_sensor
        
        # Convert to spherical coordinates (range, azimuth, elevation)
        #############################################################
        # TODO: 노이즈 적용 - 노이즈 적용 실습
        # Convert Cartesian to spherical coordinates
        # Hint: 직교좌표를 구면좌표(거리, 방위각, 고도각)로 변환
        range_true = math.sqrt(x_sensor**2 + y_sensor**2 + z_sensor**2)  # 실습: math.sqrt을 사용한 거리 계산
        azimuth_true = math.atan2(y_sensor, x_sensor)  # 실습: math.atan2로 방위각 계산
        range_plan = math.sqrt(x_sensor**2 + y_sensor**2)
        elevation_true = math.atan2(z_sensor, range_plan)  # 실습: math.atan2로 고도각 계산, Hint: z_sensor와 수평면 거리 사용
        
        # Add Gaussian noise in polar coordinates
        # Hint: 구면좌표계에서 가우시안 노이즈 적용
        range_noise = self.rng.normal(0, self.range_noise_std)  # 실습: self.rng.normal로 거리 노이즈 생성
        azimuth_noise_rad = math.radians(self.rng.normal(0, self.azimuth_noise_std))  # 실습: 방위각 노이즈 생성 및 라디안 변환, math.radians 사용, self.azimuth_noise_std 사용
        elevation_noise_rad = math.radians(self.rng.normal(0, self.elevation_noise_std))  # 실습: 고도각 노이즈 생성 및 라디안 변환, math.radians 사용, self.elevation_noise_std 사용
        
        # Apply noise
        range_noisy = max(range_true + range_noise, 0.0)  # 실습: 노이즈를 적용한 거리 (양수 보장)
        azimuth_noisy = azimuth_true + azimuth_noise_rad  # 실습: 노이즈를 적용한 방위각
        elevation_noisy = elevation_true + elevation_noise_rad  # 실습: 노이즈를 적용한 고도각
        #############################################################
        
        # Convert back to Cartesian coordinates
        # First project to horizontal plane, then add elevation
        horizontal_range = range_noisy * math.cos(elevation_noisy)
        x_sensor_noisy = horizontal_range * math.cos(azimuth_noisy)
        y_sensor_noisy = horizontal_range * math.sin(azimuth_noisy)
        z_sensor_noisy = range_noisy * math.sin(elevation_noisy)
        
        return x_sensor_noisy, y_sensor_noisy, z_sensor_noisy
    
    # 클러터 포인트 생성 - 허위 양성 LiDAR 감지를 위한 랜덤 클러터 포인트 생성
    # TODO: 클러터 좌표 생성 - 클러터 좌표 생성 실습
    def GenerateClutterPoints(self) -> List[Tuple[float, float, float]]:
        """Generate random clutter (false positive) points."""
        clutter_points = []
        
        # Check if clutter should occur this scan cycle
        if self.rng.random() > self.clutter_probability:
            return clutter_points
        
        # Generate random number of clutter points
        num_clutter = max(1, int(self.rng.poisson(self.clutter_density)))
        
        for _ in range(num_clutter):
            #############################################################
            # TODO: 클러터 좌표 생성 - 클러터 좌표 생성 실습
            # Random spherical coordinates within sensor range
            # Hint: 구면좌표계에서 랜덤한 거리, 방위각, 고도각 생성
            range_val = self.rng.uniform(self.clutter_range_min, self.clutter_range_max)  # 실습: self.rng.uniform으로 클러터 범위 내 거리 생성, Hint: self.clutter_range_min, self.clutter_range_max
            azimuth_deg = self.rng.uniform(0, 360)  # 실습: 0~360도 범위의 랜덤 방위각 생성
            elevation_deg = self.rng.uniform(self.vertical_fov_min, self.vertical_fov_max)  # 실습: vertical_fov 범위의 랜덤 고도각 생성, Hint: self.vertical_fov_min, self.vertical_fov_max
            #############################################################

            # Convert to Cartesian coordinates in sensor frame
            azimuth_rad = math.radians(azimuth_deg)
            elevation_rad = math.radians(elevation_deg)
            
            x = range_val * math.cos(elevation_rad) * math.cos(azimuth_rad)
            y = range_val * math.cos(elevation_rad) * math.sin(azimuth_rad)
            z = range_val * math.sin(elevation_rad)
            
            clutter_points.append((x, y, z))
        
        return clutter_points

def Main():
    """
    LiDAR 센서 시뮬레이터 메인 함수

    알고리즘 프로세스 흐름:
    1. ROS 노드 초기화 (rospy.init_node)
    2. LidarEmu 클래스 인스턴스 생성 (__init__ 호출)
       - 파라미터 로드 및 설정
       - ROS 퍼블리셔/구독자 설정
       - 타이머 생성 (주기적 CallbackTimer 실행)
    3. rospy.spin()으로 이벤트 루프 시작

    타이머 콜백(CallbackTimer)에서 실행되는 주요 프로세스:
    1. GenerateRingBasedScan: 링 기반 라이다 스캔 생성
       - 각 링(고도각)과 방위각 조합으로 레이 캐스팅 수행
       - CastRay: 개별 광선에 대한 교차점 계산
       - ApplyNoise: 가우시안 노이즈 적용
       - GenerateClutterPoints: 클러터(허위 양성) 포인트 생성
    2. CreateXyziPointcloud2: PointCloud2 메시지 생성 및 퍼블리시

    각 프로세스에서 공통적으로 수행되는 작업:
    - 좌표계 변환 (MapToSensor): 월드 프레임 → 센서 프레임
    - 레이 캐스팅: 지면, 건물, 차량, 가드레일, 표지판, 차선과의 교차점 계산
    - 강도 계산 (CalculateIntensityForPoint): 물체 타입과 거리 기반 반사 강도 계산
    - 노이즈 적용 (ApplyNoise): 구면좌표계에서 가우시안 노이즈 추가
    - 클러터 생성: 허위 양성 감지 포인트 시뮬레이션
    """
    rospy.init_node("sensor_lidar_node")
    lidar = LidarEmu()
    if hasattr(lidar, 'enabled') and lidar.enabled:
        rospy.loginfo("sensor_lidar_node started")
    else:
        rospy.loginfo("sensor_lidar_node disabled")
    rospy.spin()

if __name__ == "__main__":
    Main()

