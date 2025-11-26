#!/usr/bin/env python
"""
Camera Sensor Simulation Node

이 노드는 차량에 장착된 카메라 센서를 시뮬레이션합니다.
차량의 현재 위치와 방향을 기준으로 주변 환경을 카메라 이미지로 렌더링합니다.

교육 목적:
1. 3D 월드 좌표계에서 2D 이미지 좌표계로의 변환 과정 이해
2. 카메라 모델과 투영 변환의 기본 개념 학습
3. 컴퓨터 비전에서의 좌표 변환 원리 학습

입력 토픽들:
- /vehicle/state: 차량의 현재 위치와 방향
- /env/buildings: 건물 정보
- /env/lanes: 차선 정보  
- /env/guardrails: 가드레일 정보
- /env/surrounding_vehicles: 다른 차량들 정보
- /env/traffic_signs: 교통 표지판 정보
- /env/traffic_lights: 신호등 정보

출력:
- /sensors/camera/image_raw: 시뮬레이션된 카메라 이미지
"""

import math
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from dcas_msgs.msg import LaneArray, TrafficLightArray, TrafficSignArray, ObjectArray, VehicleState


class CameraSimulator:
    """교육용 카메라 시뮬레이터
    
    실제 카메라의 동작을 단순화하여 시뮬레이션합니다.
    복잡한 3D 렌더링 대신 2D 투영을 사용하여 이해하기 쉽게 구현했습니다.
    """
    
    def __init__(self):
        """카메라 시뮬레이터 초기화"""
        
        # =================================================================
        # 1. ROS 노드 설정
        # =================================================================
        # rospy.loginfo("카메라 시뮬레이터를 초기화합니다...")
        
        # 카메라 이미지 발행자
        self.image_publisher = rospy.Publisher(
            "/sensors/camera/image_raw", 
            Image, 
            queue_size=1
        )
        
        # OpenCV와 ROS 이미지 메시지 변환을 위한 브리지
        self.cv_bridge = CvBridge()
        
        # =================================================================
        # 2. 카메라 파라미터 설정 (Configuration에서 읽기)
        # =================================================================
        # 타이머 주기
        self.update_period = float(rospy.get_param("/sensor_camera_node/period_s", 0.1))

        # 카메라 활성화 여부
        self.enable_sensor = rospy.get_param("/sensor_camera_node/enable_sensor", True)
        
        # 카메라 위치 (차량 중심을 기준으로 한 오프셋)
        self.camera_offset_x = float(rospy.get_param("/sensor_camera_node/offset_x", 1.1))   # 전방 오프셋
        self.camera_offset_y = float(rospy.get_param("/sensor_camera_node/offset_y", 0.0))   # 좌우 오프셋
        self.camera_offset_z = float(rospy.get_param("/sensor_camera_node/offset_z", 1.4))   # 높이 오프셋
        
        # 카메라 방향
        self.camera_roll_deg = float(rospy.get_param("/sensor_camera_node/roll_deg", 0.0))    # 롤 각도
        self.camera_pitch_deg = float(rospy.get_param("/sensor_camera_node/pitch_deg", -5.0))  # 틸트 각도
        self.camera_yaw_deg = float(rospy.get_param("/sensor_camera_node/yaw_deg", 0.0))     # 요 각도
        
        # 이미지 크기
        self.image_width = int(rospy.get_param("/sensor_camera_node/width", 640))
        self.image_height = int(rospy.get_param("/sensor_camera_node/height", 480))
        
        # 카메라가 보는 거리 범위
        self.camera_near = float(rospy.get_param("/sensor_camera_node/near_m", 1.0))     # 최소 거리
        self.camera_far = float(rospy.get_param("/sensor_camera_node/far_m", 100.0))    # 최대 거리
        
        # 카메라 내부 파라미터 (초점거리)
        self.focal_length = float(rospy.get_param("/sensor_camera_node/focal_length", 320.0))

        # 렌더링 파라미터
        self.enable_distortion = rospy.get_param("/sensor_camera_node/enable_distortion", False)
        self.distortion_k1 = float(rospy.get_param("/sensor_camera_node/distortion_k1", 0.1))
        self.distortion_k2 = float(rospy.get_param("/sensor_camera_node/distortion_k2", -0.05))
        self.distortion_k3 = float(rospy.get_param("/sensor_camera_node/distortion_k3", 0.0))
        self.distortion_p1 = float(rospy.get_param("/sensor_camera_node/distortion_p1", 0.01))
        self.distortion_p2 = float(rospy.get_param("/sensor_camera_node/distortion_p2", 0.005))
        
        # 카메라 시야각 (Field of View) 계산 - 초점거리로부터 역산
        self.camera_fov_deg = math.degrees(2.0 * math.atan(self.image_width / (2.0 * self.focal_length)))
        
        # Principal point (image center in pixels)
        self.image_center_u = self.image_width / 2.0
        self.image_center_v = self.image_height / 2.0
        
        
        # =================================================================
        # 3. 환경 데이터 저장 변수들
        # =================================================================
        
        # 차량 상태 (위치, 방향)
        self.vehicle_state = None
        
        # 환경 객체들
        self.buildings = None
        self.lanes = None
        self.guardrails = None
        self.surrounding_vehicles = None
        self.traffic_signs = None
        self.traffic_lights = None
        
        # =================================================================
        # 4. ROS 구독자들 설정
        # =================================================================
        
        # 차량 상태 구독
        rospy.Subscriber("/vehicle/state", VehicleState, self.callback_vehicle_state)
        
        # 환경 토픽들 구독
        rospy.Subscriber("/env/buildings", ObjectArray, self.callback_buildings)
        rospy.Subscriber("/env/lanes", LaneArray, self.callback_lanes)
        rospy.Subscriber("/env/guardrails", ObjectArray, self.callback_guardrails)
        rospy.Subscriber("/env/surrounding_vehicles", ObjectArray, self.callback_vehicles)
        rospy.Subscriber("/env/traffic_signs", TrafficSignArray, self.callback_signs)
        rospy.Subscriber("/env/traffic_lights", TrafficLightArray, self.callback_lights)
        
        # =================================================================
        # 5. 타이머 설정 (주기적으로 이미지 생성)
        # =================================================================
        
        self.timer = rospy.Timer(
            rospy.Duration(self.update_period), 
            self.callback_timer
        )

    # =====================================================================
    # ROS 콜백 함수들 (환경 데이터 수신)
    # =====================================================================
    
    def callback_vehicle_state(self, msg):
        """차량 상태 정보 수신"""
        self.vehicle_state = msg
        
    def callback_buildings(self, msg):
        """건물 정보 수신"""
        self.buildings = msg
        
    def callback_lanes(self, msg):
        """차선 정보 수신"""
        self.lanes = msg
        
    def callback_guardrails(self, msg):
        """가드레일 정보 수신"""
        self.guardrails = msg
        
    def callback_vehicles(self, msg):
        """다른 차량들 정보 수신"""
        self.surrounding_vehicles = msg
        
    def callback_signs(self, msg):
        """교통표지판 정보 수신"""
        self.traffic_signs = msg
        
    def callback_lights(self, msg):
        """신호등 정보 수신"""
        self.traffic_lights = msg
    
    # =====================================================================
    # 좌표 변환 함수들 (교육용 핵심 부분)
    # =====================================================================
    
    def world_to_camera_frame(self, world_x, world_y, world_z):
        """월드 좌표계를 카메라 좌표계로 4x4 동차변환행렬을 이용하여 변환
        
        교육 목적:
        이 함수는 로봇공학에서 표준적으로 사용되는 4x4 동차변환행렬(Homogeneous Transformation Matrix)을
        사용하여 좌표 변환을 수행합니다. 이를 통해 학생들이 다음을 학습할 수 있습니다:
        1. 동차변환행렬의 구조와 의미
        2. 회전과 이동을 하나의 행렬로 표현하는 방법
        3. 연속적인 좌표 변환의 합성
        4. 역변환 행렬의 활용
        
        변환 과정:
        1. world_to_vehicle 변환행렬 생성 (차량의 위치와 방향)
        2. vehicle_to_camera 변환행렬 생성 (카메라의 오프셋과 방향)
        3. world_to_camera = vehicle_to_camera @ world_to_vehicle
        4. camera_to_world = inv(world_to_camera)
        5. camera_point = camera_to_world @ world_point
        
        Args:
            world_x, world_y, world_z: 월드 좌표계의 점
            
        Returns:
            (cam_x, cam_y, cam_z): 카메라 좌표계의 점
        """
        
        if self.vehicle_state is None:
            return None, None, None
            
        # 차량의 현재 위치와 방향 얻기
        vehicle_x = self.vehicle_state.pose.position.x
        vehicle_y = self.vehicle_state.pose.position.y
        vehicle_yaw = self.quaternion_to_yaw(self.vehicle_state.pose.orientation)
        
        # ================================================================
        # 1단계: World -> Vehicle 변환행렬 생성
        # ================================================================
        
        # 차량의 회전행렬 (Z축 중심 회전 - yaw)
        cos_vehicle_yaw = math.cos(vehicle_yaw)
        sin_vehicle_yaw = math.sin(vehicle_yaw)
        
        ###########################################################################################
        # Todo
        # 4x4 world_to_vehicle 변환행렬 (x, y, yaw)
        T_world_to_vehicle = np.array([
            [ cos_vehicle_yaw, -sin_vehicle_yaw, 0.0, vehicle_x],
            [ sin_vehicle_yaw, cos_vehicle_yaw, 0.0, vehicle_y],
            [ 0.0, 0.0, 1.0, 0.0],
            [ 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        ###########################################################################################
        # ================================================================
        # 2단계: Vehicle -> Camera 변환행렬 생성
        # ================================================================
        
        # 카메라 yaw 회전행렬
        camera_yaw_rad = math.radians(self.camera_yaw_deg)
        
        # 카메라 pitch 회전행렬  
        camera_pitch_rad = math.radians(self.camera_pitch_deg)

        # 카메라 roll 회전행렬
        camera_roll_rad = math.radians(self.camera_roll_deg)

        # 카메라 병진 벡터
        camera_translation_x = self.camera_offset_x
        camera_translation_y = self.camera_offset_y
        camera_translation_z = self.camera_offset_z
        
        ###########################################################################################
        # Todo
        # 합성 회전행렬 (camera_yaw_rad, camera_pitch_rad, camera_roll_rad)
        rot_x = np.array([
            [1, 0, 0],
            [0, math.cos(camera_roll_rad), -math.sin(camera_roll_rad)],
            [0, math.sin(camera_roll_rad), math.cos(camera_roll_rad)]
        ])
        rot_y = np.array([
            [math.cos(camera_pitch_rad), 0, math.sin(camera_pitch_rad)],
            [0, 1, 0],
            [-math.sin(camera_pitch_rad), 0, math.cos(camera_pitch_rad)]
        ])
        rot_z = np.array([
            [math.cos(camera_yaw_rad), -math.sin(camera_yaw_rad), 0],
            [math.sin(camera_yaw_rad), math.cos(camera_yaw_rad), 0],
            [0, 0, 1]
        ])
        R_camera = rot_z @ rot_y @ rot_x
        ###########################################################################################
        
        ###########################################################################################
        # Todo
        # 4x4 vehicle_to_camera 변환행렬
        T_vehicle_to_camera = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        T_vehicle_to_camera[:3, :3] = R_camera
        T_vehicle_to_camera[:3, 3] = np.transpose(np.array([camera_translation_x, camera_translation_y, camera_translation_z]))
        
        ###########################################################################################

        # ================================================================
        # 3단계: World -> Camera 변환행렬 계산
        # ================================================================
        
        # 변환행렬 합성: T_world_to_camera = T_world_to_vehicle @ T_vehicle_to_camera
        ###########################################################################################
        # Todo
        T_world_to_camera = T_world_to_vehicle @ T_vehicle_to_camera
        ###########################################################################################

        # ================================================================
        # 4단계: Camera -> World 역변환행렬 계산
        # ================================================================
        
        ###########################################################################################
        # Todo
        try:
            # 역행렬 계산
            T_camera_to_world = np.linalg.inv(T_world_to_camera)
        except np.linalg.LinAlgError:
            # 역행렬이 존재하지 않는 경우 (특이행렬)
            rospy.logwarn("Camera transformation matrix is singular, cannot compute inverse")
            return None, None, None
        ###########################################################################################

        # ================================================================
        # 5단계: 월드 좌표점을 카메라 좌표계로 변환
        # ================================================================
        
        # 월드 좌표를 동차좌표로 변환 [x, y, z, 1]
        world_point = np.array([world_x, world_y, world_z, 1.0])
        
        ###########################################################################################
        # Todo
        # 변환 수행
        camera_point = T_camera_to_world @ world_point
        ###########################################################################################

        # 동차좌표에서 일반좌표로 변환 (w=1로 정규화)
        if abs(camera_point[3]) < 1e-10:
            rospy.logwarn("Invalid homogeneous coordinate (w≈0)")
            return None, None, None
            
        camera_x_right   = -camera_point[1] / camera_point[3]
        camera_y_down    = -camera_point[2] / camera_point[3]
        camera_z_forward =  camera_point[0] / camera_point[3]
        
        return camera_x_right, camera_y_down, camera_z_forward
    
    def camera_to_image_coordinates(self, camera_x_right, camera_y_down, camera_z_forward):
        """카메라 좌표계를 이미지 좌표계로 투영
        
        좌표계 정의:
        ============
        카메라 좌표계 (입력):
        - cam_x: 오른쪽 (right) - 카메라 기준 오른쪽 방향
        - cam_y: 아래 (down)    - 카메라 기준 아래쪽 방향  
        - cam_z: 전방 (forward) - 카메라가 바라보는 방향
        
        이미지 좌표계 (출력):
        - u (이미지 X): 오른쪽 (right) - 이미지 픽셀의 오른쪽 방향
        - v (이미지 Y): 아래 (down) - 이미지 픽셀의 아래쪽 방향
        - depth: 전방 (forward) - 카메라로부터의 거리
        
        좌표 변환 관계:
        ==============
        - cam_x (전방) → depth (깊이)
        - cam_y (좌측) → -u (오른쪽이 양수이므로 음수)
        - cam_z (위) → -v (아래가 양수이므로 음수)
        
        핀홀 카메라 투영 공식:
        u = (-cam_y / cam_x) * focal_length + cx
        v = (-cam_z / cam_x) * focal_length + cy
        depth = cam_x
        
        Args:
            cam_x, cam_y, cam_z: 카메라 좌표계의 점
            
        Returns:
            (u, v, depth): 이미지 좌표 (u, v)와 깊이 정보
            유효하지 않은 경우 (None, None, None) 반환
        """
        
        # 카메라 앞쪽에 있는지 확인 (양수 camera_x_forward 값 - 전방)
        if camera_z_forward <= 0:
            return None, None, None
            
        # 시야 거리 범위 내에 있는지 확인 (camera_x_forward가 depth)
        if camera_z_forward < self.camera_near or camera_z_forward > self.camera_far:
            return None, None, None

        ###########################################################################################
        # Todo
        # 이미지 평면으로 투영 (division-by-zero 보호)
        if abs(camera_z_forward) < 1e-9:
            return None, None, None
        img_proj_x = camera_x_right / camera_z_forward
        img_proj_y = camera_y_down / camera_z_forward
        ###########################################################################################
        # 수치 안정성 체크
        if not (math.isfinite(img_proj_x) and math.isfinite(img_proj_y)):
            return None, None, None

        # 왜곡 적용 (https://darkpgmr.tistory.com/31)
        if self.enable_distortion:
            ###########################################################################################
            # Todo
            # x = (1 + k1*r^2 + k2*r^4 + k3*r^6) * x + 2*p1*x*y + p2*(r^2 + 2*x^2)
            # y = (1 + k1*r^2 + k2*r^4 + k3*r^6) * y + p1*(r^2 + 2*y^2) + 2p2*x*y
            r = math.sqrt(math.pow(img_proj_x,2) + math.pow(img_proj_y,2))
            img_proj_x = (1+self.distortion_k1*math.pow(r,2) + self.distortion_k2*math.pow(r,4) + self.distortion_k3*math.pow(r,6))*img_proj_x + 2*self.distortion_p1*img_proj_x*img_proj_y + self.distortion_p2*(math.pow(r,2) + 2*math.pow(img_proj_x,2))
            img_proj_y = (1+self.distortion_k1*math.pow(r,2) + self.distortion_k2*math.pow(r,4) + self.distortion_k3*math.pow(r,6))*img_proj_y + self.distortion_p1*(math.pow(r,2) + 2*img_proj_y**2) + 2*self.distortion_p2*img_proj_x*img_proj_y
            ###########################################################################################
            # 수치 안정성 체크
            if not (math.isfinite(img_proj_x) and math.isfinite(img_proj_y)):
                return None, None, None
            # 비현실적 투영 방지 (극단 FOV): tan(84deg)≈9.5 기준 여유치
            max_proj = 10.0
            if abs(img_proj_x) > max_proj or abs(img_proj_y) > max_proj:
                return None, None, None

        # 핀홀 카메라 모델을 사용한 투영
        ###########################################################################################
        # Todo
        # 좌표계 변환을 고려한 투영 공식
        f = self.focal_length
        cx = self.image_center_u
        cy = self.image_center_v
        u = f*img_proj_x + cx
        v = f*img_proj_y + cy
        ###########################################################################################
        # 수치 및 경계 체크
        if not (math.isfinite(u) and math.isfinite(v)):
            return None, None, None

        return int(u), int(v), camera_z_forward  # depth는 camera_z_forward (전방 거리)
    
    def world_to_image_coordinates(self, world_x, world_y, world_z=0.0):
        """월드 좌표를 이미지 좌표로 직접 변환 (편의 함수)

        Args:
            world_x: X in world frame (meters).
            world_y: Y in world frame (meters).
            world_z: Z in world frame (meters), default ground.
        Returns:
            Tuple (u, v, depth) if point is within frustum and image bounds; otherwise (None, None, None).
        """
        
        # 1단계: 월드 -> 카메라 좌표계
        camera_x_right, camera_y_down, camera_z_forward = self.world_to_camera_frame(world_x, world_y, world_z)
        
        if camera_x_right is None:
            return None, None, None
            
        # 2단계: 카메라 -> 이미지 좌표계
        u, v, depth = self.camera_to_image_coordinates(camera_x_right, camera_y_down, camera_z_forward)
        return u, v, depth
    
    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw (rotation around Z axis in radians).

        Uses the standard Tait-Bryan ZYX extraction formula for yaw.
        Args:
            quat: geometry_msgs/Quaternion-like with fields x, y, z, w.
        Returns:
            Yaw angle in radians.
        """
        
        # 쿼터니언에서 yaw 각도 추출
        # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
        qx, qy, qz, qw = quat.x, quat.y, quat.z, quat.w
        
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)
        )
        
        return yaw
    
    # =====================================================================
    # 렌더링 함수들 (각 환경 요소를 이미지에 그리기)
    # =====================================================================
    
    def render_background(self, image):
        """간단한 수평선과 함께 하늘과 지면 배경 그리기

        - 하늘: 연한 파란색
        - 지면: 녹색
        - 수평선은 cy + f * tan(pitch)로 근사화
        """
        # 피치(세로 오프셋)와 롤(기울기)을 반영한 카메라 수평선 계산/렌더링
        # 강의용: 주석을 한국어로 정리했습니다.
        height = self.image_height
        width = self.image_width

        # 1) 각도를 라디안으로 변환, 피치와 롤 각도는 음수로 변환(이미지는 반대로 회전하므로)
        pitch_rad = math.radians(-self.camera_pitch_deg)
        roll_rad = math.radians(-self.camera_roll_deg)

        # 2) 이미지 좌표계에서 수평선의 선형 모델: v(u) = m*u + b
        #    - 피치: 수평선의 세로 위치를 이동(롤=0일 때는 중심을 지남)
        #    - 롤: 수평선을 기울임; 기울기 m = tan(roll)
        v_at_center = self.image_center_v + self.focal_length * math.tan(pitch_rad)
        slope_dv_du = math.tan(roll_rad)
        intercept_v_at_u0 = v_at_center - slope_dv_du * self.image_center_u

        # 3) 색상 (BGR)
        sky_color = (255, 220, 200)
        ground_color = (90, 120, 90)

        # 4) 먼저 지면색으로 채우고, 수평선 위쪽 영역만 하늘색으로 덮기
        image[:, :] = ground_color

        # 각 열(u)에서 수평선의 v 값을 계산하고, 브로드캐스팅으로 하늘 영역 마스크 생성
        u_cols = np.arange(width, dtype=np.float32)
        v_line = slope_dv_du * u_cols + intercept_v_at_u0
        rows = np.arange(height, dtype=np.float32)[:, None]
        sky_mask = rows < v_line[None, :]
        image[sky_mask] = sky_color

        # 5) 실제 수평선(선분) 그리기: v = m*u + b 와 이미지 경계의 교점 2개를 찾아 연결
        def line_rect_intersections(m, b, w, h):
            pts = []
            # 왼쪽 경계 (u=0)
            y_left = b
            if 0.0 <= y_left <= (h - 1):
                pts.append((0, int(round(y_left))))
            # 오른쪽 경계 (u=w-1)
            y_right = m * (w - 1) + b
            if 0.0 <= y_right <= (h - 1):
                pts.append((w - 1, int(round(y_right))))
            # 위쪽 경계 (v=0) / 아래쪽 경계 (v=h-1)
            if abs(m) > 1e-8:
                x_top = (0.0 - b) / m
                if 0.0 <= x_top <= (w - 1):
                    pts.append((int(round(x_top)), 0))
                x_bottom = ((h - 1) - b) / m
                if 0.0 <= x_bottom <= (w - 1):
                    pts.append((int(round(x_bottom)), h - 1))
            # 중복 제거
            uniq = []
            for p in pts:
                if p not in uniq:
                    uniq.append(p)
            return uniq
        # Draw horizontal line
        border_pts = line_rect_intersections(slope_dv_du, intercept_v_at_u0, width, height)

        if len(border_pts) >= 2:
            # 가장 멀리 떨어진 두 점을 골라 안정적인 긴 수평선 그리기
            best_pair = None
            best_dist2 = -1.0
            for i in range(len(border_pts)):
                for j in range(i + 1, len(border_pts)):
                    dx = border_pts[i][0] - border_pts[j][0]
                    dy = border_pts[i][1] - border_pts[j][1]
                    d2 = dx * dx + dy * dy
                    if d2 > best_dist2:
                        best_dist2 = d2
                        best_pair = (border_pts[i], border_pts[j])
            if best_pair is not None:
                pt1 = best_pair[0]
                pt2 = best_pair[1]
                cv2.line(image, pt1, pt2, (240, 180, 200), 1, cv2.LINE_AA)

    def render_buildings(self, image, commands):
        """건물들을 이미지에 다각형으로 그리기

        Args:
            image: OpenCV BGR image buffer to draw into.
            commands: List accumulating drawing primitives with their approximate depth for painter's algorithm.
        """
        if self.buildings is None:
            return
            
        for building in self.buildings.objects:
            # 건물 높이 정보 추출 (여러 소스에서 시도)
            height_val = getattr(building, 'height', None)
            if height_val is None:
                _size = getattr(building, 'size', None)
                if _size is not None and hasattr(_size, 'z') and _size.z is not None:
                    height_val = _size.z
                else:
                    height_val = 5.0  # 기본값

            # 건물 footprint의 모든 점을 이미지 좌표로 변환
            base_points = []   # 바닥 점들: (u, v, depth) 또는 (None, ...)
            top_points = []    # 지붕 점들: (u, v, depth) 또는 (None, ...)
            depths_for_sort = []

            for point in building.footprint:
                ###########################################################################################
                # Todo
                # 바닥 점 (z=0.0)과 지붕 점 (z=height_val) 변환
                # 바닥 점 변환
                u0, v0, d0 = self.world_to_image_coordinates(point.x, point.y, 0.0)
                # 지붕 점 변환
                u1, v1, d1 = self.world_to_image_coordinates(point.x, point.y, height_val)
                ###########################################################################################
                # 바닥 점 처리
                if u0 is not None and d0 > 0:
                    base_points.append((u0, v0, d0))
                else:
                    base_points.append((None, None, None))
                
                # 지붕 점 처리
                if u1 is not None and d1 > 0:
                    top_points.append((u1, v1, d1))
                    depths_for_sort.append(d1)
                else:
                    top_points.append((None, None, None))

            # 건물 면들을 그리기 (최소 3개 점이 있어야 함)
            n = len(building.footprint)
            if n >= 3:
                # 측면 벽들 그리기 (연속된 점들 사이의 사각형)
                for i in range(n):
                    j = (i + 1) % n
                    b0 = base_points[i]     # 현재 점의 바닥 (u, v, depth)
                    b1 = base_points[j]     # 다음 점의 바닥 (u, v, depth)
                    t1 = top_points[j]      # 다음 점의 지붕 (u, v, depth)
                    t0 = top_points[i]      # 현재 점의 지붕 (u, v, depth)
                    
                    # 모든 점이 유효한지 확인
                    if None in b0 or None in b1 or None in t0 or None in t1:
                        continue
                    
                    # 사각형(quad) 좌표 생성 (b0->b1->t1->t0 순서)
                    quad = np.array([
                        [b0[0], b0[1]], 
                        [b1[0], b1[1]], 
                        [t1[0], t1[1]], 
                        [t0[0], t0[1]]
                    ], dtype=np.int32)
                    
                    # 거리에 따른 색상 조정 (멀수록 어둡게)
                    face_depth = (b0[2] + b1[2] + t0[2] + t1[2]) / 4.0  # 평균 깊이
                    distance_factor = min(1.0, max(0.25, 25.0 / face_depth))
                    side_color = int(170 * distance_factor)
                    edge_color = int(110 * distance_factor)
                    
                    # 그리기 명령 큐에 추가
                    commands.append({
                        'type': 'poly', 
                        'points': quad, 
                        'fill': (side_color, side_color, side_color), 
                        'edge': (edge_color, edge_color, edge_color), 
                        'th': 1, 
                        'depth': face_depth
                    })

                # 지붕면 그리기
                top_poly = []
                top_depths = []
                for (u1, v1, d1) in top_points:
                    if u1 is not None:
                        top_poly.append([u1, v1])
                        top_depths.append(d1)
                
                # 지붕면이 최소 3개 점이 있는지 확인
                if len(top_poly) >= 3:
                    top_poly_np = np.array(top_poly, dtype=np.int32)
                    top_depth = np.mean(top_depths)
                    distance_factor = min(1.0, max(0.3, 25.0 / top_depth))
                    top_color = int(190 * distance_factor)
                    outline_color = int(130 * distance_factor)
                    
                    # 지붕 그리기 명령 큐에 추가
                    commands.append({
                        'type': 'poly', 
                        'points': top_poly_np, 
                        'fill': (top_color, top_color, top_color), 
                        'edge': (outline_color, outline_color, outline_color), 
                        'th': 2, 
                        'depth': top_depth
                    })

    def render_lanes(self, image):
        """차선들을 타입(색/두께/패턴)에 따라 그리기.

        Lane.lane_type codes (assumed):
        1 WHITE_SOLID, 2 WHITE_DASHED, 3 YELLOW_SOLID, 4 YELLOW_DASHED, 5 DOUBLE_YELLOW_SOLID
        """
        
        if self.lanes is None:
            return
            
        for lane in self.lanes.lanes:
            # 차선의 각 점들을 이미지 좌표계로 투영하여 폴리라인 생성
            image_polyline_points: list = []
            ###########################################################################################
            # Todo
            for point in lane.lane_lines:
                # 월드 좌표를 카메라 좌표계로 변환 (z=0.03은 차선이 바닥보다 약간 위에 있음을 표현)
                u, v, depth = self.world_to_image_coordinates(point.x, point.y, 0.03)

                # 유효한 이미지 좌표가 나오면 폴리라인 포인트에 추가
                if u is not None:
                    image_polyline_points.append((u, v))
            ###########################################################################################

            # 최소 2개의 점이 있어야 선을 그릴 수 있음
            if len(image_polyline_points) < 2:
                continue

            # 2) Style by lane_type
            lane_type = getattr(lane, 'lane_type', 0)
            if lane_type == 1:  # WHITE_SOLID
                color = (255, 255, 255)
                thickness = 2
                pattern = 'solid'
            elif lane_type == 2:  # WHITE_DASHED
                color = (220, 220, 220)
                thickness = 2
                pattern = 'dashed'
            elif lane_type == 3:  # YELLOW_SOLID
                color = (0, 255, 255)
                thickness = 3
                pattern = 'solid'
            elif lane_type == 4:  # YELLOW_DASHED
                color = (0, 200, 200)
                thickness = 3
                pattern = 'dashed'
            elif lane_type == 5:  # DOUBLE_YELLOW_SOLID
                color = (0, 255, 255)
                thickness = 5
                pattern = 'solid'
            else:  # UNKNOWN
                color = (160, 160, 160)
                thickness = 2
                pattern = 'solid'

            # 3) Draw
            if pattern == 'solid':
                cv2.polylines(image, [np.array(image_polyline_points, dtype=np.int32)], isClosed=False, color=color, thickness=thickness)
            else:
                # dashed: draw every other segment
                on = True
                for i in range(len(image_polyline_points) - 1):
                    if on:
                        p1 = image_polyline_points[i]
                        p2 = image_polyline_points[i + 1]
                        cv2.line(image, p1, p2, color, thickness)
                    on = not on
    
    def render_guardrails(self, image, commands):
        """가드레일들을 다각형으로 이미지에 그리기"""
        
        if self.guardrails is None:
            return
            
        for guardrail in self.guardrails.objects:
            # 가드레일의 위치와 크기 정보 추출
            center_x = guardrail.pose.position.x
            center_y = guardrail.pose.position.y
            base_z = guardrail.pose.position.z
            
            # 가드레일 크기 정보
            length = guardrail.size.x  # 길이 (가드레일이 늘어선 방향)
            height = guardrail.size.z  # 높이 (지면에서 위로)
            
            # 가드레일 방향 각도
            guardrail_yaw = self.quaternion_to_yaw(guardrail.pose.orientation)
            
            # 가드레일을 직육면체로 모델링 - 8개 모서리 점 계산
            cos_yaw = math.cos(guardrail_yaw)
            sin_yaw = math.sin(guardrail_yaw)
            
            # 길이 방향 벡터 (가드레일이 늘어선 방향)
            half_length = length / 2.0
            length_vec_x = half_length * cos_yaw
            length_vec_y = half_length * sin_yaw
            
            # 하단면 2개 모서리 (정면만, 지면 레벨)
            base_corners_world = [
                (center_x - length_vec_x, center_y - length_vec_y, base_z),
                (center_x + length_vec_x, center_y + length_vec_y, base_z)
            ]
            # 상단면 2개 모서리 (높이만큼 위)
            top_corners_world = [(x, y, base_z + height) for (x, y, _) in base_corners_world]
            
            ###########################################################################################
            # Todo
            # 월드 좌표를 이미지 좌표로 변환하는 헬퍼 함수
            def proj(world_pts):
                pts = []
                depths = []
                for (world_point_x, world_point_y, world_point_z) in world_pts:
                    #
                    u, v, d = self.world_to_image_coordinates(world_point_x, world_point_y, world_point_z)
                    if u is None or d <= 0:
                        pts.append(None)
                    else:
                        pts.append((u, v))
                        depths.append(d)
                return pts, depths
            ###########################################################################################

            # 하단면과 상단면 모서리들을 이미지 좌표로 변환
            base_img, base_depths = proj(base_corners_world)
            top_img, top_depths = proj(top_corners_world)

            # 가드레일 정면 1개 면만 그리기 (2개 모서리로 간단화)
            # 모든 모서리가 유효한지 확인 (2개씩)
            if all(p is not None for p in base_img) and all(p is not None for p in top_img):
                # 정면을 사각형으로 정의 (하단 -> 상단 순서)
                quad = np.array([base_img[0], base_img[1], top_img[1], top_img[0]], dtype=np.int32)
                
                # 거리에 따른 색상 조정 (멀수록 어둡게)
                face_depth = (top_depths[0] if len(top_depths) > 0 else 30.0)
                distance_factor = min(1.0, max(0.3, 15.0 / face_depth))
                body_color = int(192 * distance_factor)
                edge_color = int(128 * distance_factor)
                
                # 정면을 렌더링 큐에 추가
                commands.append({
                    'type': 'poly', 
                    'points': quad, 
                    'fill': (body_color, body_color, body_color),  # 회색 채우기
                    'edge': (edge_color, edge_color, edge_color),  # 어두운 회색 테두리
                    'th': 2, 
                    'depth': face_depth
                })


    def render_vehicles(self, image, commands):
        """다른 차량들을 이미지에 그리기"""
        
        if self.surrounding_vehicles is None:
            return
            
        for vehicle in self.surrounding_vehicles.objects:
            # Vehicle oriented cuboid based on pose.size (length x, width y, height z)
            center_x = vehicle.pose.position.x
            center_y = vehicle.pose.position.y
            center_z = vehicle.pose.position.z
            yaw = self.quaternion_to_yaw(vehicle.pose.orientation)
            length = getattr(vehicle.size, 'x', 4.0)
            width = getattr(vehicle.size, 'y', 1.8)
            height = getattr(vehicle.size, 'z', 1.6)

            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            half_length = length / 2.0
            half_width = width / 2.0

            # Bottom rectangle (z = center_z)
            bottom_world = [
                (center_x - half_length * cos_yaw + half_width * sin_yaw, center_y - half_length * sin_yaw - half_width * cos_yaw, center_z),  # FL
                (center_x + half_length * cos_yaw + half_width * sin_yaw, center_y + half_length * sin_yaw - half_width * cos_yaw, center_z),  # FR
                (center_x + half_length * cos_yaw - half_width * sin_yaw, center_y + half_length * sin_yaw + half_width * cos_yaw, center_z),  # RR
                (center_x - half_length * cos_yaw - half_width * sin_yaw, center_y - half_length * sin_yaw + half_width * cos_yaw, center_z),  # RL
            ]
            top_world = [(x, y, center_z + height) for (x, y, _) in bottom_world]

            ###########################################################################################
            # Todo
            def proj(world_pts):
                pts = []
                depths = []
                for (world_point_x, world_point_y, world_point_z) in world_pts:
                    #
                    u, v, d = self.world_to_image_coordinates(world_point_x, world_point_y, world_point_z)
                    if u is None or d <= 0:
                        pts.append(None)
                    else:
                        pts.append((u, v))
                        depths.append(d)
                return pts, depths
            ###########################################################################################
            
            bottom_img, bottom_depths = proj(bottom_world)
            top_img, top_depths = proj(top_world)
            # 차량의 측면을 그리기 위한 면(face) 인덱스 정의
            # 각 튜플은 (시작점, 끝점)을 나타내며, 사각형의 인접한 두 점을 연결
            # (0,1): FL->FR (전면 좌측 -> 전면 우측)
            # (1,2): FR->RR (전면 우측 -> 후면 우측) 
            # (2,3): RR->RL (후면 우측 -> 후면 좌측)
            # (3,0): RL->FL (후면 좌측 -> 전면 좌측)
            faces = [(0, 1), (1, 2), (2, 3), (3, 0)]
            
            # 각 측면을 사각형(quad)으로 그리기
            for (i, j) in faces:
                # 현재 면을 구성하는 4개 점이 모두 유효한지 확인
                # bottom_img[i/j]: 바닥면의 시작/끝 점
                # top_img[i/j]: 상단면의 시작/끝 점
                if bottom_img[i] is None or bottom_img[j] is None or top_img[i] is None or top_img[j] is None:
                    continue  # 하나라도 None이면 이 면은 그리지 않음
                
                # 측면 사각형의 4개 꼭짓점 정의 (시계방향 또는 반시계방향)
                # bottom_img[i] -> bottom_img[j] -> top_img[j] -> top_img[i] 순서로 연결
                quad = np.array([bottom_img[i], bottom_img[j], top_img[j], top_img[i]], dtype=np.int32)
                
                # 측면의 깊이 계산 (거리 기반 음영 효과를 위해)
                # top_depths가 있으면 평균 사용, 없으면 기본값 30.0 사용
                face_depth = np.mean(top_depths) if len(top_depths) > 0 else 30.0
                
                # 거리 기반 조명 효과 계산 (거리 팩터)
                # 가까운 객체는 밝게(1.0에 가까움), 먼 객체는 어둡게(0.25에 가까움)
                # 20.0은 조명 강도 조절 상수
                df = min(1.0, max(0.25, 20.0 / face_depth))
                
                # 측면 렌더링 명령을 큐에 추가
                # fill: 면 채우기 색상 (파란색 계열, 거리에 따라 밝기 조절)
                # edge: 테두리 색상 (더 어두운 파란색)
                # th: 테두리 두께
                # depth: 깊이 정보 (페인터 알고리즘 정렬용)
                commands.append({
                    'type': 'poly', 
                    'points': quad, 
                    'fill': (0, 0, int(190 * df)), 
                    'edge': (0, 0, int(140 * df)), 
                    'th': 2, 
                    'depth': face_depth
                })

            # 차량의 상단면(지붕) 그리기
            # 상단면의 모든 점이 유효한지 확인
            if all(p is not None for p in top_img):
                # 상단면 다각형 생성 (일반적으로 4각형)
                top_poly = np.array(top_img, dtype=np.int32)
                
                # 상단면의 평균 깊이 계산
                td = np.mean(top_depths) if len(top_depths) > 0 else 30.0
                
                # 상단면용 거리 기반 조명 효과 계산
                # 측면보다 약간 밝게 표현하기 위해 동일한 공식 사용
                df = min(1.0, max(0.25, 20.0 / td))
                
                # 상단면 렌더링 명령을 큐에 추가
                # fill: 측면보다 밝은 파란색 (210 vs 190)
                # edge: 테두리는 측면보다 밝은 색상 (150 vs 140)
                commands.append({
                    'type': 'poly', 
                    'points': top_poly, 
                    'fill': (0, 0, int(210 * df)), 
                    'edge': (0, 0, int(150 * df)), 
                    'th': 2, 
                    'depth': td
                })
                
    def render_traffic_signs(self, image, commands):
        """교통 표지판들을 이미지에 그리기 (occlusion-aware)"""
        # 교통 표지판 데이터가 없으면 렌더링 생략
        if self.traffic_signs is None:
            return
            
        # TrafficSignArray.msg의 'signs' 필드 사용
        for sign in getattr(self.traffic_signs, 'signs', []):
            ###########################################################################################
            # TODO
            # 교통 표지판의 월드 좌표를 이미지 좌표로 변환
            u, v, depth = self.world_to_image_coordinates(sign.pose.position.x, sign.pose.position.y, sign.pose.position.z)
            ###########################################################################################
            
            # 이미지 좌표가 유효하고 카메라 앞쪽에 있는지 확인
            if u is not None and depth is not None and depth > 0:
                # 표지판의 yaw 각도 계산 (quaternion에서 yaw 추출)
                sign_yaw = self.quaternion_to_yaw(sign.pose.orientation)
                
                # 차량에서 표지판을 바라보는 방향 계산
                vehicle_x = self.vehicle_state.pose.position.x
                vehicle_y = self.vehicle_state.pose.position.y
                sign_x = sign.pose.position.x
                sign_y = sign.pose.position.y
                
                # 차량 위치에서 표지판 위치로의 방향각 계산
                viewing_direction = math.atan2(sign_y - vehicle_y, sign_x - vehicle_x)
                
                # 표지판의 정면 방향과 차량이 바라보는 방향 사이의 각도차 계산
                # 0도: 표지판이 차량을 정면으로 향함, 180도: 표지판이 차량에게 등을 돌림
                relative_angle = sign_yaw - viewing_direction
                # 각도를 [-π, π] 범위로 정규화
                while relative_angle > math.pi:
                    relative_angle -= 2 * math.pi
                while relative_angle < -math.pi:
                    relative_angle += 2 * math.pi
                # 거리에 따른 표지판 크기 조절 (거리가 멀수록 작게)
                # 가까운 거리(1m): 반지름 30픽셀, 먼 거리(30m): 반지름 5픽셀
                base_radius = 100.0
                min_radius = 5
                max_radius = 5000
                radius = int(base_radius / max(1.0, depth))
                radius = max(min_radius, min(max_radius, radius))
                
                # 표지판 방향에 따른 타원 형태 계산
                # 정면(0도): 원형, 측면(90도): 매우 납작한 타원
                angle_factor = abs(math.cos(relative_angle))  # 0~1 범위
                # 뒷면일 때도 최소 가시성 유지
                visibility_factor = max(0.1, angle_factor)
                
                # 타원의 장축과 단축 계산
                major_axis = radius  # 장축은 원래 크기 유지
                minor_axis = int(radius * visibility_factor)  # 단축은 각도에 따라 줄어듦
                
                # sign_class에 따른 색상 설정 (BGR 형식)
                if sign.sign_class == 1:  # STOP
                    base_fill_color = (0, 0, 255)     # 빨간색
                    base_edge_color = (255, 255, 255)  # 흰색
                elif sign.sign_class == 2:  # YIELD
                    base_fill_color = (0, 255, 255)   # 노란색
                    base_edge_color = (0, 0, 0)       # 검은색
                elif sign.sign_class in [3, 4, 5, 6]:  # SPEED_LIMIT_30/50/60/80
                    base_fill_color = (255, 255, 255) # 흰색
                    base_edge_color = (0, 0, 255)     # 빨간색
                elif sign.sign_class == 7:  # NO_ENTRY
                    base_fill_color = (255, 255, 255) # 흰색
                    base_edge_color = (0, 0, 255)     # 빨간색
                elif sign.sign_class == 8:  # NO_PARKING
                    base_fill_color = (255, 255, 255) # 흰색
                    base_edge_color = (0, 0, 255)     # 빨간색
                elif sign.sign_class == 9:  # PEDESTRIAN_CROSSING
                    base_fill_color = (0, 255, 0)     # 초록색
                    base_edge_color = (255, 255, 255) # 흰색
                elif sign.sign_class == 10:  # SCHOOL_ZONE
                    base_fill_color = (0, 255, 255)   # 노란색
                    base_edge_color = (0, 0, 0)       # 검은색
                elif sign.sign_class == 11:  # CONSTRUCTION
                    base_fill_color = (0, 165, 255)   # 주황색
                    base_edge_color = (0, 0, 0)       # 검은색
                elif sign.sign_class in [12, 13, 14]:  # TURN_LEFT/RIGHT, GO_STRAIGHT
                    base_fill_color = (255, 0, 0)     # 파란색
                    base_edge_color = (255, 255, 255) # 흰색
                else:  # UNKNOWN or others
                    base_fill_color = (128, 128, 128) # 회색
                    base_edge_color = (255, 255, 255) # 흰색
                
                # 각도에 따른 색상 조절 (뒷면일수록 어둡게)
                brightness_factor = max(0.3, visibility_factor)  # 최소 30% 밝기 유지
                fill_color = tuple(int(c * brightness_factor) for c in base_fill_color)
                edge_color = tuple(int(c * brightness_factor) for c in base_edge_color)
                
                # 타원형 표지판 렌더링 명령을 큐에 추가
                commands.append({
                    'type': 'ellipse', 
                    'center': (u, v), 
                    'axes': (major_axis, minor_axis),  # (장축, 단축)
                    'angle': 90,  # 회전 각도 (수평 타원)
                    'fill': fill_color, 
                    'edge': edge_color, 
                    'th': 2, 
                    'depth': depth
                })

    def render_traffic_lights(self, image, commands):
        """신호등들을 이미지에 그리기 (occlusion-aware)"""
        # 신호등 데이터가 없으면 렌더링 생략
        if self.traffic_lights is None:
            return
            
        # 각 신호등에 대해 처리
        for light in self.traffic_lights.lights:
            ###########################################################################################
            # TODO
            # 신호등의 월드 좌표를 이미지 좌표로 변환
            u, v, depth = self.world_to_image_coordinates(light.pose.position.x, light.pose.position.y, light.pose.position.z)
            ###########################################################################################
            
            # 이미지 좌표가 유효하고 카메라 앞쪽에 있는지 확인
            if u is not None and depth is not None and depth > 0:
                # 거리에 따른 신호등 크기 조절 (거리가 멀수록 작게)
                # 가까운 거리(1m): 반지름 25픽셀, 먼 거리(25m): 반지름 4픽셀
                base_radius = 25.0
                min_radius = 4
                max_radius = 4000  # 최대 크기 제한
                radius = int(base_radius / max(1.0, depth))
                radius = max(min_radius, min(max_radius, radius))
                
                # 신호등 상태에 따른 색상 설정 (BGR 형식)
                if light.state == 0:  # RED
                    light_color = (0, 0, 255)     # 빨간색
                elif light.state == 1:  # YELLOW
                    light_color = (0, 255, 255)   # 노란색
                else:  # GREEN (state == 2)
                    light_color = (0, 255, 0)     # 초록색
                
                # 신호등 배경(외곽) - 검은색 원 (더 큰 반지름)
                bg_radius = radius + 3  # 배경은 3픽셀 더 크게
                commands.append({
                    'type': 'circle', 
                    'center': (u, v), 
                    'radius': bg_radius, 
                    'fill': (0, 0, 0),           # 검은색 배경
                    'edge': (128, 128, 128),     # 회색 테두리
                    'th': 1, 
                    'depth': depth + 0.001       # 배경이 뒤에 그려지도록
                })
                
                # 신호등 램프 - 상태별 색상 원
                commands.append({
                    'type': 'circle', 
                    'center': (u, v), 
                    'radius': radius, 
                    'fill': light_color,         # 상태별 색상
                    'edge': (0, 0, 0),     # 흰색 테두리
                    'th': 0, 
                    'depth': depth
                })

    def draw_commands(self, image, commands):
        """Draw queued primitives sorted by depth (far -> near).

        Implements a simple painter's algorithm: deeper objects are filled first,
        then closer ones are drawn on top. Each command may be a polygon, circle, ellipse, or line.
        """
        if not commands:
            return
        # 큰 depth가 먼 곳. 먼 것부터 그리기 위해 내림차순 정렬
        commands.sort(key=lambda c: c.get('depth', 0.0), reverse=True)
        for cmd in commands:
            if cmd.get('type') == 'poly':
                pts = cmd['points']
                fill = cmd.get('fill', (255, 255, 255))
                edge = cmd.get('edge', None)
                th = cmd.get('th', 1)
                cv2.fillPoly(image, [pts], fill)
                if edge is not None:
                    cv2.polylines(image, [pts], isClosed=True, color=edge, thickness=th)
            elif cmd.get('type') == 'circle':
                center = cmd['center']
                radius = cmd['radius']
                fill = cmd.get('fill', (255, 255, 255))
                edge = cmd.get('edge', None)
                th = cmd.get('th', 1)
                cv2.circle(image, center, radius, fill, -1)
                if edge is not None:
                    cv2.circle(image, center, radius, edge, th)
            elif cmd.get('type') == 'ellipse':
                center = cmd['center']
                axes = cmd['axes']  # (장축, 단축)
                angle = cmd.get('angle', 0)  # 회전 각도
                fill = cmd.get('fill', (255, 255, 255))
                edge = cmd.get('edge', None)
                th = cmd.get('th', 1)
                # OpenCV ellipse: (center, (major_axis, minor_axis), angle, start_angle, end_angle, color, thickness)
                cv2.ellipse(image, center, axes, angle, 0, 360, fill, -1)
                if edge is not None:
                    cv2.ellipse(image, center, axes, angle, 0, 360, edge, th)
            elif cmd.get('type') == 'line':
                start = cmd['start']
                end = cmd['end']
                color = cmd.get('color', (255, 255, 255))
                th = cmd.get('th', 1)
                cv2.line(image, start, end, color, th)

    
    def add_camera_info_text(self, image):
        """이미지에 카메라 정보 텍스트 추가"""
        
        if self.vehicle_state is None:
            return
        
        # 차량 위치 정보
        x = self.vehicle_state.pose.position.x
        y = self.vehicle_state.pose.position.y
        yaw_deg = math.degrees(self.quaternion_to_yaw(self.vehicle_state.pose.orientation))
        
        # 카메라 월드 위치 계산 (디버그 정보용)
        cos_vehicle_yaw = math.cos(yaw_deg * math.pi / 180.0)
        sin_vehicle_yaw = math.sin(yaw_deg * math.pi / 180.0)
        
        camera_world_x = x + (self.camera_offset_x * cos_vehicle_yaw - self.camera_offset_y * sin_vehicle_yaw)
        camera_world_y = y + (self.camera_offset_x * sin_vehicle_yaw + self.camera_offset_y * cos_vehicle_yaw)
        
        # 텍스트 정보들
        info_lines = [
            f"Vehicle: ({x:.1f}, {y:.1f}) @ {yaw_deg:.1f}deg",
            f"Camera: ({camera_world_x:.1f}, {camera_world_y:.1f}, {self.camera_offset_z:.1f})",
            f"Offset: X={self.camera_offset_x:.1f}m Y={self.camera_offset_y:.1f}m Z={self.camera_offset_z:.1f}m",
            f"FOV: {self.camera_fov_deg:.1f}deg Pitch: {self.camera_pitch_deg:.1f}deg"
        ]
        
        # 텍스트를 이미지 왼쪽 상단에 추가
        y_offset = 30
        for line in info_lines:
            cv2.putText(
                image,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # 흰색
                1,
                cv2.LINE_AA
            )
            y_offset += 20

    def callback_timer(self, event):
        """주기적으로 호출되는 이미지 생성 함수"""
        
        # 차량 상태가 없으면 대기
        if self.vehicle_state is None or not self.enable_sensor:
            return
        
        # Create empty image
        image = np.zeros(
            (self.image_height, self.image_width, 3),
            dtype=np.uint8
        )

        # Draw background first (sky and ground)
        self.render_background(image)
        
        # Occlusion-aware rendering queue
        commands = []
        # Enqueue geometry with depth
        self.render_buildings(image, commands)
        self.render_guardrails(image, commands)
        self.render_vehicles(image, commands)
        self.render_traffic_signs(image, commands)
        self.render_traffic_lights(image, commands)
        # Draw lane lines first (they are on ground and should appear on top of ground but under objects)
        self.render_lanes(image)
        # Execute painter's algorithm
        self.draw_commands(image, commands)
        
        # 카메라 정보 텍스트 추가
        self.add_camera_info_text(image)
        
        # ROS 이미지 메시지로 변환하여 발행
        try:
            # OpenCV 이미지를 ROS 이미지 메시지로 변환
            ros_image = self.cv_bridge.cv2_to_imgmsg(image, "bgr8")
            
            # 헤더 정보 설정
            ros_image.header = Header()
            ros_image.header.stamp = rospy.Time.now()
            ros_image.header.frame_id = "camera"
            
            # 이미지 발행
            self.image_publisher.publish(ros_image)
            
        except Exception as e:
            rospy.logerr(f"이미지 발행 중 오류: {e}")


def main():
    """메인 함수"""
    
    # ROS 노드 초기화
    rospy.init_node("sensor_camera_node")
    
    try:
        # 카메라 시뮬레이터 생성
        camera_sim = CameraSimulator()

        # ROS 스핀 (무한 대기)
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"오류 발생: {e}")
        pass


if __name__ == "__main__":
    main()


