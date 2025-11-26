#!/usr/bin/env python
import math
import struct
import numpy as np
import time
from typing import List, Tuple

import rospy
from dcas_msgs.msg import ObjectArray, VehicleState, TrafficSignArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

"""
레이더 센서 시뮬레이터 메인 함수

알고리즘 프로세스 흐름:
1. ROS 노드 초기화 (rospy.init_node)
2. RadarEmu 클래스 인스턴스 생성 (__init__ 호출)
    - 파라미터 로드 및 설정
    - ROS 퍼블리셔/구독자 설정
    - 타이머 생성 (주기적 CallbackTimer 실행)
3. rospy.spin()으로 이벤트 루프 시작

타이머 콜백(CallbackTimer)에서 실행되는 주요 프로세스:
1. _process_vehicles: 주변 차량들에 대한 레이더 포인트 생성
2. _process_buildings: 건물 표면과의 교차점에서 레이더 포인트 생성
3. _process_traffic_signs: 교통표지판에 대한 레이더 포인트 생성
4. _process_guardrails: 가드레일을 따라 레이더 포인트 생성
5. _process_clutter: 클러터(허위 양성) 포인트 생성
6. _publish_results: 생성된 모든 포인트를 PointCloud2로 퍼블리시

각 프로세스에서 공통적으로 수행되는 작업:
- 좌표계 변환 (MapToRadarXy): 맵 프레임 → 레이더 프레임
- FOV 검사 (InRadarFov): 레이더 시야각 내 객체 필터링
- 거리 필터링: 최대 감지 범위 내 객체만 처리
- 해상도 양자화: 거리, 각도, 속도 해상도 적용
- 노이즈 적용 (ApplyNoise): 가우시안 노이즈 추가
- 전력 계산 (CalculatePower): 레이더 방정식 기반 수신 전력 계산
- 감지 판정 (ShouldDetect): 전력 임계값과 확률 기반 감지 결정
"""
class RadarEmu:
    """대향 차량/환경 객체를 사용한 간단한 레이더 시뮬레이션.

    - `/env/surrounding_vehicles`를 구독하고 객체당 하나의 감지를 생성합니다.
    - Range = 평면 거리, Doppler = 공칭 접근 속도의 투영입니다.
    """
    
    # ===============================
    # 1. 초기화 및 콜백 함수들
    # ===============================
    
    # 레이더 에뮬레이터 초기화 - 파라미터 로드, 퍼블리셔/구독자 설정, 타이머 생성
    def __init__(self) -> None:
        # 센서 활성화 여부 확인
        self.enabled = rospy.get_param("/sensor_radar_node/enabled", True)
        if not self.enabled:
            rospy.loginfo("Radar sensor is disabled")
            return
        
        # 퍼블리셔들
        self.pub_pc = rospy.Publisher("/sensors/radar/points", PointCloud2, queue_size=10)  # GT 포인트들
        self.pub_pc_noise = rospy.Publisher("/sensors/radar/points_noise", PointCloud2, queue_size=10)  # 노이즈가 있는 포인트들
        self.pub_fov = rospy.Publisher("/sensors/radar/fov", MarkerArray, queue_size=10)

        # 구독자들
        self.vehicles: ObjectArray = None  # type: ignore
        self.buildings: ObjectArray = None  # type: ignore
        self.guardrails: ObjectArray = None  # type: ignore
        self.signs: TrafficSignArray = None  # type: ignore
        self.state: VehicleState = None  # type: ignore
        self._prev_targets = {}  # id -> (x, y, t)
        rospy.Subscriber("/env/surrounding_vehicles", ObjectArray, self.CallbackVehicles)
        rospy.Subscriber("/env/buildings", ObjectArray, self.CallbackBuildings)
        rospy.Subscriber("/env/guardrails", ObjectArray, self.CallbackGuardrails)
        rospy.Subscriber("/env/traffic_signs", TrafficSignArray, self.CallbackSigns)
        rospy.Subscriber("/vehicle/state", VehicleState, self.CallbackState)

        # 파라미터들
        period = float(rospy.get_param("/sensor_radar_node/period_s", 0.1))
        self.guardrail_spacing = rospy.get_param("/sensor_radar_node/guardrail_spacing", 2.0)
        # 레이더 외부 파라미터 (ego_frame에 상대적) 및 FOV
        self.radar_tx = rospy.get_param("/sensor_radar_node/offset_x", 1.0)
        self.radar_ty = rospy.get_param("/sensor_radar_node/offset_y", 0.0)
        self.radar_tz = rospy.get_param("/sensor_radar_node/offset_z", 0.5)
        self.radar_roll_deg = rospy.get_param("/sensor_radar_node/roll_deg", 0.0)
        self.radar_pitch_deg = rospy.get_param("/sensor_radar_node/pitch_deg", 0.0)
        self.radar_yaw_deg = rospy.get_param("/sensor_radar_node/yaw_deg", 0.0)
        self.fov_deg = rospy.get_param("/sensor_radar_node/fov_deg", 120.0)
        self.max_range = rospy.get_param("/sensor_radar_node/max_range", 100.0)
        
        # 신호 파라미터들
        self.range_resolution = rospy.get_param("/sensor_radar_node/range_resolution", 0.15)
        self.velocity_resolution = rospy.get_param("/sensor_radar_node/velocity_resolution", 0.1)
        self.angle_resolution = rospy.get_param("/sensor_radar_node/angle_resolution", 1.0)
        
        # 레이더 하드웨어 파라미터들 (전력 계산용)
        self.transmit_power_dbm = rospy.get_param("/sensor_radar_node/transmit_power_dbm", 20.0)
        self.transmit_antenna_gain_db = rospy.get_param("/sensor_radar_node/transmit_antenna_gain_db", 25.0)
        self.receive_antenna_gain_db = rospy.get_param("/sensor_radar_node/receive_antenna_gain_db", 25.0)
        self.frequency_ghz = rospy.get_param("/sensor_radar_node/frequency_ghz", 77.0)
        self.antenna_aperture_m2 = rospy.get_param("/sensor_radar_node/antenna_aperture_m2", 0.01)
        self.system_losses_db = rospy.get_param("/sensor_radar_node/system_losses_db", 10.0)
        self.noise_figure_db = rospy.get_param("/sensor_radar_node/noise_figure_db", 8.0)
        
        # 전력 임계값 파라미터들
        self.power_threshold_dbm = rospy.get_param("/sensor_radar_node/power_threshold_dbm", -90.0)
        self.detection_probability = rospy.get_param("/sensor_radar_node/detection_probability", 0.9)
        
        # 레거시 RCS 임계값 (호환성을 위해 유지)
        self.rcs_threshold = rospy.get_param("/sensor_radar_node/rcs_threshold", 1.0)
        
        # 가우시안 노이즈 파라미터들
        self.range_noise_std = rospy.get_param("/sensor_radar_node/range_noise_std", 0.1)  # 미터
        self.azimuth_noise_std = rospy.get_param("/sensor_radar_node/azimuth_noise_std", 0.5)  # 도
        self.enable_noise = rospy.get_param("/sensor_radar_node/enable_noise", True)
        
        # 클러터 노이즈 파라미터들
        self.enable_clutter = rospy.get_param("/sensor_radar_node/enable_clutter", True)
        self.clutter_probability = rospy.get_param("/sensor_radar_node/clutter_probability", 0.05)  # 0.0-1.0
        self.clutter_density = rospy.get_param("/sensor_radar_node/clutter_density", 2.0)  # 발생당 평균 포인트 수
        self.clutter_range_min = rospy.get_param("/sensor_radar_node/clutter_range_min", 5.0)  # 미터
        self.clutter_range_max = rospy.get_param("/sensor_radar_node/clutter_range_max", 45.0)  # 미터
        
        # 타이머 (센서가 활성화된 경우에만 생성)
        if self.enabled:
            self.timer = rospy.Timer(rospy.Duration(period), self.CallbackTimer)
        
        # 공유 센서 파라미터 로드
        self.object_props = {
            'vehicle': {
                'base_rcs': rospy.get_param('/sensor_shared_params/object_properties/vehicle/base_rcs', 10.0),
                'reflectivity': rospy.get_param('/sensor_shared_params/object_properties/vehicle/reflectivity', 0.3)
            },
            'building': {
                'base_rcs': rospy.get_param('/sensor_shared_params/object_properties/building/base_rcs', 100.0),
                'reflectivity': rospy.get_param('/sensor_shared_params/object_properties/building/reflectivity', 0.8)
            },
            'guardrail': {
                'base_rcs': rospy.get_param('/sensor_shared_params/object_properties/guardrail/base_rcs', 5.0),
                'reflectivity': rospy.get_param('/sensor_shared_params/object_properties/guardrail/reflectivity', 0.9)
            },
            'traffic_sign': {
                'base_rcs': rospy.get_param('/sensor_shared_params/object_properties/traffic_sign/base_rcs', 8.0),
                'reflectivity': rospy.get_param('/sensor_shared_params/object_properties/traffic_sign/reflectivity', 0.7)
            },
            'ground': {
                'base_rcs': rospy.get_param('/sensor_shared_params/object_properties/ground/base_rcs', 0.05),
                'reflectivity': rospy.get_param('/sensor_shared_params/object_properties/ground/reflectivity', 0.1)
            },
            'lane_marking': {
                'base_rcs': rospy.get_param('/sensor_shared_params/object_properties/lane_marking/base_rcs', 0.08),
                'reflectivity': rospy.get_param('/sensor_shared_params/object_properties/lane_marking/reflectivity', 0.8)
            }
        }
        
        # 레이더 방정식용 계산된 상수들
        self.speed_of_light = 3e8  # m/s
        self.wavelength = self.speed_of_light / (self.frequency_ghz * 1e9)  # meters
        
        # dBm을 와트로 변환하는 헬퍼 (P_watts = 10^((P_dBm - 30) / 10))
        self.transmit_power_watts = 10**((self.transmit_power_dbm - 30) / 10)
        
        # dB를 리니어 스케일로 변환
        self.transmit_gain_linear = 10**(self.transmit_antenna_gain_db / 10)
        self.receive_gain_linear = 10**(self.receive_antenna_gain_db / 10)
        self.system_losses_linear = 10**(self.system_losses_db / 10)
        
        # 클러터 파라미터들
        self.clutter_rcs_min = rospy.get_param('/sensor_shared_params/clutter/rcs_min', 0.1)
        self.clutter_rcs_max = rospy.get_param('/sensor_shared_params/clutter/rcs_max', 1.0)
        
        # 재현 가능한 노이즈를 위한 난수 생성기
        self.rng = np.random.RandomState(int(time.time() * 1000) % 2**32)

    # 주변 차량 정보 수신 콜백 - 차량 데이터 저장
    def CallbackVehicles(self, msg: ObjectArray) -> None:
        self.vehicles = msg

    # 건물 정보 수신 콜백 - 건물 데이터 저장
    def CallbackBuildings(self, msg: ObjectArray) -> None:
        self.buildings = msg

    # 교통표지판 정보 수신 콜백 - 표지판 데이터 저장
    def CallbackSigns(self, msg) -> None:
        self.signs = msg

    # 자차 상태 정보 수신 콜백 - 위치, 속도 등 저장
    def CallbackState(self, msg: VehicleState) -> None:
        self.state = msg

    # 가드레일 정보 수신 콜백 - 가드레일 데이터 저장
    def CallbackGuardrails(self, msg: ObjectArray) -> None:
        self.guardrails = msg

    # ===============================
    # 2. 메인 타이머 콜백 및 스캔 생성 함수들
    # ===============================
    
    # 타이머 콜백 - 주기적으로 레이더 스캔 생성
    def CallbackTimer(self, _) -> None:
        if not self.enabled:
            return
        
        now = rospy.Time.now()

        # 레이더 포인트 클라우드 구축 (x, y, z, power_dbm, doppler_velocity, rcs)
        radar_points_gt: List[Tuple[float, float, float, float, float, float]] = []  # 실측값
        radar_points_noise: List[Tuple[float, float, float, float, float, float]] = []  # 노이즈 포함

        # 차량 처리
        self._process_vehicles(radar_points_gt, radar_points_noise, now)
        
        # 건물 처리
        self._process_buildings(radar_points_gt, radar_points_noise, now)
        
        # 교통표지판 처리
        self._process_traffic_signs(radar_points_gt, radar_points_noise, now)

        # 가드레일 처리
        self._process_guardrails(radar_points_gt, radar_points_noise, now)
        
        # 클러터 처리
        self._process_clutter(radar_points_gt, radar_points_noise, now)
        
        # 결과 퍼블리시
        self._publish_results(radar_points_gt, radar_points_noise, now)

    
    # ===============================
    # 3. 각 객체 처리 함수들
    # ===============================
    
    # 차량 객체 처리 - 각 차량에 대해 레이더 포인트 생성 (거리, 도플러, RCS 계산)
    def _process_vehicles(self, radar_points_gt: List[Tuple[float, float, float, float, float, float]],
                         radar_points_noise: List[Tuple[float, float, float, float, float, float]],
                         now: rospy.Time) -> None:
        """차량 처리 로직"""
        if not self.vehicles:
            return
            
        for obj in self.vehicles.objects:
            # 표면 포인트 계산을 위해 맵 프레임에서 레이더 위치 가져오기
            radar_x_map = 0.0
            radar_y_map = 0.0
            if self.state is not None:
                ex = self.state.pose.position.x
                ey = self.state.pose.position.y
                qw = self.state.pose.orientation.w
                qz = self.state.pose.orientation.z
                ego_yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
                # ego 프레임에서 맵 프레임으로 레이더 오프셋 변환
                ryaw = math.radians(self.radar_yaw_deg)
                cos_ego = math.cos(ego_yaw)
                sin_ego = math.sin(ego_yaw)
                cos_r = math.cos(ego_yaw + ryaw)
                sin_r = math.sin(ego_yaw + ryaw)
                radar_x_map = ex + cos_ego * self.radar_tx - sin_ego * self.radar_ty
                radar_y_map = ey + sin_ego * self.radar_tx + cos_ego * self.radar_ty
            
            # 중심 대신 바운딩 박스 표면의 가장 가까운 점 가져오기
            closest_x, closest_y = self.GeoClosestBboxPoint(obj, radar_x_map, radar_y_map)
            x_m = closest_x
            y_m = closest_y
            z = 0.0  # 2D 레이더 - 모든 포인트가 레이더 센서 높이에 있음
            
            # 맵 -> 레이더 프레임
            x_radar, y_radar = self.MapToRadarXy(x_m, y_m)
            # FOV 필터 (전방)
            if not self.InRadarFov(x_radar, y_radar):
                continue
            
            # 거리 필터 - 최대 범위 내에 있는지 확인
            range_gt = math.hypot(x_radar, y_radar)
            if range_gt > self.max_range:
                continue
            
            # 거리 해상도 양자화 적용
            range_quantized = self.QuantizeRange(range_gt)
            angle_original = math.atan2(y_radar, x_radar)
            
            # 각도 해상도 양자화 적용  
            angle_quantized = self.QuantizeAngle(angle_original)
            
            range_epsilon = 1e-6  # 불필요한 재계산을 피하기 위한 작은 값
            if abs(range_quantized - range_gt) > range_epsilon or abs(angle_quantized - angle_original) > range_epsilon:
                # 양자화된 거리와 각도를 기반으로 x_radar, y_radar 재계산
                x_radar = range_quantized * math.cos(angle_quantized)
                y_radar = range_quantized * math.sin(angle_quantized)
            
            # 레이더 측정값에 노이즈 적용
            x_radar_noisy, y_radar_noisy = self.ApplyNoise(x_radar, y_radar)
            range_noisy = math.hypot(x_radar_noisy, y_radar_noisy)

            # 맵 프레임에서의 ego 속도
            if self.state is not None:
                qw = self.state.pose.orientation.w
                qz = self.state.pose.orientation.z
                # 쿼터니언에서 yaw (평면 가정)
                yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
                v_ego = self.state.twist.linear.x
                v_ex = v_ego * math.cos(yaw)
                v_ey = v_ego * math.sin(yaw)
            else:
                v_ex = 0.0
                v_ey = 0.0

            # 목표 속도 (맵 프레임): 제공된 twist를 우선 사용, 유한 차분법을 폴백으로 사용
            vx_t = obj.twist.linear.x
            vy_t = obj.twist.linear.y
            if abs(vx_t) < 1e-6 and abs(vy_t) < 1e-6:
                t_prev = self._prev_targets.get(obj.id)
                if t_prev is not None:
                    px, py, pt = t_prev
                    dt = max(1e-3, (now.to_sec() - pt))
                    vx_t = (x_m - px) / dt
                    vy_t = (y_m - py) / dt
            # 다음 차분을 위해 현재 맵 위치 저장
            self._prev_targets[obj.id] = (x_m, y_m, now.to_sec())

            # ego(base)에서 목표까지의 시선 단위 벡터
            min_range_for_normalization = 1e-6  # 0으로 나누기 방지
            
            # 레이더 프레임에서의 LOS 단위 벡터 (감지용 노이즈 좌표 사용)
            range_norm_noisy = max(min_range_for_normalization, math.hypot(x_radar_noisy, y_radar_noisy))
            normal_x_noisy, normal_y_noisy = x_radar_noisy / range_norm_noisy, y_radar_noisy / range_norm_noisy
            
            # GT용 LOS 단위 벡터
            range_norm_gt = max(min_range_for_normalization, math.hypot(x_radar, y_radar))
            normal_x_gt, normal_y_gt = x_radar / range_norm_gt, y_radar / range_norm_gt

            # LOS에 투영된 상대 속도 (목표 - ego) -> 도플러
            relative_vel_x = vx_t - v_ex
            relative_vel_y = vy_t - v_ey
            # 상대 속도를 map->base->radar로 회전 (평행이동은 영향 없음)
            ego_yaw = 0.0
            if self.state is not None:
                qw = self.state.pose.orientation.w
                qz = self.state.pose.orientation.z
                ego_yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
            # map -> base
            cos_ego_yaw = math.cos(-ego_yaw)
            sin_ego_yaw = math.sin(-ego_yaw)
            relative_vel_base_x = cos_ego_yaw * relative_vel_x - sin_ego_yaw * relative_vel_y
            relative_vel_base_y = sin_ego_yaw * relative_vel_x + cos_ego_yaw * relative_vel_y
            # base -> radar
            radar_yaw_rad = math.radians(self.radar_yaw_deg)
            cos_radar_yaw = math.cos(-radar_yaw_rad)
            sin_radar_yaw = math.sin(-radar_yaw_rad)
            relative_vel_radar_x = cos_radar_yaw * relative_vel_base_x - sin_radar_yaw * relative_vel_base_y
            relative_vel_radar_y = sin_radar_yaw * relative_vel_base_x + cos_radar_yaw * relative_vel_base_y
            
            # GT와 노이즈 모두에 대해 도플러 계산
            doppler_noisy_raw = relative_vel_radar_x * normal_x_noisy + relative_vel_radar_y * normal_y_noisy
            doppler_gt_raw = relative_vel_radar_x * normal_x_gt + relative_vel_radar_y * normal_y_gt
            
            # 도플러에 속도 해상도 양자화 적용
            doppler_noisy = self.QuantizeVelocity(doppler_noisy_raw)
            doppler_gt = self.QuantizeVelocity(doppler_gt_raw)
            
            # 차량의 고유 RCS 사용 (물체의 고유 속성)
            rcs = self.object_props['vehicle']['base_rcs']
            # 레이더 방정식을 사용한 전력 계산 (GT 거리 사용)
            power_dbm = self.CalculatePower(range_norm_gt, rcs, 'vehicle')
            
            # 전력 임계값과 감지 확률 적용
            if not self.ShouldDetect(power_dbm):
                continue

            # 두 포인트 클라우드에 모두 추가  
            # 참고: 해상도 효과는 GT 포인트에서, 노이즈 효과는 노이즈 포인트에서 보임
            radar_points_noise.append((x_radar_noisy, y_radar_noisy, z, power_dbm, doppler_noisy, rcs))
            radar_points_gt.append((x_radar, y_radar, z, power_dbm, doppler_gt, rcs))

    # 건물 객체 처리 - 건물 표면과의 교차점에서 레이더 포인트 생성
    def _process_buildings(self, radar_points_gt: List[Tuple[float, float, float, float, float, float]],
                          radar_points_noise: List[Tuple[float, float, float, float, float, float]],
                          now: rospy.Time) -> None:
        """건물 처리 로직"""
        if not self.buildings:
            return
            
        for building in self.buildings.objects:
            # 맵 프레임에서 레이더 위치 가져오기
            radar_x_map = 0.0
            radar_y_map = 0.0
            if self.state is not None:
                ex = self.state.pose.position.x
                ey = self.state.pose.position.y
                qw = self.state.pose.orientation.w
                qz = self.state.pose.orientation.z
                ego_yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
                ryaw = math.radians(self.radar_yaw_deg)
                cos_ego = math.cos(ego_yaw)
                sin_ego = math.sin(ego_yaw)
                radar_x_map = ex + cos_ego * self.radar_tx - sin_ego * self.radar_ty
                radar_y_map = ey + sin_ego * self.radar_tx + cos_ego * self.radar_ty
            
            # 더 정확한 건물 표면 감지를 위해 LiDAR 스타일 교차점 사용
            building_hit = self.IntersectRadarWithBuilding(radar_x_map, radar_y_map, building)
            if building_hit is None:
                continue
            
            x_m, y_m = building_hit
            z = 0.0  # 2D 레이더 - 모든 포인트가 레이더 센서 높이에 있음
            
            # 레이더 프레임으로 변환
            x_radar, y_radar = self.MapToRadarXy(x_m, y_m)
            
            # FOV 필터
            if not self.InRadarFov(x_radar, y_radar):
                continue
            
            # 거리 필터
            range_gt = math.hypot(x_radar, y_radar)
            if range_gt > self.max_range:
                continue
            
            # 거리와 각도 해상도 적용
            range_quantized = self.QuantizeRange(range_gt)
            angle_original = math.atan2(y_radar, x_radar)
            angle_quantized = self.QuantizeAngle(angle_original)
            
            range_epsilon = 1e-6
            if abs(range_quantized - range_gt) > range_epsilon or abs(angle_quantized - angle_original) > range_epsilon:
                x_radar = range_quantized * math.cos(angle_quantized)
                y_radar = range_quantized * math.sin(angle_quantized)
            
            # 노이즈 적용
            x_radar_noisy, y_radar_noisy = self.ApplyNoise(x_radar, y_radar)
            
            # 건물의 고유 RCS 사용 및 레이더 방정식을 사용한 전력 계산
            rcs = self.object_props['building']['base_rcs']
            power_dbm = self.CalculatePower(range_gt, rcs, 'building')
            
            # 전력 임계값 적용
            if not self.ShouldDetect(power_dbm):
                continue
            
            # 건물은 정적이므로 도플러는 ego 모션에 기반
            if self.state is not None:
                qw = self.state.pose.orientation.w
                qz = self.state.pose.orientation.z
                yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
                v_ego = self.state.twist.linear.x
                v_ex = v_ego * math.cos(yaw)
                v_ey = v_ego * math.sin(yaw)
                
                # ego 속도를 레이더 프레임으로 변환
                cos_ego_yaw = math.cos(-yaw)
                sin_ego_yaw = math.sin(-yaw)
                ego_vel_base_x = cos_ego_yaw * v_ex - sin_ego_yaw * v_ey
                ego_vel_base_y = sin_ego_yaw * v_ex + cos_ego_yaw * v_ey
                radar_yaw_rad = math.radians(self.radar_yaw_deg)
                cos_radar_yaw = math.cos(-radar_yaw_rad)
                sin_radar_yaw = math.sin(-radar_yaw_rad)
                ego_vel_radar_x = cos_radar_yaw * ego_vel_base_x - sin_radar_yaw * ego_vel_base_y
                ego_vel_radar_y = sin_radar_yaw * ego_vel_base_x + cos_radar_yaw * ego_vel_base_y
                
                # LOS 단위 벡터 계산
                min_range_for_los = 1e-6
                range_norm_noisy = max(min_range_for_los, math.hypot(x_radar_noisy, y_radar_noisy))
                normal_x_noisy, normal_y_noisy = x_radar_noisy / range_norm_noisy, y_radar_noisy / range_norm_noisy
                
                range_norm_gt = max(min_range_for_los, math.hypot(x_radar, y_radar))
                normal_x_gt, normal_y_gt = x_radar / range_norm_gt, y_radar / range_norm_gt
                
                # 정적 건물의 도플러 (ego 속도 투영의 음수)
                doppler_noisy_raw = (0.0 - ego_vel_radar_x) * normal_x_noisy + (0.0 - ego_vel_radar_y) * normal_y_noisy
                doppler_gt_raw = (0.0 - ego_vel_radar_x) * normal_x_gt + (0.0 - ego_vel_radar_y) * normal_y_gt
                
                # 속도 해상도 적용
                doppler_noisy = self.QuantizeVelocity(doppler_noisy_raw)
                doppler_gt = self.QuantizeVelocity(doppler_gt_raw)
            else:
                doppler_noisy = 0.0
                doppler_gt = 0.0
            
            # 포인트 클라우드에 추가
            radar_points_noise.append((x_radar_noisy, y_radar_noisy, z, power_dbm, doppler_noisy, rcs))
            radar_points_gt.append((x_radar, y_radar, z, power_dbm, doppler_gt, rcs))

    # 건물 교차점 계산 - 레이더 빔과 건물 표면의 교차점 찾기 (면 컬링 적용)
    def IntersectRadarWithBuilding(self, radar_x_map: float, radar_y_map: float, building) -> Tuple[float, float]:
        """LiDAR 스타일 면 컬링을 사용하여 건물 표면의 교차점을 찾습니다.
        
        Args:
            radar_x_map: 맵 프레임에서 레이더 위치 x
            radar_y_map: 맵 프레임에서 레이더 위치 y
            building: 포즈와 크기/풋프린트를 가진 건물 객체
            
        Returns:
            맵 프레임에서 교차점의 (x, y) 좌표 튜플, 또는 None
        """
        # 성능을 위해 먼저 거리 확인
        building_distance = math.sqrt((building.pose.position.x - radar_x_map)**2 + 
                                    (building.pose.position.y - radar_y_map)**2)
        max_building_size = max(building.size.x, building.size.y)
        if building_distance > self.max_range + max_building_size:
            return None
        
        # 폴리곤 건물의 경우 정확한 폴리곤 교차점 사용
        if hasattr(building, 'footprint') and len(building.footprint) >= 3:
            return self.IntersectRadarWithPolygonBuilding(radar_x_map, radar_y_map, building)
        else:
            # 단순 박스 건물 - 가장 가까운 점 방법 사용
            return self.GeoClosestBboxPoint(building, radar_x_map, radar_y_map)
        
    # 폴리곤 건물 교차점 계산 - 복잡한 폴리곤 건물과 레이더 빔의 교차점 계산
    def IntersectRadarWithPolygonBuilding(self, radar_x_map: float, radar_y_map: float, building) -> Tuple[float, float]:
        """Calculate intersection with polygon building using face culling.
        
        Args:
            radar_x_map: Radar position x in map frame
            radar_y_map: Radar position y in map frame
            building: Building object with footprint
            
        Returns:
            Tuple of (x, y) coordinates of closest visible wall intersection, or None
        """
        footprint_world = [(p.x, p.y) for p in building.footprint]
        
        closest_intersection = None
        closest_distance = float('inf')
        
        # 폴리곤 건물의 각 벽 확인
        for i in range(len(footprint_world)):
            p1_world = footprint_world[i]
            p2_world = footprint_world[(i + 1) % len(footprint_world)]
            
            # 면 컬링을 위한 벽 방향과 중심
            wall_dx = p2_world[0] - p1_world[0]
            wall_dy = p2_world[1] - p1_world[1]
            wall_center_x = (p1_world[0] + p2_world[0]) * 0.5
            wall_center_y = (p1_world[1] + p2_world[1]) * 0.5
            
            # 벽 중심에서 레이더로의 벡터
            to_radar_x = radar_x_map - wall_center_x
            to_radar_y = radar_y_map - wall_center_y
            
            # 벽 법선 (수직, 건물에서 바깥쪽을 가리킴)
            wall_normal_x = -wall_dy  # 벽 벡터를 90도 회전
            wall_normal_y = wall_dx
            
            # 면 컬링: 레이더로부터 등을 돌린 벽들은 건너뛰기
            # 내적이 <= 0이면, 벽이 레이더로부터 등을 돌림
            if (wall_normal_x * to_radar_x + wall_normal_y * to_radar_y) <= 0:
                continue
            
            # 이 벽 구간에서 레이더에 가장 가까운 점 찾기
            wall_intersection = self.ClosestPointOnWallSegment(radar_x_map, radar_y_map, p1_world, p2_world)
            if wall_intersection is not None:
                dist = math.sqrt((wall_intersection[0] - radar_x_map)**2 + 
                               (wall_intersection[1] - radar_y_map)**2)
                if dist < closest_distance:
                    closest_distance = dist
                    closest_intersection = wall_intersection
        
        return closest_intersection

    # 벽면 최근접점 계산 - 레이더 위치에서 벽 선분까지의 최단거리 점 찾기
    def ClosestPointOnWallSegment(self, radar_x: float, radar_y: float,
                                 p1_world: Tuple[float, float], p2_world: Tuple[float, float]) -> Tuple[float, float]:
        """Find closest point on wall segment to radar position.
        
        Args:
            radar_x, radar_y: Radar position
            p1_world, p2_world: Wall segment endpoints
            
        Returns:
            Closest point on wall segment, or None if invalid
        """
        # p1에서 p2로의 벡터 (벽 방향)
        wall_dx = p2_world[0] - p1_world[0]
        wall_dy = p2_world[1] - p1_world[1]
        wall_length_sq = wall_dx**2 + wall_dy**2
        
        if wall_length_sq < 1e-6:  # 퇴화된 벽 (너무 짧음)
            return p1_world
        
        # p1에서 레이더로의 벡터
        to_radar_dx = radar_x - p1_world[0]
        to_radar_dy = radar_y - p1_world[1]
        
        # 레이더 위치를 벽 선 위에 투영
        t = (to_radar_dx * wall_dx + to_radar_dy * wall_dy) / wall_length_sq
        
        # t를 [0, 1]로 클램프하여 벽 구간에 머물게 함
        t = max(0.0, min(1.0, t))
        
        # 가장 가까운 점 계산
        closest_x = p1_world[0] + t * wall_dx
        closest_y = p1_world[1] + t * wall_dy
        
        return (closest_x, closest_y)

    # 교통표지판 처리 - 표지판 위치에서 레이더 포인트 생성
    def _process_traffic_signs(self, radar_points_gt: List[Tuple[float, float, float, float, float, float]],
                              radar_points_noise: List[Tuple[float, float, float, float, float, float]],
                              now: rospy.Time) -> None:
        """교통표지판 처리 로직"""
        if not self.signs:
            return
            
        for sign in self.signs.signs:
            # 맵 프레임에서 레이더 위치 가져오기
            radar_x_map = 0.0
            radar_y_map = 0.0
            if self.state is not None:
                ex = self.state.pose.position.x
                ey = self.state.pose.position.y
                qw = self.state.pose.orientation.w
                qz = self.state.pose.orientation.z
                ego_yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
                ryaw = math.radians(self.radar_yaw_deg)
                cos_ego = math.cos(ego_yaw)
                sin_ego = math.sin(ego_yaw)
                radar_x_map = ex + cos_ego * self.radar_tx - sin_ego * self.radar_ty
                radar_y_map = ey + sin_ego * self.radar_tx + cos_ego * self.radar_ty
            
            # 교통표지판 위치
            x_m = sign.pose.position.x
            y_m = sign.pose.position.y
            z = 0.0  # 2D 레이더 - 모든 포인트가 레이더 센서 높이에 있음
            
            # 레이더 프레임으로 변환
            x_radar, y_radar = self.MapToRadarXy(x_m, y_m)
            
            # FOV 필터
            if not self.InRadarFov(x_radar, y_radar):
                continue
            
            # 거리 필터
            range_gt = math.hypot(x_radar, y_radar)
            if range_gt > self.max_range:
                continue
            
            # 거리와 각도 해상도 적용
            range_quantized = self.QuantizeRange(range_gt)
            angle_original = math.atan2(y_radar, x_radar)
            angle_quantized = self.QuantizeAngle(angle_original)
            
            range_epsilon = 1e-6
            if abs(range_quantized - range_gt) > range_epsilon or abs(angle_quantized - angle_original) > range_epsilon:
                x_radar = range_quantized * math.cos(angle_quantized)
                y_radar = range_quantized * math.sin(angle_quantized)
            
            # 노이즈 적용
            x_radar_noisy, y_radar_noisy = self.ApplyNoise(x_radar, y_radar)
            
            # 교통표지판용 RCS 계산 및 레이더 방정식을 사용한 전력 계산
            rcs = self.object_props['traffic_sign']['base_rcs']
            power_dbm = self.CalculatePower(range_gt, rcs, 'traffic_sign')
            
            # 전력 임계값 적용
            if not self.ShouldDetect(power_dbm):
                continue
            
            # 정적 표지판 도플러 계산
            if self.state is not None:
                qw = self.state.pose.orientation.w
                qz = self.state.pose.orientation.z
                yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
                v_ego = self.state.twist.linear.x
                v_ex = v_ego * math.cos(yaw)
                v_ey = v_ego * math.sin(yaw)
                
                # ego 속도를 레이더 프레임으로 변환
                cos_ego_yaw = math.cos(-yaw)
                sin_ego_yaw = math.sin(-yaw)
                ego_vel_base_x = cos_ego_yaw * v_ex - sin_ego_yaw * v_ey
                ego_vel_base_y = sin_ego_yaw * v_ex + cos_ego_yaw * v_ey
                radar_yaw_rad = math.radians(self.radar_yaw_deg)
                cos_radar_yaw = math.cos(-radar_yaw_rad)
                sin_radar_yaw = math.sin(-radar_yaw_rad)
                ego_vel_radar_x = cos_radar_yaw * ego_vel_base_x - sin_radar_yaw * ego_vel_base_y
                ego_vel_radar_y = sin_radar_yaw * ego_vel_base_x + cos_radar_yaw * ego_vel_base_y
                
                # LOS 단위 벡터 계산
                min_range_for_los = 1e-6
                range_norm_noisy = max(min_range_for_los, math.hypot(x_radar_noisy, y_radar_noisy))
                normal_x_noisy, normal_y_noisy = x_radar_noisy / range_norm_noisy, y_radar_noisy / range_norm_noisy
                
                range_norm_gt = max(min_range_for_los, math.hypot(x_radar, y_radar))
                normal_x_gt, normal_y_gt = x_radar / range_norm_gt, y_radar / range_norm_gt
                
                # 정적 표지판의 도플러
                doppler_noisy_raw = (0.0 - ego_vel_radar_x) * normal_x_noisy + (0.0 - ego_vel_radar_y) * normal_y_noisy
                doppler_gt_raw = (0.0 - ego_vel_radar_x) * normal_x_gt + (0.0 - ego_vel_radar_y) * normal_y_gt
                
                # 속도 해상도 적용
                doppler_noisy = self.QuantizeVelocity(doppler_noisy_raw)
                doppler_gt = self.QuantizeVelocity(doppler_gt_raw)
            else:
                doppler_noisy = 0.0
                doppler_gt = 0.0
            
            # 포인트 클라우드에 추가
            radar_points_noise.append((x_radar_noisy, y_radar_noisy, z, power_dbm, doppler_noisy, rcs))
            radar_points_gt.append((x_radar, y_radar, z, power_dbm, doppler_gt, rcs))

    # 가드레일 처리 - 가드레일을 따라 일정 간격으로 레이더 포인트 생성
    def _process_guardrails(self, radar_points_gt: List[Tuple[float, float, float, float, float, float]],
                           radar_points_noise: List[Tuple[float, float, float, float, float, float]],
                           now: rospy.Time) -> None:
        """가드레일 처리 로직"""
        if not self.guardrails:
            return
            
        # 정적 목표물에 대한 도플러용 맵 프레임 ego 속도 벡터
        if self.state is not None:
            qw = self.state.pose.orientation.w
            qz = self.state.pose.orientation.z
            yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
            v_ego = self.state.twist.linear.x
            v_ex = v_ego * math.cos(yaw)
            v_ey = v_ego * math.sin(yaw)
            ex = self.state.pose.position.x
            ey = self.state.pose.position.y
        else:
            v_ex = v_ey = 0.0
            ex = ey = 0.0

        for obj in self.guardrails.objects:
            cx = obj.pose.position.x
            cy = obj.pose.position.y
            # 방향에서 yaw (평면 가정)
            qw = obj.pose.orientation.w
            qz = obj.pose.orientation.z
            gyaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
            min_guardrail_length = 0.1  # 최소 가드레일 구간 길이 (미터)
            half_len = 0.5 * max(min_guardrail_length, obj.size.x)
            min_step_size = 0.1  # 가드레일 샘플링 최소 스텝 크기 (미터)
            step = max(min_step_size, float(self.guardrail_spacing))
            s = -half_len
            while s <= half_len:
                xm = cx + s * math.cos(gyaw)
                ym = cy + s * math.sin(gyaw)
                z = 0.0  # 2D 레이더 - 모든 포인트가 레이더 센서 높이에 있음
                x_radar, y_radar = self.MapToRadarXy(xm, ym)
                if not self.InRadarFov(x_radar, y_radar):
                    s += step
                    continue

                # 거리 필터 - 최대 범위 내에 있는지 확인
                range_gt = math.hypot(x_radar, y_radar)
                if range_gt > self.max_range:
                    s += step
                    continue

                # 가드레일 포인트들에 노이즈 적용
                x_radar_noisy, y_radar_noisy = self.ApplyNoise(x_radar, y_radar)

                # 정적 객체에 대한 LOS와 도플러 (GT와 노이즈 모두)
                min_range_for_los = 1e-6  # LOS 계산에서 0으로 나누기 방지
                range_norm_noisy = max(min_range_for_los, math.hypot(x_radar_noisy, y_radar_noisy))
                normal_x_noisy, normal_y_noisy = x_radar_noisy / range_norm_noisy, y_radar_noisy / range_norm_noisy
                
                range_norm_gt = max(min_range_for_los, math.hypot(x_radar, y_radar))
                normal_x_gt, normal_y_gt = x_radar / range_norm_gt, y_radar / range_norm_gt
                # 레이더 프레임의 ego 속도
                ego_yaw = 0.0
                if self.state is not None:
                    qw = self.state.pose.orientation.w
                    qz = self.state.pose.orientation.z
                    ego_yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
                    v_ego = self.state.twist.linear.x
                    v_ex = v_ego * math.cos(ego_yaw)
                    v_ey = v_ego * math.sin(ego_yaw)
                cos_ego_yaw = math.cos(-ego_yaw)
                sin_ego_yaw = math.sin(-ego_yaw)
                ego_vel_base_x = cos_ego_yaw * v_ex - sin_ego_yaw * v_ey
                ego_vel_base_y = sin_ego_yaw * v_ex + cos_ego_yaw * v_ey
                radar_yaw_rad = math.radians(self.radar_yaw_deg)
                cos_radar_yaw = math.cos(-radar_yaw_rad)
                sin_radar_yaw = math.sin(-radar_yaw_rad)
                ego_vel_radar_x = cos_radar_yaw * ego_vel_base_x - sin_radar_yaw * ego_vel_base_y
                ego_vel_radar_y = sin_radar_yaw * ego_vel_base_x + cos_radar_yaw * ego_vel_base_y
                
                # GT와 노이즈 모두에 대해 도플러 계산
                doppler_noisy_raw = (0.0 - ego_vel_radar_x) * normal_x_noisy + (0.0 - ego_vel_radar_y) * normal_y_noisy
                doppler_gt_raw = (0.0 - ego_vel_radar_x) * normal_x_gt + (0.0 - ego_vel_radar_y) * normal_y_gt
                
                # 속도 해상도 양자화 적용
                doppler_noisy = self.QuantizeVelocity(doppler_noisy_raw)
                doppler_gt = self.QuantizeVelocity(doppler_gt_raw)

                # 가드레일용 RCS 계산 및 레이더 방정식을 사용한 전력 계산
                rcs = self.object_props['guardrail']['base_rcs']
                power_dbm = self.CalculatePower(range_gt, rcs, 'guardrail')
                
                # 전력 임계값 적용
                if not self.ShouldDetect(power_dbm):
                    s += step
                    continue
                
                # 두 포인트 클라우드에 모두 추가
                radar_points_noise.append((x_radar_noisy, y_radar_noisy, z, power_dbm, doppler_noisy, rcs))
                radar_points_gt.append((x_radar, y_radar, z, power_dbm, doppler_gt, rcs))

                # 시각화를 위해 감지도 선택적으로 추가 (희소하게)
                s += step
    
    # ===============================
    # 4. 공통 사용 함수들
    # ===============================

    # 바운딩박스 최근접점 계산 - 레이더에서 객체 바운딩박스 표면의 가장 가까운 점 찾기
    def GeoClosestBboxPoint(self, obj, radar_x: float, radar_y: float) -> Tuple[float, float]:
        """레이더에 가장 가까운 바운딩 박스 표면의 점을 가져옵니다.
        
        Args:
            obj: 포즈와 크기 정보를 가진 객체
            radar_x, radar_y: 맵 프레임에서의 레이더 위치
            
        Returns:
            맵 프레임에서 가장 가까운 표면 점의 (x, y) 좌표 튜플
        """
        # 객체 중심과 방향
        cx = obj.pose.position.x
        cy = obj.pose.position.y
        qw = obj.pose.orientation.w
        qz = obj.pose.orientation.z
        obj_yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
        
        # 객체 치수 (직사각형 바운딩 박스 가정)
        half_length = obj.size.x * 0.5  # 차량의 전진 방향
        half_width = obj.size.y * 0.5   # 차량의 측면 방향
        
        # 레이더 위치를 객체의 로컬 프레임으로 변환
        dx = radar_x - cx
        dy = radar_y - cy
        cos_yaw = math.cos(-obj_yaw)
        sin_yaw = math.sin(-obj_yaw)
        local_rx = cos_yaw * dx - sin_yaw * dy
        local_ry = sin_yaw * dx + cos_yaw * dy
        
        # 로컬 프레임에서 직사각형 바운딩 박스의 가장 가까운 점 찾기
        # 박스 경계에 클램프
        closest_local_x = max(-half_length, min(half_length, local_rx))
        closest_local_y = max(-half_width, min(half_width, local_ry))
        
        # 레이더가 박스 내부에 있으면 가장 가까운 표면으로 이동
        if abs(local_rx) <= half_length and abs(local_ry) <= half_width:
            # 각 가장자리까지의 거리
            dist_to_front = half_length - local_rx
            dist_to_back = half_length + local_rx
            dist_to_right = half_width - local_ry
            dist_to_left = half_width + local_ry
            
            min_dist = min(dist_to_front, dist_to_back, dist_to_right, dist_to_left)
            
            if min_dist == dist_to_front:
                closest_local_x = half_length
            elif min_dist == dist_to_back:
                closest_local_x = -half_length
            elif min_dist == dist_to_right:
                closest_local_y = half_width
            else:  # dist_to_left
                closest_local_y = -half_width
        
        # 맵 프레임으로 다시 변환
        cos_yaw_inv = math.cos(obj_yaw)
        sin_yaw_inv = math.sin(obj_yaw)
        closest_x = cx + cos_yaw_inv * closest_local_x - sin_yaw_inv * closest_local_y
        closest_y = cy + sin_yaw_inv * closest_local_x + cos_yaw_inv * closest_local_y
        
        return closest_x, closest_y

    # 좌표계 변환 - 맵 프레임 좌표를 레이더 프레임 좌표로 변환
    # TODO: 좌표 변환 - 맵 좌표를 레이더 좌표로 변환 
    def MapToRadarXy(self, x_map: float, y_map: float) -> Tuple[float, float]:
        """ego 포즈와 레이더 외부 파라미터를 사용하여 맵 프레임 XY를 레이더 프레임 XY로 변환."""
        #############################################################
        # TODO: 좌표 변환 - 맵 좌표를 레이더 좌표로 변환 실습
        # 실습: 전체 좌표 변환 파이프라인 구현
        if self.state is None:
            return x_map, y_map
        
        ego_x = self.state.pose.position.x
        ego_y = self.state.pose.position.y
        qw = self.state.pose.orientation.w
        qz = self.state.pose.orientation.z
        ego_yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
        delta_x = x_map - ego_x
        delta_y = y_map - ego_y
        # 맵 -> ego_frame
        cos_ego_yaw = math.cos(-ego_yaw)
        sin_ego_yaw = math.sin(-ego_yaw)
        
        # Hint: 회전행렬 계산 수식
        # [x_base]   [ cos(θ) -sin(θ)] [delta_x]
        # [y_base] = [ sin(θ)  cos(θ)] [delta_y]
        x_base = cos_ego_yaw * delta_x - sin_ego_yaw * delta_y # TODO: x_base 계산
        y_base = sin_ego_yaw * delta_x + cos_ego_yaw * delta_y # TODO: y_base 계산
        
        # ego_frame -> 레이더
        radar_yaw_rad = math.radians(self.radar_yaw_deg)
        cos_radar_yaw = math.cos(-radar_yaw_rad)
        sin_radar_yaw = math.sin(-radar_yaw_rad)
        
        # Hint: 회전행렬 계산 수식
        # [x_radar]   [ cos(φ) -sin(φ)] [x_base - tx]
        # [y_radar] = [ sin(φ)  cos(φ)] [y_base - ty]
        x_radar = cos_radar_yaw*(x_base - self.radar_tx) - sin_radar_yaw*(y_base - self.radar_ty) # TODO: x_radar 계산, Hint: self.radar_tx, self.radar_ty 사용
        y_radar = sin_radar_yaw*(x_base - self.radar_tx) + cos_radar_yaw*(y_base - self.radar_ty) # TODO: y_radar 계산, Hint: self.radar_tx, self.radar_ty 사용
        
        return x_radar, y_radar
        #############################################################

    # FOV 검사 - 주어진 좌표가 레이더 시야각(FOV) 내에 있는지 확인
    # TODO: FOV 검사 - 레이더 시야각 내에 있는지 확인 실습
    def InRadarFov(self, x_radar: float, y_radar: float) -> bool:
        angle_deg = math.degrees(math.atan2(y_radar, x_radar))
        
        # 실습: FOV 내에 있는지 확인 HOW?
        b_is_front = True if x_radar > 0 else False # TODO: 전방에 포인트가 있는지 검사
        b_is_fov = True if abs(angle_deg) < self.fov_deg/2 else False  # TODO: FOV 내에 포인트가 있는지 검사, Hint: self.fov_deg 사용
        return b_is_front and b_is_fov

    # 거리 양자화 - 레이더 거리 해상도에 따른 거리값 양자화
    def QuantizeRange(self, range_val: float) -> float:
        """거리 해상도 양자화를 적용합니다."""
        return round(range_val / self.range_resolution) * self.range_resolution
    
    # 속도 양자화 - 레이더 속도 해상도에 따른 도플러 속도 양자화
    def QuantizeVelocity(self, velocity_val: float) -> float:
        """속도 해상도 양자화를 적용합니다."""
        return round(velocity_val / self.velocity_resolution) * self.velocity_resolution
    
    # 각도 양자화 - 레이더 각도 해상도에 따른 방위각 양자화
    # TODO: 각도 양자화 - 레이더 각도 해상도에 따른 방위각 양자화 실습
    def QuantizeAngle(self, angle_rad: float) -> float:
        """각도 해상도 양자화를 적용합니다."""
        angle_deg = math.degrees(angle_rad)
        # 실습: 각도 해상도에 따른 양자화 구현
        # TODO: self.angle_resolution 사용
        quantized_deg = round(angle_deg / self.angle_resolution) * self.angle_resolution
        return math.radians(quantized_deg)
    
    # 전력 계산 - 레이더 방정식을 이용한 수신 전력 계산 (거리, RCS 기반)
    # TODO: 전력 계산 - 레이더 방정식을 이용한 수신 전력 계산 실습
    def CalculatePower(self, range_m: float, rcs: float, object_type: str) -> float:
        """레이더 방정식을 사용하여 수신 전력을 계산합니다.
        
        레이더 방정식: P_r = (P_t * G_t * G_r * λ² * σ) / ((4π)³ * R⁴ * L)
        
        Args:
            range_m: 타겟까지의 거리 (미터)
            rcs: 제곱미터 단위의 레이더 반사 단면적
            object_type: 반사율 조회용 객체 타입
            
        Returns:
            dBm 단위의 수신 전력
        """
        #############################################################
        # TODO: 전력 계산 - 레이더 방정식을 이용한 수신 전력 계산 실습
        # Hint: 1) 레이더 방정식 P_r = K * σ_eff / R^4
        # Hint: 2) 효과적 RCS = base_RCS * reflectivity
        # Hint: 3) 와트를 dBm으로 변환: P_dBm = 10*log10(P_watts*1000)
        
        # 실습: 0 나누기 방지 및 안전 거리 설정
        range_safe = max(range_m, 0.1)  # 실습: max(range_m, 0.1)로 최소값 보장
        
        # 객체 타입에 따른 반사율 계산
        if object_type in self.object_props:
            reflectivity = self.object_props[object_type]['reflectivity']
        else:
            reflectivity = 0.3  # 기본 반사율
        
        # 효과적 RCS 계산
        effective_rcs = rcs * reflectivity
        
        # 실습: 레이더 방정식으로 수신 전력 계산
        # 레이더 방정식 상수 (P_t * G_t * G_r * λ² * σ) / ((4π)³ * L)
        # TODO: 레이더 방정식 상수 완성, Hint: self.transmit_power_watts, self.transmit_gain_linear,
        #       self.receive_gain_linear, self.wavelength, self.system_losses_linear 사용
        num = self.transmit_power_watts * self.transmit_gain_linear * self.receive_gain_linear * (self.wavelength**2)
        den = pow(4*math.pi, 3) * self.system_losses_linear
        self.radar_constant = num / den
        received_power_watts = self.radar_constant * effective_rcs / pow(range_safe, 4)  # 실습: self.radar_constant * effective_rcs / (range_safe**4)
        
        # 실습: 와트를 dBm으로 변환
        if received_power_watts <= 0:
            return -200.0  # 매우 낮은 값 (사실상 감지 불가능)
        
        received_power_dbm = 10.0 * math.log10(received_power_watts * 1000)  # 실습: 10.0 * math.log10(received_power_watts * 1000)
        
        return received_power_dbm
        #############################################################
    
    # 감지 판정 - 전력 임계값과 확률을 기반으로 객체 감지 여부 결정
    # TODO: 감지 판정 - 전력 임계값과 확률 기반 감지 판정 실습
    def ShouldDetect(self, power_dbm: float) -> bool:
        """전력 임계값과 확률에 기반하여 객체가 감지되어야 하는지 결정합니다.
        
        Args:
            power_dbm: dBm 단위의 수신 전력
            
        Returns:
            객체가 감지되어야 하면 True, 그렇지 않으면 False
        """
        #############################################################
        # TODO: 감지 판정 - 전력 임계값과 확률 기반 감지 판정 실습
        # Hint: 1) power_dbm과 power_threshold_dbm 비교
        # Hint: 2) 감지 확률(detection_probability) 적용
        # Hint: 3) 랜덤 수와 확률 비교
        
        # 실습: 전력 임계값 검사
        # TODO: if 문을 통한 임계값 검사, Hint: self.power_threshold_dbm 사용
        if power_dbm < self.power_threshold_dbm:
            return False
        
        # 실습: 확률적 감지 구현
        # if self.detection_probability >= 1.0:
        #     return True
        
        # 랜덤 수와 감지 확률 비교
        return self.rng.random() < self.detection_probability
        #############################################################

    # ===============================
    # 5. 노이즈 처리 함수들
    # ===============================
    
    # 클러터 처리 - 허위 양성 레이더 감지 포인트 생성 
    def _process_clutter(self, radar_points_gt: List[Tuple[float, float, float, float, float, float]],
                        radar_points_noise: List[Tuple[float, float, float, float, float, float]],
                        now: rospy.Time) -> None:
        """클러터 처리 로직"""
        # 클러터 포인트 생성 (허위 양성)
        clutter_points = self.GenerateClutterPoints()
        for x_clutter, y_clutter, z_clutter, power_clutter_dbm, doppler_clutter, rcs_clutter in clutter_points:
            # 클러터를 노이즈 포인트 클라우드에만 추가 (GT에는 안함)
            radar_points_noise.append((x_clutter, y_clutter, z_clutter, power_clutter_dbm, doppler_clutter, rcs_clutter))

    # 노이즈 적용 - 레이더 측정값에 가우시안 노이즈 추가 (거리와 방위각)
    # TODO: 노이즈 적용 - 극좌표계에서 가우시안 노이즈 적용 실습
    def ApplyNoise(self, x_radar: float, y_radar: float) -> Tuple[float, float]:
        """레이더 측정값에 가우시안 노이즈를 적용합니다.
        
        Args:
            x_radar: 레이더 프레임의 x 좌표
            y_radar: 레이더 프레임의 y 좌표
            
        Returns:
            노이즈가 적용된 (x_radar, y_radar) 좌표 튜플
        """
        #############################################################
        # TODO: 노이즈 적용 - 극좌표계에서 가우시안 노이즈 적용 실습
        # Hint: 1) 직교좌표 -> 극좌표 변환 (range, azimuth)
        # Hint: 2) 각 성분에 가우시안 노이즈 적용
        # Hint: 3) 극좌표 -> 직교좌표 변환
        if not self.enable_noise:
            return x_radar, y_radar
            
        # 실습: 직교좌표를 극좌표로 변환
        range_true = math.hypot(x_radar, y_radar)  # 실습: math.hypot으로 거리 계산
        azimuth_true = math.atan2(y_radar, x_radar)  # 실습: math.atan2로 방위각 계산
        
        # 실습: 가우시안 노이즈 생성
        range_noise = self.rng.normal(0, self.range_noise_std)  # 실습: self.rng.normal로 거리 노이즈 생성, Hint: self.range_noise_std 사용
        azimuth_noise_rad = math.radians(self.rng.normal(0, self.azimuth_noise_std))  # 실습: 각도 노이즈 생성 및 라디안 변환, Hint: self.azimuth_noise_std 사용
        
        # 실습: 노이즈 적용
        range_noisy = max(range_true + range_noise, 0) # 실습: 거리 노이즈 적용 (양수 보장)
        azimuth_noisy = azimuth_true + azimuth_noise_rad  # 실습: 방위각 노이즈 적용
        
        # 실습: 극좌표를 다시 직교좌표로 변환
        x_radar_noisy = range_noisy * math.cos(azimuth_noisy)  # 실습: math.cos 사용
        y_radar_noisy = range_noisy * math.sin(azimuth_noisy)  # 실습: math.sin 사용
        
        return x_radar_noisy, y_radar_noisy
        #############################################################
    
    # 클러터 포인트 생성 - 허위 양성 레이더 감지를 위한 랜덤 클러터 포인트 생성
    # TODO: 클러터 생성 - 허위 양성 레이더 감지 생성 실습
    def GenerateClutterPoints(self) -> List[Tuple[float, float, float, float, float, float]]:
        """클러터(허위 양성) 레이더 감지를 생성합니다.
        
        Returns:
            클러터 포인트 목록: (x, y, z, intensity, doppler, rcs)
        """
        #############################################################
        # TODO: 클러터 생성 - 허위 양성 레이더 감지 생성 실습
        # Hint: 1) 클러터 발생 확률 검사
        # Hint: 2) FOV 내 랜덤 좌표 생성 (거리, 방위각)
        # Hint: 3) 극좌표 -> 직교좌표 변환
        # Hint: 4) RCS와 도플러 랜덤 값 생성
        
        clutter_points = []
        
        if not self.enable_clutter:
            return clutter_points
        
        # 실습: 클러터 발생 확률 검사
        # if self.rng.random() > self.clutter_probability:
        #     return clutter_points
        
        # 실습: 클러터 포인트 수 결정 (포아송 분포)
        num_clutter = max(1, int(self.rng.poisson(self.clutter_density)))  # 실습: max(1, int(self.rng.poisson(self.clutter_density)))
        
        for _ in range(num_clutter):
            # 실습: FOV 내 랜덤 좌표 생성
            clutter_range = self.rng.uniform(self.clutter_range_min,self.clutter_range_max)  # 실습: self.rng.uniform(range_min, range_max), Hint: self.clutter_range_min, self.clutter_range_max 사용
            
            # FOV 내에서 임의의 방위각 생성
            half_fov_rad = math.radians(self.fov_deg / 2.0)
            clutter_azimuth = self.rng.uniform(-half_fov_rad, half_fov_rad)  # 실습: self.rng.uniform(-half_fov, half_fov)
            
            # 실습: 극좌표를 직교좌표로 변환
            x_clutter = clutter_range * math.cos(clutter_azimuth)  # 실습: math.cos 사용
            y_clutter = clutter_range * math.sin(clutter_azimuth)   # 실습: math.sin 사용
            z_clutter = 0.0   # 지면 레벨
            
            # 실습: 클러터 RCS와 전력 계산
            rcs_clutter = self.rng.uniform(self.clutter_rcs_min, self.clutter_rcs_max)  # 실습: self.rng.uniform(rcs_min, rcs_max), Hint: self.clutter_rcs_min, self.clutter_rcs_max 사용
            clutter_range = math.sqrt(x_clutter**2 + y_clutter**2) # 실습: math.sqrt 을 사용해 거리 계산, Hint: x_clutter, y_clutter 사용
            power_clutter_dbm = self.CalculatePower(clutter_range, rcs_clutter, 'ground')  # 실습: CalculatePower 함수 호출, Hint: clutter_range, rcs_clutter, 'ground' 사용
            
            # 실습: 랜덤 도플러 생성
            doppler_noise_std = 1.0  # 도플러 노이즈의 표준편차
            doppler_clutter = self.rng.normal(0, doppler_noise_std)  # 실습: 가우시안 노이즈 사용, self.rng.normal
            
            clutter_points.append((x_clutter, y_clutter, z_clutter, power_clutter_dbm, doppler_clutter, rcs_clutter))
        
        return clutter_points
        #############################################################

    # ===============================
    # 6. 결과 publish 함수들
    # ===============================
    
    # 결과 퍼블리시 - 생성된 레이더 포인트들을 PointCloud2와 FOV 마커로 퍼블리시
    def _publish_results(self, radar_points_gt: List[Tuple[float, float, float, float, float, float]],
                        radar_points_noise: List[Tuple[float, float, float, float, float, float]],
                        now: rospy.Time) -> None:
        """결과 퍼블리시 로직"""
        # PointCloud2 퍼블리시 - GT와 노이즈
        if radar_points_gt:
            self.pub_pc.publish(self.ToPointcloud2(radar_points_gt, frame_id="radar", stamp=now))
        if radar_points_noise:
            self.pub_pc_noise.publish(self.ToPointcloud2(radar_points_noise, frame_id="radar", stamp=now))

        # FOV 시각화 퍼블리시
        self.pub_fov.publish(self.CreateFovMarker(now))

    # PointCloud2 변환 - 레이더 포인트 리스트를 ROS PointCloud2 메시지로 변환
    def ToPointcloud2(self, pts: List[Tuple[float, float, float, float, float, float]], frame_id: str, stamp: rospy.Time) -> PointCloud2:
        header = Header(stamp=stamp, frame_id=frame_id)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('power', 12, PointField.FLOAT32, 1),
            PointField('doppler', 16, PointField.FLOAT32, 1),
            PointField('rcs', 20, PointField.FLOAT32, 1),
        ]
        point_step = 24
        data = b''.join([struct.pack('ffffff', *p) for p in pts])
        pc2 = PointCloud2(
            header=header,
            height=1,
            width=len(pts),
            fields=fields,
            is_bigendian=False,
            point_step=point_step,
            row_step=point_step * len(pts),
            is_dense=True,
            data=data,
        )
        return pc2

    # FOV 마커 생성 - 레이더 시야각 시각화를 위한 마커 생성
    def CreateFovMarker(self, stamp: rospy.Time) -> MarkerArray:
        """투명도가 있는 FOV 시각화 마커를 생성합니다."""
        marker_array = MarkerArray()
        
        # FOV를 위한 섹터(부채꼴 모양) 마커 생성
        marker = Marker()
        marker.header.frame_id = "radar"
        marker.header.stamp = stamp
        marker.ns = "radar_fov"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        
        # 마커 스케일 설정
        marker.scale.x = 1.0
        marker.scale.y = 1.0 
        marker.scale.z = 1.0
        
        # 투명도가 있는 색상 설정 (alpha = 0.2로 높은 투명도)
        marker.color = ColorRGBA()
        marker.color.r = 0.0
        marker.color.g = 1.0  # 녹색
        marker.color.b = 0.0
        marker.color.a = 0.2  # 매우 투명
        
        # FOV 섹터 포인트 생성
        half_fov_rad = math.radians(self.fov_deg / 2.0)
        fov_visualization_segments = 20  # 부드러운 섹터 시각화를 위한 구간 수
        
        # 중심점 (레이더 원점)
        center = Point()
        center.x = 0.0
        center.y = 0.0
        center.z = 0.0
        
        # 섹터를 형성하기 위한 삼각형 구간들 생성
        for i in range(fov_visualization_segments):
            # 현재와 다음 각도
            angle1 = -half_fov_rad + (i * 2 * half_fov_rad) / fov_visualization_segments
            angle2 = -half_fov_rad + ((i + 1) * 2 * half_fov_rad) / fov_visualization_segments
            
            # 호 위의 포인트들
            point1 = Point()
            point1.x = self.max_range * math.cos(angle1)
            point1.y = self.max_range * math.sin(angle1)
            point1.z = 0.0
            
            point2 = Point()
            point2.x = self.max_range * math.cos(angle2)
            point2.y = self.max_range * math.sin(angle2)
            point2.z = 0.0
            
            # 삼각형 생성 (중심 -> point1 -> point2)
            marker.points.append(center)
            marker.points.append(point1)
            marker.points.append(point2)
            
            # 각 정점에 색상 추가 (같은 색상)
            vertices_per_triangle = 3
            for _ in range(vertices_per_triangle):
                marker.colors.append(marker.color)
        
        marker_array.markers.append(marker)
        
        # 선택사항: 더 나은 시각성을 위해 가장자리 선 추가
        edge_marker = Marker()
        edge_marker.header.frame_id = "radar"
        edge_marker.header.stamp = stamp
        edge_marker.ns = "radar_fov_edges"
        edge_marker.id = 1
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        
        fov_edge_line_width = 0.02  # FOV 가장자리 시각화용 선 두께
        edge_marker.scale.x = fov_edge_line_width
        edge_marker.color = ColorRGBA()
        edge_marker.color.r = 0.0
        edge_marker.color.g = 0.8
        edge_marker.color.b = 0.0
        edge_marker.color.a = 0.6  # 반투명 가장자리
        
        # 호 가장자리 선들 추가
        for i in range(fov_visualization_segments + 1):
            angle = -half_fov_rad + (i * 2 * half_fov_rad) / fov_visualization_segments
            point = Point()
            point.x = self.max_range * math.cos(angle)
            point.y = self.max_range * math.sin(angle)
            point.z = 0.0
            
            # 중심에서 호 포인트까지의 선 (방사형 선, 가장자리에서만)
            if i == 0 or i == fov_visualization_segments:
                edge_marker.points.append(center)
                edge_marker.points.append(point)
            
            # 호 선들 (호 위 인접 포인트들 연결)
            if i > 0:
                prev_angle = -half_fov_rad + ((i - 1) * 2 * half_fov_rad) / fov_visualization_segments
                prev_point = Point()
                prev_point.x = self.max_range * math.cos(prev_angle)
                prev_point.y = self.max_range * math.sin(prev_angle)
                prev_point.z = 0.0
                
                edge_marker.points.append(prev_point)
                edge_marker.points.append(point)
        
        marker_array.markers.append(edge_marker)
        
        return marker_array

def main():

    rospy.init_node("sensor_radar_node")
    radar = RadarEmu()
    if hasattr(radar, 'enabled') and radar.enabled:
        rospy.loginfo("sensor_radar_node started")
    else:
        rospy.loginfo("sensor_radar_node disabled")
    rospy.spin()


if __name__ == "__main__":
    main()


