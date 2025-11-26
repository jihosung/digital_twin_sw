#!/usr/bin/env python
"""
GNSS Sensor Simulation Node

이 노드는 차량에 장착된 GNSS(Global Navigation Satellite System) 센서를 시뮬레이션합니다.
차량의 현재 상태를 기반으로 위성 항법 데이터(위도, 경도, 고도)를 생성합니다.

교육 목적:
1. GNSS 센서의 동작 원리와 측정 데이터 이해
2. WGS84 좌표계와 ENU 좌표계 간의 변환 학습
3. 지구 곡률과 측지학적 계산 방법 이해
4. GNSS 음영 지역과 신호 품질 저하 현상 학습
5. 센서 융합에서 GNSS 데이터의 활용 방법 학습

입력 토픽들:
- /vehicle/state: 차량의 현재 상태 (위치, 속도, 자세 등)

출력 토픽들:
- /sensors/gnss/fix: GNSS 위치 데이터 (NavSatFix)
- /sensors/gnss/odom: GNSS 기반 오도메트리 (Odometry)
- /sensors/gnss/shadow_zones: GNSS 음영 지역 시각화 (MarkerArray)

센서 사양:
- 위치 정확도: 가우시안 노이즈 적용 (일반적으로 ±0.5m)
- 고도 정확도: 수평 위치보다 낮은 정확도 (일반적으로 ±1.0m)
- 음영 지역: 건물, 터널 등에서 정확도 저하 시뮬레이션
- 좌표 변환: ENU ↔ WGS84 측지학적 변환
"""
import rospy
from sensor_msgs.msg import NavSatFix, NavSatStatus
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance
from dcas_msgs.msg import VehicleState
import time
import numpy as np
import math
import os
import csv
try:
    import rospkg
except Exception:
    rospkg = None  # Optional; path resolution fallback


class GNSSEmu:
    """GNSS 센서 시뮬레이터
    
    실제 차량의 GNSS 센서를 시뮬레이션합니다.
    
    주요 기능:
    1. ENU 좌표계를 WGS84 좌표계로 변환
    2. 측지학적 계산 (지구 곡률반지름 등)
    3. GNSS 측정 노이즈 및 음영 지역 시뮬레이션
    4. NavSatFix와 Odometry 메시지 생성
    
    센서 모델링:
    - 위치 = 실제값 + 가우시안 노이즈 + 음영지역 오차
    - 좌표 변환 = 측지학적 공식 기반 정확한 계산
    """

    def __init__(self) -> None:
        """GNSS 센서 시뮬레이터 초기화
        
        ROS 매개변수를 통해 센서 설정을 로드하고,
        퍼블리셔와 구독자를 설정합니다.
        """
        # =================================================================
        # 1. ROS 노드 설정 및 센서 활성화 확인
        # =================================================================
        
        # 센서 활성화 상태 확인
        self.enabled = rospy.get_param("/sensor_gnss_node/enabled", True)
        if not self.enabled:
            rospy.loginfo("GNSS sensor is disabled")
            return
        
        # =================================================================
        # 2. 퍼블리셔 설정 (센서 출력)
        # =================================================================
        
        # GNSS 위치 데이터 퍼블리셔 (NavSatFix)
        self.pub_fix = rospy.Publisher("/sensors/gnss/fix", NavSatFix, queue_size=10)
        
        # GNSS 오도메트리 퍼블리셔 (Odometry)
        self.pub_odom = rospy.Publisher("/sensors/gnss/odom", Odometry, queue_size=10)

        # =================================================================
        # 3. 센서 파라미터 설정 (Configuration에서 읽기)
        # =================================================================
        
        # 센서 업데이트 주기 (초)
        period = float(rospy.get_param("/sensor_gnss_node/period_s", 0.1))

        # GNSS 기준점 설정 (WGS84 좌표계)
        self.lat0 = float(rospy.get_param("/sensor_gnss_node/lat0", 37.0))    # 기준 위도 (도)
        self.lon0 = float(rospy.get_param("/sensor_gnss_node/lon0", 127.0))   # 기준 경도 (도)
        self.alt0 = float(rospy.get_param("/sensor_gnss_node/alt0", 50.0))    # 기준 고도 (미터)
        
        # =================================================================
        # 4. WGS84 지구 타원체 상수 설정
        # =================================================================
        
        # WGS84 타원체 매개변수 (측지학적 계산용)
        self.WGS84_A = 6378137.0        # 장반축 (미터)
        self.WGS84_B = 6356752.314245   # 단반축 (미터)

        # =================================================================
        # 5. 구독자 설정 (차량 상태 정보 수신)
        # =================================================================
        
        # 차량 상태 정보 저장
        self.state = None  # type: ignore
        
        # 차량 상태 구독자
        rospy.Subscriber("/vehicle/state", VehicleState, self.CallbackState)

        # =================================================================
        # 6. 노이즈 모델 파라미터 설정
        # =================================================================
        
        # 위치 노이즈 표준편차 (미터) - 수평 위치 오차
        self.pos_noise_std = rospy.get_param("/sensor_gnss_node/pos_noise_std", 0.5)
        
        # 속도 노이즈 표준편차 (m/s) - 도플러 효과 기반 속도 측정 오차
        self.vel_noise_std = rospy.get_param("/sensor_gnss_node/vel_noise_std", 0.05)
        
        # 각속도 노이즈 표준편차 (rad/s)
        self.ang_noise_std = rospy.get_param("/sensor_gnss_node/ang_noise_std", 0.002)
        
        # 자세 노이즈 표준편차 (rad) - GNSS는 자세 측정 정확도가 상대적으로 낮음
        self.ori_noise_std = rospy.get_param("/sensor_gnss_node/ori_noise_std", 0.01)
        
        # 고도 노이즈 표준편차 (미터) - 수직 위치는 수평보다 정확도가 낮음
        self.alt_noise_std = rospy.get_param("/sensor_gnss_node/alt_noise_std", 1.0)
        
        # 노이즈 적용 활성화 여부
        self.enable_noise = rospy.get_param("/sensor_gnss_node/enable_noise", True)

        # =================================================================
        # 7. GNSS 음영 지역 파라미터 설정
        # =================================================================
        
        # 음영 지역 시뮬레이션 활성화 여부
        self.enable_shadow_zones = rospy.get_param("/sensor_gnss_node/enable_shadow_zones", False)
        
        # 음영 지역 맵 CSV 파일 경로
        self.shadow_map_csv = rospy.get_param("/sensor_gnss_node/shadow_map_csv", "")
        
        # 음영 지역 데이터 저장 (x, y, pos_std, radius)
        self.shadow_zones = []
        if self.enable_shadow_zones and self.shadow_map_csv:
            csv_path = self._resolve_shadow_csv_path(self.shadow_map_csv)
            self.shadow_zones = self._load_shadow_zones(csv_path)

        # =================================================================
        # 8. 랜덤 시드 및 시각화 설정
        # =================================================================
        
        # 노이즈 생성을 위한 랜덤 상태 초기화
        self.rng = np.random.RandomState(int(time.time() * 1000) % 2**32)

        # 음영 지역 시각화 설정
        self.enable_shadow_viz = rospy.get_param("/sensor_gnss_node/enable_shadow_viz", False)
        if self.enable_shadow_viz:
            self.pub_shadow = rospy.Publisher("/sensors/gnss/shadow_zones", MarkerArray, queue_size=1, latch=True)
            self._publish_shadow_markers()
            
        # =================================================================
        # 9. 타이머 설정
        # =================================================================
        
        # 타이머 (센서가 활성화된 경우에만 생성)
        if self.enabled:
            self.timer = rospy.Timer(rospy.Duration(period), self.CallbackTimer)


    # =====================================================================
    # ROS 콜백 함수들
    # =====================================================================

    def CallbackState(self, msg: VehicleState) -> None:
        """차량 상태 정보 수신 콜백 함수"""
        self.state = msg

    def CallbackTimer(self, _evt) -> None:
        """주기적으로 호출되는 GNSS 데이터 생성 및 발행 함수
        
        이 함수는 타이머에 의해 주기적으로 호출되며,
        차량 상태에 따라 GNSS 센서 데이터를 생성합니다.
        
        처리 순서:
        1. 차량 상태 확인
        2. ENU 좌표를 WGS84 좌표로 변환
        3. NavSatFix 및 Odometry 메시지 생성
        4. 음영 지역 효과 적용
        5. 노이즈 적용 및 공분산 설정
        6. GNSS 데이터 발행
        """
        if not self.enabled:
            return
        
        # =================================================================
        # 1. 차량 상태 확인
        # =================================================================
        
        # 마지막 상태 사용 또는 스킵
        if self.state is None:
            return

        # 차량의 현재 포즈와 속도 정보 가져오기
        pose = self.state.pose
        twist = self.state.twist

        # =================================================================
        # 2. ENU 좌표계에서 WGS84 좌표계로 변환
        # =================================================================

        # ENU 좌표 추출 (동쪽, 북쪽, 위쪽)
        e = pose.position.x  # East (동쪽)
        n = pose.position.y  # North (북쪽)  
        u = pose.position.z  # Up (위쪽)

        #############################################################
        # TODO: 좌표 변환 실습 - ENU → WGS84
        # ENU 좌표를 배열로 구성
        enu_coords = np.array([e, n, u])
        
        # 기준점 WGS84 좌표 (위도, 경도, 고도)
        ref_llh = np.array([self.lat0, self.lon0, self.alt0])
        
        # ENU to WGS84 변환 함수 호출
        llh_result = self.enu_to_wgs84(enu_coords, ref_llh) # TODO: ENU to WGS84 변환 함수 완성
        
        # 변환된 WGS84 좌표 추출
        lat = llh_result[0]  # 위도 (도)
        lon = llh_result[1]  # 경도 (도)
        alt = llh_result[2]  # 고도 (미터)
        #############################################################

        # =================================================================
        # 3. NavSatFix 메시지 생성
        # =================================================================

        # GNSS 위치 데이터 메시지 생성
        fix = NavSatFix()
        fix.header.stamp = rospy.Time.now()
        fix.header.frame_id = "gps"
        
        # 유효한 GPS 신호로 표시 (하위 소비자가 공분산을 사용하도록)
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude = lat
        fix.longitude = lon
        fix.altitude = alt

        # =================================================================
        # 4. Odometry 메시지 생성
        # =================================================================

        # GNSS 기반 오도메트리 메시지 생성
        od = Odometry()
        od.header = fix.header
        od.header.frame_id = "map"
        od.child_frame_id = "ego_frame"
        od.pose = PoseWithCovariance()
        od.pose.pose = pose
        od.twist = TwistWithCovariance()
        od.twist.twist = twist
        
        # =================================================================
        # 5. 음영 지역 효과 적용
        # =================================================================
        
        # 노이즈 적용 전에 음영 지역 표준편차 조회 (추가 음영 노이즈를 위치에 적용)
        e = pose.position.x
        n = pose.position.y
        # TODO: GNSS Shadow Zone 반영 함수 완성
        shadow_pos_std = self._query_shadow_pos_std(e, n) if self.enable_shadow_zones else 0.0 
        
        # =================================================================
        # 6. 노이즈 적용 및 공분산 설정
        # =================================================================
        
        # 노이즈 적용
        if self.enable_noise == True:
            self._apply_noise(fix, od, shadow_pos_std) # TODO: GNSS 노이즈 적용 함수 완성

        # 음영 지역 옵션과 함께 공분산 설정
        self._set_covariances(fix, od, shadow_pos_std)

        # =================================================================
        # 7. GNSS 데이터 발행
        # =================================================================

        self.pub_fix.publish(fix)
        self.pub_odom.publish(od)
        
    # =====================================================================
    # 측지학적 계산 함수들 (교육용 핵심 부분)
    # =====================================================================
        
    def meridional_radius(self, ref_latitude_deg: float) -> float:
        """자오선 곡률반지름을 계산합니다.
        
        교육 목적:
        지구 타원체의 기하학적 특성을 이해하고 측지학적 계산의 기초를 학습합니다:
        1. 지구 타원체의 수학적 모델링
        2. 곡률반지름의 물리적 의미와 계산 방법
        3. 위도에 따른 지구 모양의 변화 이해
        4. GPS 정확도에 영향을 미치는 기하학적 요인
        
        자오선 곡률반지름 (M):
        - 남북 방향의 곡률반지름
        - 위도가 높을수록 (극지방) 작아짐
        - 공식: M = a(1-e²) / (1-e²sin²φ)^(3/2)
        
        Args:
            ref_latitude_deg: 기준점 위도 (도 단위)
            
        Returns:
            float: 자오선 곡률반지름 (미터)
        """
        lat = np.deg2rad(ref_latitude_deg)
        
        a = self.WGS84_A # 장반축 6378137.0m
        b = self.WGS84_B # 단반축 6356752.314245m
        
        #############################################################
        # TODO: 자오선 곡률반지름 계산식 작성
        # 자오선 곡률반지름: M = (ab)^2 / ((acosφ)^2+(bsinφ)^2)^(3/2)
        # 여기서 φ는 위도, a는 장반축, b는 단반축
        num = (a*b)**2
        den = math.pow((a*math.cos(lat))**2 + (b*math.sin(lat))**2, 3/2)

        radius = num / den
        #############################################################
        
        return radius
    
    def normal_radius(self, ref_latitude_deg: float) -> float:
        """수직 곡률반지름(법선 곡률반지름)을 계산합니다.
        
        교육 목적:
        측지학에서 중요한 또 다른 곡률반지름 개념을 학습합니다:
        1. 동서 방향의 곡률 특성 이해
        2. 자오선 곡률반지름과의 차이점 학습
        3. 타원체 기하학의 완전한 이해
        4. 좌표 변환에서 정확한 스케일 팩터 계산
        
        수직 곡률반지름 (N):
        - 동서 방향의 곡률반지름  
        - 적도에서 최소, 극지방에서 최대
        - 공식: N = a / √(1-e²sin²φ)
        
        Args:
            ref_latitude_deg: 기준점 위도 (도 단위)
            
        Returns:
            float: 수직 곡률반지름 (미터)
        """
        lat = np.deg2rad(ref_latitude_deg)
        
        a = self.WGS84_A # 장반축 6378137.0m
        b = self.WGS84_B # 단반축 6356752.314245m

        #############################################################
        # TODO: 수직 곡률반지름 계산식 작성
        # 수직 곡률반지름: N = a^2 / ((acosφ)^2 + (bsinφ)^2)^(1/2)
        # 여기서 φ는 위도, a는 장반축, b는 단반축
        num = a**2
        den = math.pow((a*math.cos(lat))**2 + (b*math.sin(lat))**2, 1/2)
        
        normal_radius = num / den
        #############################################################
        
        return normal_radius
    
    def enu_to_wgs84(self, enu: np.ndarray, ref_llh: np.ndarray) -> np.ndarray:
        """ENU 좌표계를 WGS84 좌표계로 변환합니다.
        
        교육 목적:
        이 함수는 로봇공학과 측지학에서 핵심적인 좌표 변환을 다룹니다:
        1. 지역 좌표계(ENU)와 전역 좌표계(WGS84)의 관계 이해
        2. 측지학적 변환의 수학적 원리 학습
        3. GPS/GNSS 데이터 처리의 실제 과정 체험
        4. 지구 곡률을 고려한 정확한 변환 방법 학습
        
        좌표계 설명:
        - ENU: East-North-Up (동-북-상) 지역 직교 좌표계
        - WGS84: World Geodetic System 1984 (위도-경도-고도) 전역 좌표계
        
        변환 과정:
        1. 기준점에서의 지구 곡률반지름 계산
        2. ENU 변위를 각도 변화량으로 변환
        3. 기준점 좌표에 각도 변화량 더하기
        
        Args:
            enu (np.ndarray): 변환할 ENU 좌표 [동쪽, 북쪽, 위쪽] (3)
            ref_llh (np.ndarray): 기준점 좌표 [위도, 경도, 고도]
            
        Returns:
            np.ndarray: WGS84 좌표 [위도, 경도, 고도] (3)
        """
        llh = np.zeros_like(enu)

        #############################################################
        # TODO: 좌표 변환 실습 - ENU → WGS84
        # 기준점 정보 추출
        ref_height = ref_llh[2]
        ref_lat = np.deg2rad(ref_llh[0])
        
        # 곡률반지름 계산 (측지학적 정확도를 위해 필요)
        meridional_r = self.meridional_radius(ref_llh[0]) # TODO: 자오선 곡률반지름 계산 함수 완성
        normal_r = self.normal_radius(ref_llh[0]) # TODO: 수직 곡률반지름 계산 함수 완성

        # 각도 변화량 계산
        # 북쪽 변위 → 위도 변화: Δφ = ΔN / M
        # 동쪽 변위 → 경도 변화: Δλ = ΔE / (N × cos(φ))
        delta_lat_deg = np.rad2deg(enu[1] / (meridional_r + ref_height)) # TODO: 위도 변화량 계산, np.rad2deg 사용
        delta_lon_deg = np.rad2deg(enu[0] / ((normal_r + ref_height) * np.cos(ref_lat))) # TODO: 경도 변화량 계산, np.rad2deg 사용
        #############################################################
        
        # TODO: 최종 WGS84 좌표 계산
        llh[0] = ref_llh[0] + delta_lat_deg # Latitude
        llh[1] = ref_llh[1] + delta_lon_deg # Longitude
        llh[2] = ref_llh[2] + enu[2] # Height
            
        return llh

    def _apply_noise(self, msg: NavSatFix, msg_odom: Odometry, shadow_pos_std: float = 0.0) -> None:
        """GNSS 메시지에 노이즈를 적용합니다.
        
        교육 목적:
        실제 GNSS 센서의 측정 불확실성을 시뮬레이션하여 다음을 학습합니다:
        1. 센서 노이즈의 특성과 분포 이해
        2. 위치/속도/각속도 각각의 노이즈 모델링
        3. 음영 지역에서의 추가 오차 반영
        4. 실제 환경에서의 센서 신뢰도 평가
        
        노이즈 적용 방식:
        - 위치: 가우시안 노이즈 + 음영 지역 추가 노이즈
        - 속도: 선형/각속도에 독립적 가우시안 노이즈
        - 좌표 변환: 미터 단위를 도 단위로 변환
        
        Args:
            msg: NavSatFix
            msg: Odometry
            shadow_pos_std: Additional position std (m) to add for x/y when inside a shadow zone
            
        Returns:
            NavSatFix, Odometry data
        """
        lat_rad = math.radians(msg.latitude if msg.latitude else self.lat0)
        m2deg_lat = 1.0 / 111320.0
        m2deg_lon = 1.0 / (111320.0 * max(1e-6, math.cos(lat_rad)))

        # TODO: GNSS 노이즈 적용
        n_lat_m = self.rng.normal(0.0, self.pos_noise_std)   # [m], self.rng.normal() 사용
        n_lon_m = self.rng.normal(0.0, self.pos_noise_std)   # [m], self.rng.normal() 사용
        n_alt_m = self.rng.normal(0.0, self.alt_noise_std)   # [m], self.rng.normal() 사용

        # TODO: 노이즈를 도 단위로 변환하여 적용
        msg.latitude  += (n_lat_m * m2deg_lat)
        msg.longitude += (n_lon_m * m2deg_lon)
        msg.altitude  += n_alt_m

        # TODO: 음영 지역에서 추가 위치 노이즈 적용
        if shadow_pos_std > 0.0:
            sh_lat_m = self.rng.normal(0.0, shadow_pos_std)
            sh_lon_m = self.rng.normal(0.0, shadow_pos_std)
            # TODO: 노이즈를 도 단위로 변환하여 적용
            msg.latitude += (sh_lat_m * m2deg_lat)
            msg.longitude += (sh_lon_m * m2deg_lon)

        # TODO: Odometry 노이즈 적용
        msg_odom.pose.pose.position.x += self.rng.normal(0.0, self.pos_noise_std) # self.rng.normal() 사용
        msg_odom.pose.pose.position.y += self.rng.normal(0.0, self.pos_noise_std) # self.rng.normal() 사용
        msg_odom.pose.pose.position.z += self.rng.normal(0.0, self.pos_noise_std) # self.rng.normal() 사용

        # TODO: 음영 지역에서 추가 위치 노이즈 적용
        if shadow_pos_std > 0.0:
            msg_odom.pose.pose.position.x += self.rng.normal(0.0, shadow_pos_std)
            msg_odom.pose.pose.position.y += self.rng.normal(0.0, shadow_pos_std)

        # TODO: twist 노이즈 적용
        msg_odom.twist.twist.linear.x  += self.rng.normal(0.0, self.vel_noise_std) # self.rng.normal() 사용
        msg_odom.twist.twist.linear.y  += self.rng.normal(0.0, self.vel_noise_std) # self.rng.normal() 사용
        msg_odom.twist.twist.linear.z  += self.rng.normal(0.0, self.vel_noise_std) # self.rng.normal() 사용

        msg_odom.twist.twist.angular.x += self.rng.normal(0.0, self.ang_noise_std) # self.rng.normal() 사용
        msg_odom.twist.twist.angular.y += self.rng.normal(0.0, self.ang_noise_std) # self.rng.normal() 사용용
        msg_odom.twist.twist.angular.z += self.rng.normal(0.0, self.ang_noise_std) # self.rng.normal() 사용용

        return msg, msg_odom

    def _set_covariances(self, fix: NavSatFix, od: Odometry, shadow_pos_std: float) -> None:
        """GNSS 출력에 대한 공분산 행렬을 설정합니다.
        
        교육 목적:
        센서 데이터의 불확실성을 정량적으로 표현하는 방법을 학습합니다:
        1. 공분산 행렬의 의미와 구조 이해
        2. 센서 신뢰도의 수치적 표현 방법
        3. 음영 지역에서의 불확실성 증가 모델링
        4. 칼만 필터 등 추정 알고리즘의 입력 데이터 생성
        
        공분산 행렬 구조:
        - 대각선 요소: 각 변수의 분산 (불확실성의 크기)
        - 비대각선 요소: 변수 간 상관관계 (보통 0으로 설정)
        
        Args:
            fix: NavSatFix message to set covariance on.
            od: Odometry message to set covariance on.
            shadow_pos_std: Additional position std dev [m] to add in quadrature for x/y.
        """
        # 음영 지역 효과를 고려한 위치 분산 계산
        pos_var = self.pos_noise_std ** 2  # 기본 위치 분산
        if shadow_pos_std > 0.0:
            pos_var += shadow_pos_std ** 2  # 음영 지역 추가 분산
        
        # NavSatFix 위치 공분산 설정 (3x3 행렬: 위도, 경도, 고도)
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        fix.position_covariance = [0] * 9  # 3x3 행렬을 1차원 배열로 표현
        
        # 미터를 도 단위로 변환
        m2deg_lat = 1.0 / 111320.0
        m2deg_lon = 1.0 / (111320.0 * math.cos(math.radians(fix.latitude)))
        
        # 대각선 요소에 분산 설정 (위도, 경도, 고도)
        fix.position_covariance[0] = pos_var * (m2deg_lat ** 2)  # 위도 분산
        fix.position_covariance[4] = pos_var * (m2deg_lon ** 2)  # 경도 분산  
        fix.position_covariance[8] = pos_var                      # 고도 분산

        # Odometry 위치 공분산 설정 (6x6 행렬: x, y, z, roll, pitch, yaw)
        od.pose.covariance = [0] * 36  # 6x6 행렬을 1차원 배열로 표현
        
        # 위치 분산 설정 (인덱스 0, 7, 14: x, y, z 위치)
        od.pose.covariance[0]  = pos_var  # x 위치 분산
        od.pose.covariance[7]  = pos_var  # y 위치 분산
        od.pose.covariance[14] = pos_var  # z 위치 분산
        
        # 자세 분산 설정 (인덱스 21, 28, 35: roll, pitch, yaw)
        od.pose.covariance[21] = self.ori_noise_std ** 2  # roll 분산
        od.pose.covariance[28] = self.ori_noise_std ** 2  # pitch 분산
        od.pose.covariance[35] = self.ori_noise_std ** 2  # yaw 분산

        # Odometry 속도 공분산 설정 (6x6 행렬: vx, vy, vz, wx, wy, wz)
        od.twist.covariance = [0] * 36
        
        # 선형 속도 분산 설정
        od.twist.covariance[0]  = self.vel_noise_std ** 2  # vx 분산
        od.twist.covariance[7]  = self.vel_noise_std ** 2  # vy 분산
        od.twist.covariance[14] = self.vel_noise_std ** 2  # vz 분산
        
        # 각속도 분산 설정
        od.twist.covariance[21] = self.ang_noise_std ** 2  # wx 분산
        od.twist.covariance[28] = self.ang_noise_std ** 2  # wy 분산
        od.twist.covariance[35] = self.ang_noise_std ** 2  # wz 분산

    def _resolve_shadow_csv_path(self, path_param: str) -> str:
        """CSV 파일 경로를 해석합니다. 절대 경로나 maps/ 디렉토리 상대 경로 지원.
        
        교육 목적:
        파일 시스템과 ROS 패키지 구조를 이해하고 경로 처리 방법을 학습합니다:
        1. 절대 경로와 상대 경로의 차이점 이해
        2. ROS 패키지 내 리소스 파일 접근 방법
        3. 파일 시스템 탐색과 예외 처리
        4. 크로스 플랫폼 호환성을 고려한 경로 처리
        
        경로 해석 우선순위:
        1. 절대 경로인 경우 그대로 사용
        2. 패키지 maps/ 디렉토리 내 상대 경로로 시도
        3. 현재 작업 디렉토리 기준 절대 경로로 변환
        
        Args:
            path_param: 설정에서 읽은 파일 경로 문자열
            
        Returns:
            str: 해석된 절대 파일 경로
        """
        if os.path.isabs(path_param):
            return path_param
        # ROS 패키지 maps 디렉토리를 기준으로 상대 경로 해석 시도
        if rospkg is not None:
            try:
                pkg_path = rospkg.RosPack().get_path('dcas_perception_sim')
                candidate = os.path.join(pkg_path, 'maps', path_param)
                if os.path.exists(candidate):
                    return candidate
            except Exception:
                pass
        # 현재 작업 디렉토리 기준으로 폴백
        return os.path.abspath(path_param)

    def _load_shadow_zones(self, csv_path: str):
        """CSV 파일에서 음영 지역을 로드합니다.
        
        교육 목적:
        데이터 파일 처리와 GNSS 음영 지역 개념을 학습합니다:
        1. CSV 파일 형식과 데이터 파싱 방법 이해
        2. GNSS 신호 차단 환경과 그 영향 학습
        3. 실제 환경 조건을 시뮬레이션에 반영하는 방법
        4. 파일 I/O와 예외 처리의 실제 적용
        
        음영 지역이란:
        - 건물, 터널, 다리 등으로 인해 GNSS 신호가 약하거나 차단되는 구역
        - 이런 지역에서는 측위 정확도가 크게 떨어짐
        - 도심이나 실내 환경에서 흔히 발생
        
        CSV 형식: x, y, pos_std, radius (미터 단위)
        - x, y: 음영 지역 중심 좌표 (ENU)
        - pos_std: 추가 위치 오차 표준편차
        - radius: 음영 지역 영향 반경
        
        Args:
            csv_path: 음영 지역 데이터가 포함된 CSV 파일 경로
            
        Returns:
            list: (x, y, pos_std, radius) 튜플들의 리스트
        """
        zones = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        x = float(row.get('x', 'nan'))
                        y = float(row.get('y', 'nan'))
                        pos_std = float(row.get('pos_std', 'nan'))
                        radius = float(row.get('radius', 'nan'))
                        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(pos_std) and math.isfinite(radius)):
                            continue
                        zones.append((x, y, pos_std, radius))
                    except Exception:
                        continue
        except Exception as ex:
            rospy.logwarn("GNSS 음영 지역 CSV 로드 실패 %s: %s", csv_path, ex)
        rospy.loginfo("%d개의 GNSS 음영 지역을 %s에서 로드했습니다", len(zones), csv_path)
        return zones

    def _query_shadow_pos_std(self, x: float, y: float) -> float:
        """지정된 위치가 음영 지역 내부에 있는지 확인하고 추가 위치 표준편차를 반환합니다.
        
        교육 목적:
        공간 쿼리와 거리 계산 알고리즘을 학습합니다:
        1. 2D 공간에서의 점-원 충돌 검사
        2. 유클리드 거리 계산과 기하학적 판정
        3. 다중 음영 지역이 겹치는 경우의 처리
        4. 실시간 환경 조건 쿼리 시스템 구현
        
        알고리즘:
        - 각 음영 지역에 대해 차량 위치와의 거리 계산
        - 거리가 반지름보다 작으면 해당 음영 지역 내부로 판정
        - 여러 음영 지역이 겹치는 경우 최대 pos_std 값 사용
        
        Args:
            x: 쿼리할 x 좌표 (미터)
            y: 쿼리할 y 좌표 (미터)
            
        Returns:
            float: 추가 위치 표준편차 (미터). 음영 지역 외부면 0.0
        """
        if not self.shadow_zones:
            return 0.0
        max_std = 0.0
        #############################################################
        # TODO: GNSS Shadow Zone 반영
        for zx, zy, zstd, rr in self.shadow_zones:
            dx = zx - x
            dy = zy - y
            if dx * dx + dy * dy <= rr**2:
                if zstd > max_std:
                    max_std = zstd
        #############################################################
        return max_std

    def _publish_shadow_markers(self) -> None:
        """GNSS 음영 지역을 실린더 마커로 시각화하여 발행합니다.
        
        교육 목적:
        RViz 시각화와 마커 시스템을 학습합니다:
        1. RViz Marker 메시지 구조와 사용법
        2. 3D 시각화를 통한 디버깅과 검증 방법
        3. 색상 매핑을 통한 데이터 표현 기법
        4. 실시간 시각화 시스템 구현
        
        시각화 방식:
        - 각 음영 지역을 map 프레임의 실린더로 표현
        - 실린더 크기: 지름 = 2 × radius, 높이 = 0.02m
        - 색상: pos_std가 클수록 더 빨간색 (위험도 표시)
        - 위치: 지면 약간 아래 (-0.01m)에 배치하여 지면 효과 연출
        """
        if not self.enable_shadow_viz or not self.shadow_zones:
            return
        arr = MarkerArray()
        now = rospy.Time.now()
        for idx, (x, y, pos_std, radius) in enumerate(self.shadow_zones):
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = "map"
            m.ns = "gnss_shadow_zone"
            m.id = idx
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            # 디스크를 지면 약간 아래에 배치하여 지면 아래 나타나게 함
            m.pose.position.z = -0.01
            m.pose.orientation.w = 1.0
            m.scale.x = 2.0 * radius
            m.scale.y = 2.0 * radius
            m.scale.z = 0.02
            # 색상 매핑: pos_std가 높을수록 더 빨간색
            r = min(5.0, max(0.0, pos_std))
            m.color.r = r
            m.color.g = 0.2
            m.color.b = 1.0 - r
            m.color.a = 0.8
            m.lifetime = rospy.Duration(0)
            arr.markers.append(m)
        self.pub_shadow.publish(arr)


def Main():
    """GNSS 센서 시뮬레이션 노드의 메인 함수입니다.
    
    교육 목적:
    ROS 노드의 생명주기와 실행 패턴을 학습합니다:
    1. ROS 노드 초기화 과정 이해
    2. 객체 지향 프로그래밍과 ROS의 결합
    3. 로깅 시스템을 통한 상태 추적
    4. 무한 루프와 콜백 시스템의 동작 원리
    
    실행 과정:
    1. ROS 노드 초기화 ("sensor_gnss_node" 이름으로 등록)
    2. GNSSEmu 클래스 인스턴스 생성 (모든 초기화 수행)
    3. 노드 활성화 상태 확인 및 로깅
    4. rospy.spin()으로 콜백 대기 상태 진입
    """
    rospy.init_node("sensor_gnss_node")

    # GNSSEmu 클래스 인스턴스 생성
    gnss = GNSSEmu()

    # 노드 시작 로그 출력
    if hasattr(gnss, 'enabled') and gnss.enabled:
        rospy.loginfo("sensor_gnss_node가 시작되었습니다")
    else:
        rospy.loginfo("sensor_gnss_node가 비활성화 상태입니다")
    rospy.spin()


if __name__ == "__main__":
    Main()
