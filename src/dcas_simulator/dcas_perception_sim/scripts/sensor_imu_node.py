#!/usr/bin/env python
"""
IMU Sensor Simulation Node

이 노드는 차량에 장착된 IMU(Inertial Measurement Unit) 센서를 시뮬레이션합니다.
차량의 현재 상태를 기반으로 관성 측정 데이터(자세, 각속도, 선형가속도)를 생성합니다.

교육 목적:
1. IMU 센서의 동작 원리와 측정 데이터 이해
2. 센서 노이즈와 바이어스 모델링 학습
3. 관성 항법과 센서 융합의 기초 이해
4. 쿼터니언과 회전 행렬의 실제 활용 학습

입력 토픽들:
- /vehicle/state: 차량의 현재 상태 (위치, 속도, 가속도, 자세 등)

출력 토픽들:
- /sensors/imu/data: IMU 센서 데이터 (자세, 각속도, 선형가속도)

센서 사양:
- 자세 측정: 쿼터니언 형식, 가우시안 노이즈 적용
- 각속도 측정: 3축 자이로스코프, 바이어스 및 노이즈 적용
- 선형가속도 측정: 3축 가속도계, 바이어스 및 노이즈 적용
- 바이어스 모델: 랜덤 워크 과정으로 시간에 따른 드리프트 시뮬레이션
"""
import math
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
from dcas_msgs.msg import VehicleState
import numpy as np
from scipy.spatial.transform import Rotation as R
import time


class IMUEmu:
    """IMU 센서 시뮬레이터
    
    실제 차량의 IMU 센서를 시뮬레이션합니다.
    
    주요 기능:
    1. 자세(Orientation) 측정 및 노이즈 적용
    2. 각속도(Angular Velocity) 측정 및 바이어스/노이즈 적용
    3. 선형가속도(Linear Acceleration) 측정 및 바이어스/노이즈 적용
    4. 시간에 따른 센서 바이어스 드리프트 시뮬레이션
    
    센서 모델링:
    - 측정값 = 실제값 + 바이어스 + 가우시안 노이즈
    - 바이어스 = 랜덤 워크 과정 (시간에 따른 드리프트)
    """
    
    def __init__(self) -> None:
        """IMU 센서 시뮬레이터 초기화
        
        ROS 매개변수를 통해 센서 설정을 로드하고,
        퍼블리셔와 구독자를 설정합니다.
        """
        # =================================================================
        # 1. ROS 노드 설정 및 센서 활성화 확인
        # =================================================================
        
        # 센서 활성화 상태 확인
        self.enabled = rospy.get_param("/sensor_imu_node/enabled", True)
        if not self.enabled:
            rospy.loginfo("IMU sensor is disabled")
            return
        
        # =================================================================
        # 2. 퍼블리셔 설정 (센서 출력)
        # =================================================================
        
        # IMU 데이터 퍼블리셔 (sensor_msgs/Imu)
        self.pub = rospy.Publisher("/sensors/imu/data", Imu, queue_size=10)
        
        # =================================================================
        # 3. 센서 파라미터 설정 (Configuration에서 읽기)
        # =================================================================
        
        # 센서 업데이트 주기 (초)
        self.period = float(rospy.get_param("/sensor_imu_node/period_s", 0.02))
        
        # 시간 관련 변수들
        self.t = 0.0                    # 누적 시간
        
        # 차량 상태 정보 저장
        self.state: VehicleState = None  # type: ignore

        # =================================================================
        # 4. 구독자 설정 (차량 상태 정보 수신)
        # =================================================================
        
        # 차량 상태 구독자
        rospy.Subscriber("/vehicle/state", VehicleState, lambda m: setattr(self, 'state', m))
        
        # =================================================================
        # 5. 물리 상수 설정
        # =================================================================
        
        # 중력가속도 (m/s^2)
        self.g = rospy.get_param("/sensor_imu_node/gravity", 9.81)
        
        # =================================================================
        # 6. 가우시안 노이즈 모델 파라미터 설정
        # =================================================================
        
        # 자세 노이즈 표준편차 (라디안)
        self.orientation_noise_std = rospy.get_param("/sensor_imu_node/orientation_noise_std", 0.01)
        
        # 각속도 노이즈 표준편차 (rad/s)
        self.angular_velocity_noise_std = rospy.get_param("/sensor_imu_node/angular_velocity_noise_std", 0.01)
        
        # 선형가속도 노이즈 표준편차 (m/s^2)
        self.linear_acceleration_noise_std = rospy.get_param("/sensor_imu_node/linear_acceleration_noise_std", 0.01)
        
        # 가우시안 노이즈 활성화 여부
        self.enable_gaussian_noise = rospy.get_param("/sensor_imu_node/enable_gaussian_noise", True)

        # =================================================================
        # 7. 바이어스(랜덤 워크) 모델 파라미터 설정
        # =================================================================
        
        # 바이어스 시뮬레이션 활성화/비활성화
        self.enable_bias = rospy.get_param("/sensor_imu_node/enable_bias", True)
        
        # 랜덤 워크 강도 설정 (단위: 자이로 rad/s/sqrt(s), 가속도계 m/s^2/sqrt(s))
        self.gyro_bias_rw_std = rospy.get_param("/sensor_imu_node/gyro_bias_rw_std", 0.001)
        self.accel_bias_rw_std = rospy.get_param("/sensor_imu_node/accel_bias_rw_std", 0.02)
        
        # 바이어스 상태값 (0으로 시작)
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        
        # 바이어스 제한값 설정 (절댓값, 대칭 경계)
        self.enable_bias_limits = rospy.get_param("/sensor_imu_node/enable_bias_limits", True)
        self.gyro_bias_limit = rospy.get_param("/sensor_imu_node/gyro_bias_limit", 0.005)     # rad/s
        self.accel_bias_limit = rospy.get_param("/sensor_imu_node/accel_bias_limit", 0.05)   # m/s^2
        
        # =================================================================
        # 8. 공분산 행렬 계산 및 랜덤 시드 설정
        # =================================================================
        
        # 공분산 행렬을 위한 분산값 계산
        self._var_orient = self.orientation_noise_std ** 2
        self._var_gyro   = self.angular_velocity_noise_std ** 2
        self._var_acc    = self.linear_acceleration_noise_std ** 2
        
        # 노이즈 생성을 위한 랜덤 상태 초기화
        self.rng = np.random.RandomState(int(time.time() * 1000) % 2**32)
        
        # =================================================================
        # 9. 타이머 설정
        # =================================================================
        
        # 타이머 (센서가 활성화된 경우에만 생성)
        if self.enabled:
            self.timer = rospy.Timer(rospy.Duration(self.period), self.CallbackTimer)

    # =====================================================================
    # IMU 데이터 생성 및 발행 함수들
    # =====================================================================
    
    def CallbackTimer(self, _evt) -> None:
        """주기적으로 호출되는 IMU 데이터 생성 및 발행 함수
        
        이 함수는 타이머에 의해 주기적으로 호출되며,
        차량 상태에 따라 IMU 센서 데이터를 생성합니다.
        
        처리 순서:
        1. IMU 메시지 초기화
        2. 차량 상태 기반 센서 데이터 설정
        3. 바이어스 상태 업데이트
        4. 노이즈 적용 및 공분산 설정
        5. IMU 데이터 발행
        """
        if not self.enabled:
            return
        
        # =================================================================
        # 1. 시간 관리 및 IMU 메시지 초기화
        # =================================================================
        
        # 누적 시간 업데이트
        self.t += self.period
        
        # IMU 메시지 생성 및 헤더 설정
        msg = Imu()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "imu"

        # =================================================================
        # 2. 센서 데이터 설정 (실제 또는 기본값)
        # =================================================================
        
        # 차량 상태에 따른 IMU 데이터 설정
        if self.state is not None:
            #############################################################
            # TODO: 상태 정보 전달 - IMU 노드에서 필요한 상태 정보 전달
            # 차량 상태 정보에서 필요한 값 추출
            # rostopic echo /vehicle/state에서 메시지 구조 확인 가능
            # 자세 정보 (쿼터니언)
            msg.orientation = self.state.pose.orientation
            
            # 각속도 (rad/s) - 3축 자이로스코프 데이터
            msg.angular_velocity.x = self.state.twist.angular.x
            msg.angular_velocity.y = self.state.twist.angular.y
            msg.angular_velocity.z = self.state.twist.angular.z
            
            # 선형가속도 (m/s^2) - 3축 가속도계 데이터
            msg.linear_acceleration.x = self.state.acceleration.x
            msg.linear_acceleration.y = self.state.acceleration.y
            msg.linear_acceleration.z = self.state.acceleration.z
            #############################################################
        else:
            # 차량 상태가 없는 경우 기본값 설정
            msg.angular_velocity.z = 0.0
            msg.linear_acceleration.x = 0.0
            msg.orientation = Quaternion(0, 0, 0, 1)
        
        # =================================================================
        # 3. 센서 특성 적용 (바이어스, 노이즈, 공분산)
        # =================================================================
        
        # TODO: 바이어스 상태를 측정 노이즈 적용 전에 업데이트
        self._update_biases()
        
        # TODO IMU 메시지에 노이즈 적용 함수 완성
        msg = self._apply_noise(msg)
        
        # TODO IMU 메시지에 공분산 설정 함수 완성
        msg = self._set_covariances(msg)
        
        # =================================================================
        # 4. IMU 데이터 발행
        # =================================================================
        
        self.pub.publish(msg)
        
    # =====================================================================
    # 센서 노이즈 모델링 함수들 (교육용 핵심 부분)
    # =====================================================================
        
    def _apply_noise(self, msg: Imu) -> Imu:
        """IMU 측정값에 가우시안 노이즈를 적용합니다.
        
        교육 목적:
        실제 IMU 센서의 노이즈 특성을 시뮬레이션하여 다음을 학습할 수 있습니다:
        1. MEMS 센서의 노이즈 특성과 종류
        2. 자이로스코프와 가속도계의 오차 모델
        3. 쿼터니언 회전에서의 노이즈 적용 방법
        4. 센서 융합에서 노이즈가 미치는 영향
        
        실제 IMU 노이즈 특성:
        - 열 노이즈: 온도 변화에 의한 센서 드리프트
        - 양자화 노이즈: ADC 변환 과정의 디지털화 오차
        - 진동 노이즈: 기계적 진동에 의한 측정 오차
        - 바이어스 드리프트: 시간에 따른 영점 이동
        
        노이즈 모델:
        - 자세: 회전 벡터 공간에서 가우시안 노이즈 적용
        - 각속도/가속도: 바이어스 + 가우시안 노이즈

        Args:
            msg: IMU 메시지

        Returns:
            msg: 노이즈가 적용된 IMU 메시지
        """
        # 노이즈 적용 여부 확인
        if not self.enable_gaussian_noise:
            return msg

        # 자세에 가우시안 노이즈 추가 (회전 벡터 공간에서)
        rot = R.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        noise_rot = self.rng.normal(0.0, self.orientation_noise_std, size=3)
        noise_quat = R.from_rotvec(noise_rot)
        rot_noisy = rot * noise_quat
        msg.orientation.x = rot_noisy.as_quat()[0]
        msg.orientation.y = rot_noisy.as_quat()[1]
        msg.orientation.z = rot_noisy.as_quat()[2]
        msg.orientation.w = rot_noisy.as_quat()[3]
        
        # 바이어스 값 가져오기 (활성화된 경우)
        if self.enable_bias:
            avx_bias = self.gyro_bias[0]
            avy_bias = self.gyro_bias[1]
            avz_bias = self.gyro_bias[2]
            lax_bias = self.accel_bias[0]
            lay_bias = self.accel_bias[1]
            laz_bias = self.accel_bias[2]
        else:
            avx_bias = 0.0; avy_bias = 0.0; avz_bias = 0.0
            lax_bias = 0.0; lay_bias = 0.0; laz_bias = 0.0

        #############################################################
        # TODO: 노이즈와 바이어스 적용
        # self.rng.normal 함수 사용
        # 각속도에 바이어스와 가우시안 노이즈 추가
        angular_velocity_x_noisy = msg.angular_velocity.x + avx_bias + self.rng.normal(0.0, self.angular_velocity_noise_std)
        angular_velocity_y_noisy = msg.angular_velocity.y + avy_bias + self.rng.normal(0.0, self.angular_velocity_noise_std)
        angular_velocity_z_noisy = msg.angular_velocity.z + avz_bias + self.rng.normal(0.0, self.angular_velocity_noise_std)
        
        # 선형가속도에 바이어스와 가우시안 노이즈 추가
        linear_acceleration_x_noisy = msg.linear_acceleration.x + lax_bias + self.rng.normal(0.0, self.linear_acceleration_noise_std)
        linear_acceleration_y_noisy = msg.linear_acceleration.y + lay_bias + self.rng.normal(0.0, self.linear_acceleration_noise_std)
        linear_acceleration_z_noisy = msg.linear_acceleration.z + laz_bias + self.rng.normal(0.0, self.linear_acceleration_noise_std)
        #############################################################

        # 노이즈가 적용된 값들을 메시지에 설정
        msg.angular_velocity.x = angular_velocity_x_noisy
        msg.angular_velocity.y = angular_velocity_y_noisy
        msg.angular_velocity.z = angular_velocity_z_noisy
        msg.linear_acceleration.x = linear_acceleration_x_noisy
        msg.linear_acceleration.y = linear_acceleration_y_noisy
        msg.linear_acceleration.z = linear_acceleration_z_noisy

        return msg
    
    # =====================================================================
    # 바이어스 모델링 함수들
    # =====================================================================
    
    def _update_biases(self) -> None:
        """바이어스 상태를 랜덤 워크로 진화시킵니다.
        
        교육 목적:
        실제 IMU 센서의 바이어스 드리프트 특성을 시뮬레이션하여 다음을 학습할 수 있습니다:
        1. 센서 바이어스의 시간적 변화 특성
        2. 랜덤 워크 과정의 모델링과 시뮬레이션
        
        바이어스 모델:
        - b_{k+1} = b_k + w, 여기서 w ~ N(0, (sigma_rw^2 * dt) I)
        - 브라운 운동(Brownian Motion)과 유사한 확률 과정
        - 시간이 지날수록 바이어스가 누적되어 드리프트 발생
        """
        # 바이어스 업데이트 활성화 여부 확인
        if not self.enable_bias:
            return
            
        dt = self.period
        #############################################################
        # TODO: 바이어스 상태 업데이트
        # 랜덤 워크 증분 계산; sqrt(dt)로 스케일링
        # self.rng.normal 함수 사용
        # 자이로스코프 바이어스 업데이트: b_gyro = b_gyro + N(0, σ_gyro^2 * dt)
        self.gyro_bias = self.gyro_bias + self.rng.normal(0, self.gyro_bias_rw_std**2 * dt)

        # 가속도계 바이어스 업데이트: b_accel = b_accel + N(0, σ_accel^2 * dt)
        self.accel_bias = self.accel_bias + self.rng.normal(0, self.accel_bias_rw_std**2 * dt)

        # 바이어스 제한값 적용 (센서 물리적 한계 내로 제한)
        if self.enable_bias_limits:
            # 축별로 대칭 제한값으로 제한
            self.gyro_bias = np.clip(self.gyro_bias, -self.gyro_bias_limit, self.gyro_bias_limit)
            self.accel_bias = np.clip(self.accel_bias, -self.accel_bias_limit, self.accel_bias_limit)
        #############################################################
    
    # =====================================================================
    # 공분산 행렬 설정 함수들
    # =====================================================================
    
    def _set_covariances(self, msg: Imu) -> None:
        """IMU 메시지에 공분산 행렬을 설정합니다.
        
        교육 목적:
        센서 융합과 확률적 로봇공학에서 중요한 공분산 행렬의 설정 방법을 학습합니다:
        1. 센서 불확실성의 수치적 표현 방법
        2. 대각선 공분산 행렬의 의미 (축간 독립성 가정)
        
        공분산 행렬 구조:
        - 9x9 행렬을 1차원 배열로 표현 (row-major order)
        - 대각선 요소: 각 축의 분산값 (σ²)
        - 비대각선 요소: 축간 상관관계 (일반적으로 0)

        Args:
            msg: 공분산을 설정할 IMU 메시지
        
        Returns:
            msg: 공분산 행렬이 설정된 IMU 메시지
        """
        # 공분산 행렬 초기화 (3x3 행렬을 1차원 배열로 표현)
        oc = [0.0]*9   # orientation_covariance (자세 공분산)
        avc = [0.0]*9  # angular_velocity_covariance (각속도 공분산)
        lac = [0.0]*9  # linear_acceleration_covariance (선형가속도 공분산)
        
        #############################################################
        # TODO: 공분산 행렬 설정
        # 대각선 요소에 분산값 설정 (σ² = 표준편차²)
        # 자세 공분산 행렬 (3x3): [σ_roll², 0, 0; 0, σ_pitch², 0; 0, 0, σ_yaw²]
        # 각속도 공분산 행렬 (3x3): [σ_wx², 0, 0; 0, σ_wy², 0; 0, 0, σ_wz²]
        # 선형가속도 공분산 행렬 (3x3): [σ_ax², 0, 0; 0, σ_ay², 0; 0, 0, σ_az²]
        oc[0] = oc[4] = oc[8] = self._var_orient
        avc[0] = avc[4] = avc[8] = self._var_gyro
        lac[0] = lac[4] = lac[8] = self._var_acc
        
        #############################################################
        
        # IMU 메시지에 공분산 행렬 할당
        msg.orientation_covariance = oc
        msg.angular_velocity_covariance = avc
        msg.linear_acceleration_covariance = lac
        #############################################################

        return msg


# =====================================================================
# 메인 함수 및 노드 실행
# =====================================================================

def main():
    """메인 함수
    
    ROS 노드를 초기화하고 IMU 센서 시뮬레이터를 실행합니다.
    """
    
    # ROS 노드 초기화
    rospy.init_node("sensor_imu_node")

    # IMUEmu 클래스 인스턴스 생성
    imu = IMUEmu()
    
    # 센서 활성화 상태에 따른 로그 출력
    if hasattr(imu, 'enabled') and imu.enabled:
        rospy.loginfo("sensor_imu_node started")
    else:
        rospy.loginfo("sensor_imu_node is disabled")
    
    # ROS 스핀 (무한 대기 - 콜백 함수들이 실행됨)
    rospy.spin()


if __name__ == "__main__":
    main()