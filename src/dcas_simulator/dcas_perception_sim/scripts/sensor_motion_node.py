#!/usr/bin/env python
"""
Motion Sensor Simulation Node

이 노드는 차량에 장착된 모션 센서(조향각 센서, 휠 속도 센서)를 시뮬레이션합니다.
차량의 현재 상태를 기반으로 조향 휠 각도와 휠 속도 펄스를 생성합니다.

교육 목적:
1. 조향각 센서의 동작 원리와 노이즈 특성 이해
2. 휠 속도 센서(WSS)의 펄스 생성 메커니즘 학습
3. 차량 동역학과 센서 신호의 관계 이해
4. 센서 신호 처리와 노이즈 모델링 기법 학습

입력 토픽들:
- /vehicle/state: 차량의 현재 상태 (위치, 속도, 조향각 등)

출력 토픽들:
- /sensors/motion/steering_wheel_angle: 조향 휠 각도 (도 단위)
- /sensors/motion/wheel_speed_pulses: 휠 속도 펄스 신호 (불린 타입)

센서 사양:
- 조향각 센서: ±720도 범위, 가우시안 노이즈 적용
- 휠 속도 센서: 펄스/회전 방식, 속도 비례 펄스 생성
"""
import math
import rospy
from std_msgs.msg import Float32, Bool
from dcas_msgs.msg import VehicleState
import numpy as np
import time


class MotionSensorNode:
    """차량 모션 센서 시뮬레이터
    
    실제 차량의 조향각 센서와 휠 속도 센서를 시뮬레이션합니다.
    
    주요 기능:
    1. 조향 휠 각도 측정 및 노이즈 적용
    2. 휠 속도 기반 펄스 신호 생성
    3. 차량 상태가 없을 때 합성 신호 생성
    
    센서 모델링:
    - 조향각: 실제 조향각 + 가우시안 노이즈
    - 휠 속도: 회전 기반 디지털 펄스 생성
    """

    def __init__(self) -> None:
        """모션 센서 시뮬레이터 초기화
        
        ROS 매개변수를 통해 센서 설정을 로드하고,
        퍼블리셔와 구독자를 설정합니다.
        """
        # =================================================================
        # 1. ROS 노드 설정 및 센서 활성화 확인
        # =================================================================
        
        # 센서 활성화 상태 확인
        self.enabled = rospy.get_param("/sensor_motion_node/enabled", True)
        if not self.enabled:
            rospy.loginfo("Motion sensor is disabled")
            return
        
        # =================================================================
        # 2. 퍼블리셔 설정 (센서 출력)
        # =================================================================
        
        # 조향 휠 각도 퍼블리셔 (Float32, 도 단위)
        self.pub_angle = rospy.Publisher("/sensors/motion/steering_wheel_angle", Float32, queue_size=10)
        
        # 휠 속도 펄스 퍼블리셔 (Bool, 디지털 펄스)
        self.pub_pulse = rospy.Publisher("/sensors/motion/wheel_speed_pulses", Bool, queue_size=10)
        
        # =================================================================
        # 3. 센서 파라미터 설정 (Configuration에서 읽기)
        # =================================================================
        
        # 센서 업데이트 주기 (초)
        self.period = float(rospy.get_param("/sensor_motion_node/period_s", 0.02))
        
        # 시간 관련 변수들
        self.t = 0.0                    # 누적 시간
        self.t_prev = None              # 이전 타임스탬프
        
        # 펄스 생성을 위한 상태 변수들
        self.prev_angle = 0.0           # 이전 휠 회전 각도 (라디안)
        self.prev_position = 0.0        # 이전 위치 (사용되지 않음)
        
        # 차량 상태 정보 저장
        self.state: VehicleState = None  # type: ignore

        # =================================================================
        # 4. 구독자 설정 (차량 상태 정보 수신)
        # =================================================================
        
        # 차량 상태 구독자
        rospy.Subscriber("/vehicle/state", VehicleState, lambda m: setattr(self, 'state', m))
        
        # =================================================================
        # 5. 차량 물리 파라미터 설정
        # =================================================================
        
        # 휠 반지름 (미터) - 펄스 생성 계산에 사용
        self.wheel_radius = rospy.get_param("/vehicle_node/wheel_radius", 0.3)
        
        # 휠 1회전당 펄스 수 - WSS(Wheel Speed Sensor) 사양
        self.pulses_per_revolution = rospy.get_param("/vehicle_node/pulses_per_revolution", 20)
        
        # =================================================================
        # 6. 노이즈 모델 파라미터 설정
        # =================================================================
        
        # 가우시안 노이즈 활성화 여부
        self.enable_noise = rospy.get_param("/sensor_motion_node/enable_noise", True)
        
        # 조향각 노이즈 표준편차 (도 단위)
        self.angle_noise_std = rospy.get_param("/sensor_motion_node/angle_noise_std", 0.01)
        
        # 중복된 enable_noise 설정 (설정 파일 오류 대비)
        self.enable_noise = rospy.get_param("/sensor_motion_node/enable_noise", True)
        
        # =================================================================
        # 7. 랜덤 시드 및 타이머 설정
        # =================================================================
        
        # 노이즈 생성을 위한 랜덤 상태 초기화
        self.rng = np.random.RandomState(int(time.time() * 1000) % 2**32)

        # 타이머 (센서가 활성화된 경우에만 생성)
        if self.enabled:
            self.timer = rospy.Timer(rospy.Duration(self.period), self.CallbackTimer)

    # =====================================================================
    # 센서 데이터 생성 및 발행 함수들
    # =====================================================================

    def CallbackTimer(self, evt) -> None:
        """주기적으로 호출되는 센서 데이터 생성 및 발행 함수
        
        이 함수는 타이머에 의해 주기적으로 호출되며,
        차량 상태에 따라 조향각과 휠 속도 펄스를 생성합니다.
        
        처리 순서:
        1. 시간 델타 계산
        2. 조향각 및 펄스 생성 (실제 또는 합성)
        3. 노이즈 적용
        4. 센서 데이터 발행
        """
        if not self.enabled:
            return
        
        # =================================================================
        # 1. 시간 관리 및 델타 계산
        # =================================================================
        
        # 누적 시간 업데이트
        self.t += self.period
        
        # 현재 시간과 이전 시간 차이 계산 (실제 시간 기반)
        t_now = rospy.Time.now()
        if self.t_prev is not None:
            dt = max(1e-3, (t_now - self.t_prev).to_sec())  # 최소 1ms 보장
        else:
            dt = self.period  # 첫 번째 호출시 기본 주기 사용
        
        # =================================================================
        # 2. 센서 데이터 생성 (실제 또는 합성)
        # =================================================================
        
        pulse_generated = 0  # 생성된 펄스 (0 또는 1)
        
        if self.state is not None:
            # TODO: 실제 차량 조향각 사용
            # 실제 차량 상태가 있는 경우
            # 차량의 조향 휠 각도 사용 (도 단위)
            # rostopic echo /vehicle/state에서 메시지 구조 확인 가능
            angle = self.state.steering_wheel_angle_deg
            
            # 실제 차량 속도 기반 펄스 생성
            pulse_generated = self._generate_pulse_logic(dt, is_synthetic=False)
        else:
            # 차량 상태가 없는 경우 합성 신호 생성
            # 사인파를 사용한 합성 조향각 (±10도 범위)
            angle = 10.0 * math.sin(self.t)
            
            # 합성 속도 기반 펄스 생성
            pulse_generated = self._generate_pulse_logic(dt, is_synthetic=True)
        
        # =================================================================
        # 3. 시간 업데이트 및 노이즈 적용
        # =================================================================
        
        # 이전 시간 저장 (다음 델타 계산용)
        self.t_prev = t_now
        
        # TODO: 조향각에 노이즈 적용
        angle = self._apply_noise(angle)
        
        # =================================================================
        # 4. 센서 데이터 발행
        # =================================================================
        
        # 조향각 발행 (Float32, 도 단위)
        self.pub_angle.publish(Float32(data=angle))
        
        # 휠 속도 펄스 발행 (Bool, 디지털 신호)
        self.pub_pulse.publish(Bool(data=bool(pulse_generated)))
    
    # =====================================================================
    # 휠 속도 센서 시뮬레이션 함수들 (교육용 핵심 부분)
    # =====================================================================
    
    def _generate_pulse_logic(self, dt: float, is_synthetic: bool) -> int:
        """휠 속도에 기반하여 디지털 펄스 신호를 생성합니다.
        
        교육 목적:
        이 함수는 실제 WSS(Wheel Speed Sensor)의 동작 원리를 시뮬레이션합니다.
        1. 회전 엔코더의 펄스 생성 메커니즘
        2. 선속도와 각속도의 관계 (v = r × ω)
        3. 디지털 신호의 양자화 과정
        4. 펄스 카운팅 기반 속도 측정 원리
        
        WSS 동작 원리:
        1. 휠 회전 → 자기 링 또는 기어 톱니 회전
        2. 홀 센서/자기 센서가 주기적 신호 감지
        3. 1회전당 정해진 수의 펄스 생성
        4. 펄스 주파수 ∝ 휠 회전 속도
        
        계산 과정:
        - 선속도(v) → 휠 각속도(ω = v/r)
        - 각속도 → 회전 위치 누적
        - 회전 위치 → 펄스 위치 계산
        - 펄스 위치 → 디지털 신호 (0 또는 1)
        
        Args:
            dt: 시간 간격 (초)
            is_synthetic: 합성 신호 여부
            
        Returns:
            int: 생성된 펄스 (0 또는 1)
        """        
        if is_synthetic:
            # 합성 모드: 사인파 기반 속도 생성 (교육/테스트용)
            # 기본 속도 1m/s에 사인파 변동 추가 (0~2m/s 범위)
            base_speed = 1.0 + math.sin(self.t)
        else:
            # TODO: 실제 차량 속도 사용
            # 실제 모드: 차량 상태의 전진 속도 사용
            # 음수 속도 방지 (후진시 0으로 처리)
            # rostopic echo /vehicle/state에서 메시지 구조 확인 가능
            base_speed = max(0.0, self.state.twist.linear.x) # 음수 속도 방지
        
        #############################################################
        # TODO: Wheel Speed Sensor 펄스 생성 로직 구현
        
        # 위치 변화량 계산
        delta_position = base_speed * dt

        # 선형 거리로 휠 각도 계산
        # θ = s / r (라디안)
        wheel_angle = delta_position / self.wheel_radius
        
        # 현재 누적 휠 회전 각도 계산
        curr_angle = self.prev_angle + wheel_angle
        
        # 각도를 [0, 2π] 범위로 정규화 (한 바퀴 = 2π 라디안)
        # np.remainder() 함수 사용
        curr_angle = np.remainder(curr_angle, 2*math.pi)

        # 펄스 위치 계산
        # position_in_pulse_cycle = (curr_angle / (2π)) * 2 * pulses_per_revolution
        position_in_pulse_cycle = (curr_angle / (2*math.pi)) * 2 * self.pulses_per_revolution

        # 펄스 상태 계산
        # 연속적인 위치를 가장 가까운 정수로 양자화
        # np.round() 함수 사용
        quant_position_in_pulse_cycle = np.round(position_in_pulse_cycle)

        # 펄스 신호 생성
        # 0 또는 1
        pulse = np.remainder(quant_position_in_pulse_cycle, 2)

        # 다음 계산을 위해 현재 각도 저장
        self.prev_angle = curr_angle
        #############################################################
        return pulse
    
    # =====================================================================
    # 센서 노이즈 모델링 함수들
    # =====================================================================
        
    def _apply_noise(self, angle):
        """조향각 센서 측정값에 가우시안 노이즈를 적용합니다.
        
        교육 목적:
        실제 센서의 노이즈 특성을 시뮬레이션하여 학생들이 다음을 학습할 수 있습니다:
        1. 센서 노이즈의 종류와 특성
        2. 가우시안(정규) 분포의 노이즈 모델
        3. 신호 대 잡음비(SNR)의 개념
        4. 노이즈가 제어 시스템에 미치는 영향
        
        실제 조향각 센서 노이즈 특성:
        - 열 노이즈: 전자 부품의 온도에 의한 노이즈
        - 양자화 노이즈: ADC 변환 과정의 디지털화 오차
        - 진동 노이즈: 차량 진동에 의한 기계적 노이즈
        - 전자기 간섭: 주변 전자 장치의 영향
        
        노이즈 모델:
        측정값 = 실제값 + 가우시안_노이즈(평균=0, 표준편차=σ)

        Args:
            angle (float): 실제 조향 휠 각도 (도 단위)

        Returns:
            float: 노이즈가 적용된 조향 휠 각도 (도 단위)
        """
        # 노이즈 적용 여부 확인
        if not self.enable_noise:
            return angle
        
        # =================================================================
        # 가우시안 노이즈 생성 및 적용
        # =================================================================

        # TODO: 평균 0, 표준편차 self.angle_noise_std인 가우시안 노이즈 생성
        noisy = self.rng.normal(0.0, self.angle_noise_std)
        
        # 실제 신호에 노이즈 추가
        angle_noisy = angle + noisy

        return angle_noisy


# =====================================================================
# 메인 함수 및 노드 실행
# =====================================================================

def main():
    """메인 함수
    
    ROS 노드를 초기화하고 모션 센서 시뮬레이터를 실행합니다.
    """
    
    # ROS 노드 초기화
    rospy.init_node("sensor_motion_node")

    # MotionSensorNode 클래스 인스턴스 생성
    motion_sensor = MotionSensorNode()
    
    # 센서 활성화 상태에 따른 로그 출력
    if hasattr(motion_sensor, 'enabled') and motion_sensor.enabled:
        rospy.loginfo("sensor_motion_node started")
    else:
        rospy.loginfo("sensor_motion_node is disabled")
    
    # ROS 스핀 (무한 대기 - 콜백 함수들이 실행됨)
    rospy.spin()


if __name__ == "__main__":
    main()


