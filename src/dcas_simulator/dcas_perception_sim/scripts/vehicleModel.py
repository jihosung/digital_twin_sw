# -*- coding: utf-8 -*-
"""
Vehicle Dynamics Model

이 모듈은 차량의 종/횡방향 동역학을 시뮬레이션하는 통합 차량 모델을 제공합니다.
자전거 모델 기반의 차량 동역학과 오일러 적분법을 사용하여 실시간 시뮬레이션을 지원합니다.

교육 목적:
1. 차량 동역학의 기본 원리와 수학적 모델링 이해
2. 종방향/횡방향 동역학의 분리 및 통합 학습
3. 키네마틱 모델과 다이나믹 모델의 차이점 이해
4. 자전거 모델(Bicycle Model)의 구현과 응용
5. 수치적분 방법과 안정성 고려사항 학습

주요 기능:
- 종방향 동역학: 가속/제동, 저항력, 모터 토크 모델링
- 횡방향 동역학: 코너링 강성, 슬립각, 측력 계산
- 키네마틱/다이나믹 모델 자동 전환 (저속/고속)
- 실시간 수치적분 (오일러 방법)

입력/출력:
- 입력: [조향각(rad), 스로틀(-1~1)]
- 상태: [x, y, yaw, vx, vy, yaw_rate, ax, ay]
- 출력: 다음 시간 스텝의 상태 벡터
"""
import numpy as np

class VehicleModel:
    """차량 동역학 시뮬레이터
    
    자전거 모델 기반의 차량 동역학을 시뮬레이션합니다.
    
    주요 특징:
    1. 종방향/횡방향 동역학 통합 모델
    2. 키네마틱-다이나믹 모델 자동 전환
    3. 오일러 적분법을 통한 실시간 시뮬레이션
    4. 물리적 제약 조건 및 포화 한계 적용
    
    동역학 모델:
    - 종방향: 모터 토크, 브레이크 토크, 저항력 고려
    - 횡방향: 자전거 모델, 코너링 강성, 슬립각 기반
    """

    # =================================================================
    # 생성자 및 차량 매개변수 설정
    # =================================================================
    def __init__(self):
        """차량 동역학 모델 초기화
        
        실제 차량의 물리적 특성을 기반으로 매개변수를 설정합니다.
        매개변수는 일반적인 승용차 수준의 값들로 구성되어 있습니다.
        """
        # =================================================================
        # 종방향 동역학 매개변수
        # =================================================================
        self.m = 1319.91                    # 차량 질량 [kg]
        self.Iw = 0.3312                    # 바퀴 관성 모멘트 [kg m^2] (등가값)
        self.Rw = 0.305                     # 유효 바퀴 반지름 [m]
        self.rho = 1.225                    # 공기 밀도 [kg/m^3]
        self.Cd = 0.32                      # 항력 계수 (무차원)
        self.A = 1.55 * 1.8                 # 정면 투영 면적 [m^2]
        self.k_R = 0.015                    # 구름 저항 계수 (무차원)
        self.g = 9.81                       # 중력 가속도 [m/s^2]
        self.slope = 0.0                    # 도로 경사 [rad]
        self.gear_ratio = 1.0 / 7.98        # 기어비 (모터축/바퀴축)
        self.Im = 0.015                     # 모터 관성 모멘트 [kg m^2]
        self.It = 0.06                      # 변속기 관성 모멘트 [kg m^2]

        # 토크 매핑 (페달 입력 → 토크 변환)
        self.motor_torque_max = 2000.0      # 최대 모터 토크 [Nm]
        self.brake_torque_max = 6000.0      # 최대 브레이크 토크 [Nm]
        self.accel_const = 293.1872055      # 가속 페달 상수: T_motor = accel_const * pedal_accel
        self.brake_const = 4488.075         # 브레이크 페달 상수: T_brake = brake_const * pedal_brake

        # =================================================================
        # 횡방향 동역학 매개변수 (자전거 모델)
        # =================================================================
        self.Iz = 2093.38                   # 요 관성 모멘트 [kg m^2]
        self.lf = 1.302                     # 무게중심에서 전륜축까지 거리 [m]
        self.lr = 1.398                     # 무게중심에서 후륜축까지 거리 [m]
        self.width = 1.6                    # 차량 폭 [m]
        self.h = 0.5                        # 무게중심 높이 [m]

        self.Caf = 154700.0                 # 전륜 코너링 강성 [N/rad]
        self.Car = 183350.0                 # 후륜 코너링 강성 [N/rad]
        self.mu = 0.9                       # 마찰 계수 (여기서는 직접 사용하지 않음)

        # =================================================================
        # 모델 전환 및 물리적 제한값
        # =================================================================
        self.v_switch = 5.0                 # 키네마틱/다이나믹 모델 전환 속도 [m/s]
        self.max_steer = np.deg2rad(30.0)   # 최대 조향각 [rad]
        self.max_alpha = np.deg2rad(7.0)    # 최대 슬립각 [rad]
        self.beta_max = np.deg2rad(30.0)    # 최대 차체 슬립각 [rad]
        self.yaw_rate_max = np.deg2rad(120.0) # 최대 요 각속도 [rad/s]

        # =================================================================
        # 내부 시뮬레이션 상태 변수
        # =================================================================
        self._x = 0.0           # 종방향 위치 [m]
        self._y = 0.0           # 횡방향 위치 [m]
        self._yaw = 0.0         # 요 각도 [rad]
        self._v = 0.0           # 종방향 속도 [m/s]
        self._beta = 0.0        # 차체 슬립각 [rad]
        self._yaw_rate = 0.0    # 요 각속도 [rad/s]

    # =================================================================
    # 유틸리티 함수들
    # =================================================================
    def _sync_from_legacy_states(self, states):
        """레거시 상태 벡터에서 내부 상태를 동기화합니다.
        
        교육 목적:
        외부 상태 벡터 형식과 내부 상태 표현 간의 변환을 학습합니다:
        1. 상태 벡터의 구조와 의미 이해
        2. 속도 성분으로부터 슬립각 계산 방법
        3. 물리적 제약 조건 적용 (비음수 속도, 슬립각 제한)
        
        Args:
            states: [x, y, psi, vx, vy, yaw_rate, ax, ay] 8차원 상태 벡터
        """
        self._x = float(states[0])          # x 위치
        self._y = float(states[1])          # y 위치
        self._yaw = float(states[2])        # 요 각도
        vx = max(0.0, float(states[3]))     # 종속도 (비음수)
        vy = float(states[4])               # 횡속도
        self._yaw_rate = float(states[5])   # 요 각속도

        # vx, vy로부터 차체 슬립각 추정
        if vx > 1e-6:
            self._beta = float(np.clip(np.arcsin(np.clip(vy / vx, -1.0, 1.0)),
                                       -self.beta_max, self.beta_max))
        else:
            self._beta = 0.0
        self._v = vx

    def calculate_equivalent_inertia(self):
        """등가 관성 모멘트를 계산합니다.
        
        교육 목적:
        회전 관성이 동역학에 미치는 영향을 학습합니다:
        1. 모터, 변속기, 바퀴의 각 관성 모멘트 합성
        2. 기어비에 따른 관성 환산 방법 이해
        3. 차량 병진 운동과 회전 운동의 관계
        4. 등가 관성이 가속 성능에 미치는 영향
        
        등가 관성 계산:
        J_eq = Im + It + (Iw + m*Rw²) * gear_ratio²
        
        Returns:
            float: 모터축 기준 등가 관성 모멘트 [kg⋅m²]
        """
        # TODO: 등가 관성 계산식 완성 Done
        J_eq = self.Im + self.It + (self.Iw + self.m * self.Rw**2) * self.gear_ratio**2 
        return J_eq
    
    def calculate_resistances(self, velocity):
        """차량에 작용하는 저항력들을 계산합니다.
        
        교육 목적:
        차량 운동에 영향을 미치는 다양한 저항 요소를 학습합니다:
        1. 구름 저항: 타이어 변형과 노면 상호작용
        2. 공기 저항: 속도의 제곱에 비례하는 항력
        3. 중력 저항: 경사로에서의 중력 성분
        4. 각 저항의 물리적 원리와 수학적 모델링
        
        저항력 계산:
        - F_roll = k_R * m * g * cos(slope) * sgn(v)
        - F_aero = 0.5 * ρ * Cd * A * v²
        - F_grav = m * g * sin(slope)
        
        Args:
            velocity: 차량 속도 [m/s]
            
        Returns:
            tuple: (구름저항, 공기저항, 중력저항) [N]
        """
        # 부호 함수 (속도 방향에 따른 저항 방향 결정)
        if velocity > 0.0:
            sgn = 1.0
        elif velocity == 0.0:
            sgn = 0.0
        else:
            sgn = -1.0
        
        # TODO: 저항 계산식 완성
        # 구름저항
        F_roll = self.k_R*self.m*self.g*np.cos(self.slope)*sgn
        
        # 공기저항
        F_aero = 0.5*self.rho*self.Cd*self.A*(velocity**2)
        
        # 중력저항 (경사)
        F_grav = self.m*self.g*np.sin(self.slope)
        
        return F_roll, F_aero, F_grav
    
    # =================================================================
    # 페달 입력을 토크로 변환
    # =================================================================
    def _pedal_to_torques(self, pedal_input):
        """페달 입력을 모터 토크와 브레이크 토크로 변환합니다.
        
        교육 목적:
        차량의 가속/제동 시스템과 토크 매핑을 학습합니다:
        1. 가속/브레이크 페달의 물리적 의미와 범위
        2. 선형 토크 매핑과 실제 차량의 특성 곡선
        3. 모터 토크와 브레이크 토크의 독립적 제어
        4. 토크 제한과 물리적 한계 적용
        
        페달 입력 범위:
        - +1.0: 최대 가속 (모터 토크만 적용)
        - 0.0: 중립 (토크 없음)
        - -1.0: 최대 제동 (브레이크 토크만 적용)
        
        Args:
            pedal_input: 페달 입력 [-1.0 ~ +1.0], 양수=가속, 음수=브레이크
            
        Returns:
            tuple: (모터토크, 브레이크토크) [Nm]
        """
        u = np.clip(pedal_input, -1.0, 1.0)
        
        # TODO: 페달 입력에 따른 모터 토크 및 브레이크 토크 계산식 완성, 물리적 제한 적용
        if u >= 0.0:
            Tm = self.accel_const*u if self.accel_const*u < self.motor_torque_max else self.motor_torque_max
            Tb = 0.0
        else:
            Tm = 0.0
            Tb = self.brake_const*u if self.brake_const*u < self.brake_torque_max else self.brake_torque_max                  
        return Tm, -Tb

    # =================================================================
    # 종방향 동역학 (오일러 적분)
    # =================================================================
    def _longitudinal_step(self, pedal_input, dt):
        """종방향 동역학을 한 스텝 적분합니다.
        
        교육 목적:
        차량의 종방향 운동 방정식과 수치적분을 학습합니다:
        1. 뉴턴의 제2법칙을 회전 운동에 적용
        2. 토크 평형 방정식과 각가속도 계산
        3. 모터축과 바퀴축 간의 토크/속도 관계
        4. 오일러 적분법의 구현과 안정성
        
        운동 방정식:
        T_load = gear_ratio * Rw * (F_roll + F_aero + F_grav)
        J_eq * ω̇ = Tm - (gear_ratio * Tb) - T_load
        ax = (ω̇ * gear_ratio * Rw)
        
        Args:
            pedal_input: 페달 입력 [-1.0 ~ +1.0]
            dt: 시간 스텝 [s]
            
        Returns:
            float: 종가속도 [m/s²]
        """
        # 등가 관성 계산
        J_eq = self.calculate_equivalent_inertia() # TODO: 등가 관성 계산식 완성 Done
        
        # 페달 입력을 토크로 변환
        Tm, Tb = self._pedal_to_torques(pedal_input) # TODO: 페달 입력에 따른 토크 계산식 완성 Done

        # 저항력 계산
        v = self._v
        F_roll, F_aero, F_grav = self.calculate_resistances(v) # TODO: 저항력 계산식 완성 Done

        # 로드 토크 (바퀴축 → 모터축 환산)
        T_load = self.gear_ratio * self.Rw * (F_roll + F_aero + F_grav)  # TODO: 로드 토크 계산식 작성 Done
        
        # 모터 각가속도 계산 (토크 평형 방정식)
        omega_dot = (Tm - (self.gear_ratio * Tb) - T_load) / J_eq  # TODO: 모터 각가속도 계산식 작성 Done

        # 종가속도
        a_x = omega_dot * self.gear_ratio * self.Rw # TODO: 종가속도 계산식 작성 Done

        # 속도 업데이트 (비음수 제약)
        self._v = max(0.0, self._v + a_x * dt)

        return a_x

    # =================================================================
    # 횡방향 동역학 (오일러 적분)
    # =================================================================
    def _lateral_step(self, steering, dt):
        """횡방향 동역학을 한 스텝 적분합니다.
        
        교육 목적:
        자전거 모델과 횡방향 동역학의 핵심 개념을 학습합니다:
        1. 키네마틱 모델 vs 다이나믹 모델의 차이점과 적용 범위
        2. 슬립각의 물리적 의미와 계산 방법
        3. 코너링 강성과 측력의 관계
        4. 요 모멘트 평형과 횡방향 힘 평형
        5. 저속/고속에서의 차량 거동 차이
        
        모델 전환:
        - v < v_switch: 키네마틱 모델 (기하학적 관계)
        - v ≥ v_switch: 다이나믹 모델 (타이어 슬립 고려)
        
        Args:
            steering: 조향각 [rad]
            dt: 시간 스텝 [s]
            
        Returns:
            tuple: (요각, 요각속도) [rad, rad/s]
        """

        sgn = 1 if steering > 0 else -1
        delta = steering if abs(steering) <= self.max_steer else sgn*self.max_steer # TODO: 조향각 제한 Done

        v = self._v
        yaw_rate = self._yaw_rate
        beta = self._beta

        # TODO: 속도에 따른 모델 전환
        if v < 20 * 1000/3600:
            # =========================================================
            # 키네마틱 모델 (저속): 타이어 슬립 무시, 순수 기하학적 관계
            # =========================================================
            if abs(delta) < 1e-6:
                beta_target = 0.0
                yaw_rate_new = 0.0
            else:
                # TODO: 키네마틱 모델 슬립각 및 요 각속도 계산식 작성
                # 키네마틱 자전거 모델:
                # beta = arctan(lr * tan(delta) / (lf + lr))
                # yaw_rate = v * cos(beta) * tan(delta) / (lf + lr)
                beta_target = np.arctan(self.lr*np.tan(delta) / (self.lf + self.lr))   # TODO: 슬립각 계산식 작성 - delta_r은 0으로 가정 done
                
                yaw_rate_new = v*np.cos(beta_target)*np.tan(delta) / (self.lf + self.lr)  # TODO: 요 각속도 계산식 작성 - delta_r은 0으로 가정 done
            
            # 상태 미분값 계산 (부드러운 수렴을 위한 1차 시스템)
            beta_dot = (beta_target - beta) * 50.0
            yaw_rate_dot = (yaw_rate_new - yaw_rate) * 50.0

            # TODO: 상태 업데이트 및 물리적 제한 적용 Done, yaw각도 update
            beta += beta_dot * dt
            beta = np.clip(beta, -self.beta_max, self.beta_max)
            yaw_rate += yaw_rate_dot * dt
            yaw_rate = np.clip(yaw_rate, -self.yaw_rate_max, self.yaw_rate_max)
            self._beta = beta
            self._yaw_rate = yaw_rate
            self._yaw += yaw_rate * dt
        else:
            # =========================================================
            # 다이나믹 모델 (고속): 타이어 슬립과 측력 고려
            # =========================================================
            
            # TODO: 타이어 슬립각 계산 및 제한 Done
            # 전륜 슬립각: α_f = δ - β - lf*r/v
            # 후륜 슬립각: α_r = -β + lr*r/v
            alpha_f = delta - beta - self.lf*yaw_rate/v
            alpha_r = -beta + self.lr*yaw_rate/v

            # TODO: 측력 계산 (선형 타이어 모델) Done
            # 전륜 측력: F_yf = Caf * α_f
            # 후륜 측력: F_yr = Car * α_r
            Fyf = self.Caf*alpha_f
            Fyr = self.Car*alpha_r

            # TODO: 차체 슬립각 미분값 계산 Done
            # β̇ = (Fyf + Fyr)/(m*v) - r
            beta_dot = (Fyf + Fyr)/(self.m*v) - yaw_rate

            # TODO: 요 각속도 미분값 계산 Done
            # ṙ = (lf*Fyf - lr*Fyr) / Iz 
            yaw_rate_dot = (self.lf*Fyf - self.lr*Fyr) / self.Iz

            # TODO: 상태 업데이트 및 물리적 제한 적용 Done
            beta += beta_dot * dt
            beta = np.clip(beta, -self.beta_max, self.beta_max)
            yaw_rate += yaw_rate_dot * dt
            yaw_rate = np.clip(yaw_rate, -self.yaw_rate_max, self.yaw_rate_max)
            self._beta = beta
            self._yaw_rate = yaw_rate
            self._yaw += yaw_rate * dt

        return self._yaw, self._yaw_rate

    # =================================================================
    # 위치 운동학
    # =================================================================
    def _position_rates(self):
        """차량의 위치 변화율을 계산합니다.
        
        교육 목적:
        차량의 운동학적 관계와 좌표 변환을 학습합니다:
        1. 차체 고정 좌표계와 전역 좌표계의 관계
        2. 속도 벡터의 방향과 차체 슬립각의 관계
        3. 회전 변환 행렬의 적용
        4. 무게중심점에서의 속도와 위치 관계
        
        운동학 관계:
        - 차량 헤딩 방향: ψ + β (요각 + 슬립각)
        - 전역 좌표계 속도: [v*cos(ψ+β), v*sin(ψ+β)]
        
        Returns:
            tuple: (x_dot, y_dot) 전역 좌표계에서의 위치 변화율 [m/s]
        """
        # TODO: 차량 헤딩 = 요각 + 슬립각 (무게중심 속도 방향) Done
        heading = self._yaw + self._beta
        
        # TODO: 전역 좌표계에서의 위치 변화율 계산 Done
        x_dot = self._v * np.cos(heading)
        y_dot = self._v * np.sin(heading)
        return x_dot, y_dot

    # =================================================================
    # 레거시 공개 API (내부에서 오일러 적분 사용)
    # =================================================================
    def euler_step(self, states, inputs, dt):
        """오일러 적분법을 사용하여 차량 상태를 한 스텝 전진시킵니다.
        
        교육 목적:
        통합 차량 동역학 시뮬레이션의 전체 과정을 학습합니다:
        1. 상태 벡터 기반 시뮬레이션 아키텍처
        2. 종방향과 횡방향 동역학의 순차적 계산
        3. 오일러 적분법의 구현과 수치적 안정성
        4. 상태 제약 조건과 물리적 한계 적용
        5. 출력 변수들의 계산과 후처리
        
        시뮬레이션 순서:
        1. 입력 상태를 내부 변수로 동기화
        2. 종방향 동역학 계산 (가속도)
        3. 횡방향 동역학 계산 (요각, 요각속도)
        4. 위치 운동학 계산 (x, y 위치)
        5. 출력 상태 벡터 구성
        
        Args:
            states: [x, y, psi, vx, vy, yaw_rate, ax, ay] 현재 상태 벡터
            inputs: [steer_rad, throttle] 입력 벡터
            dt: 시간 스텝 [s]
            
        Returns:
            np.ndarray: 다음 시간 스텝의 8차원 상태 벡터
        """
        steer_rad = float(inputs[0])  # 조향각 [rad]
        throttle  = float(inputs[1])  # 스로틀 [-1~1]
        dt = float(dt)                # 시간 스텝 [s]

        # 레거시 벡터에서 내부 상태 동기화
        self._sync_from_legacy_states(states)

        # ---- 1) 종방향 동역학 (오일러 적분)
        # TODO: _longitudinal_step 함수 완성
        ax = self._longitudinal_step(throttle, dt)

        # ---- 2) 횡방향 동역학 (오일러 적분, 키네마틱/다이나믹 전환)
        # TODO: _lateral_step 함수 완성
        yaw, yaw_rate = self._lateral_step(steer_rad, dt)

        # ---- 3) 위치 적분 (오일러 적분)
        # TODO: _position_rates 함수 완성
        x_dot, y_dot = self._position_rates()
        self._x += x_dot * dt
        self._y += y_dot * dt

        # ---- 출력 변환 및 계산
        ay = self._v * yaw_rate             # 횡가속도 근사값
        vy = self._v * np.sin(self._beta)   # 무게중심에서의 횡속도

        # 출력 상태 벡터 구성
        out = np.array([
            self._x,                        # x 위치 [m]
            self._y,                        # y 위치 [m]
            yaw,                            # 요각 psi [rad]
            max(0.0, self._v),              # 종속도 vx [m/s] (비음수)
            vy,                             # 횡속도 vy [m/s]
            yaw_rate,                       # 요각속도 [rad/s]
            ax,                             # 종가속도 [m/s²] (추정값)
            ay,                             # 횡가속도 [m/s²] (근사값: v*r)
        ], dtype=float)

        return out
