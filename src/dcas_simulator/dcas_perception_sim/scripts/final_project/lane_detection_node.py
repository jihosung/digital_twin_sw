# #!/usr/bin/env python
# """
# Lane Detection Node

# 이 노드는 카메라 이미지와 lane.csv 파일로부터 차선을 검출합니다.

# 입력:
# - /sensors/camera/image_raw: 카메라 이미지
# - /vehicle/state: 차량 상태

# 출력:
# - /perception/lanes: 검출된 차선 정보
# """

# import os
# import rospy
# import cv2
# import numpy as np
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from dcas_msgs.msg import VehicleState, LaneArray, Lane
# from geometry_msgs.msg import Point
# import rospkg
# import math

# from visualization_msgs.msg import Marker, MarkerArray
# from std_msgs.msg import ColorRGBA


# class LaneDetectionNode:
#     """차선 검출 노드

#     카메라 이미지와 ground truth lane 데이터를 사용하여 차선을 검출합니다.
#     """

#     def __init__(self):
#         """노드 초기화"""

#         # ROS 파라미터
#         self.detection_mode = rospy.get_param("~detection_mode", "image")  # "image" or "csv"
#         self.detection_range = rospy.get_param("~detection_range", 50.0)  # 검출 범위 (m)
#         self.target_center_lane = rospy.get_param("~target_center_lane", 1)  # 1: lane 0-1 사이, 2: lane 1-2 사이

#         # CvBridge
#         self.bridge = CvBridge()

#         # 상태 변수
#         self.current_image = None
#         self.vehicle_state = None
#         self.ground_truth_lanes = None

#         # Ground truth lane 데이터 로드
#         self._load_lane_csv()

#         # Publisher
#         self.lane_pub = rospy.Publisher(
#             "/perception/lanes",
#             LaneArray,
#             queue_size=10
#         )
#         self.marker_pub = rospy.Publisher(
#             "/perception/lanes_marker",
#             MarkerArray,
#             queue_size=10
#         )


#         # Subscribers
#         rospy.Subscriber(
#             "/sensors/camera/image_raw",
#             Image,
#             self.callback_image
#         )
#         rospy.Subscriber(
#             "/vehicle/state",
#             VehicleState,
#             self.callback_vehicle_state
#         )
#         rospy.Subscriber(
#             "/env/lanes",
#             LaneArray,
#             self.callback_ground_truth_lanes
#         )

#         # 타이머 (10Hz)
#         self.timer = rospy.Timer(rospy.Duration(0.1), self.callback_timer)

#         rospy.loginfo("Lane detection node initialized")

#     def _to_marker_array(self, lane_array_msg):
#         ma = MarkerArray()

#         for i, lane in enumerate(lane_array_msg.lanes):
#             mk = Marker()
#             mk.header = lane_array_msg.header         # frame_id="map" 유지
#             mk.ns = "detected_lanes"
#             mk.id = i
#             mk.type = Marker.LINE_STRIP
#             mk.action = Marker.ADD
#             mk.pose.orientation.w = 1.0
#             mk.scale.x = 0.2  # 선 두께 (m)

#             # 색상(차선별로 다르게)
#             c = ColorRGBA()
#             if lane.id == 0:
#                 c.r, c.g, c.b, c.a = 1.0, 0.0, 0.0, 1.0   # 빨강
#             elif lane.id == 2:
#                 c.r, c.g, c.b, c.a = 0.0, 0.0, 1.0, 1.0   # 파랑
#             else:
#                 c.r, c.g, c.b, c.a = 0.0, 1.0, 0.0, 1.0   # 초록
#             mk.color = c

#             # points 채우기 (Lane.lane_lines 가 geometry_msgs/Point 리스트라고 가정)
#             for p in lane.lane_lines:
#                 gp = Point()
#                 gp.x, gp.y, gp.z = p.x, p.y, p.z
#                 mk.points.append(gp)

#             mk.lifetime = rospy.Duration(0.2)  # 10Hz 갱신이면 짧게
#             ma.markers.append(mk)

#         return ma

#     def _load_lane_csv(self):
#         """lane.csv 파일 로드 및 중앙 차선 계산"""
#         try:
#             # ROS 패키지 경로 찾기
#             rospack = rospkg.RosPack()
#             pkg_path = rospack.get_path('dcas_perception_sim')
#             csv_path = os.path.join(pkg_path, 'maps', 'lanes.csv')

#             if not os.path.exists(csv_path):
#                 rospy.logwarn(f"Lane CSV file not found: {csv_path}")
#                 self.lane_csv_data = {}
#                 self.center_lanes = {}
#                 return

#             # CSV 파일 읽기
#             import csv
#             self.lane_csv_data = {}

#             with open(csv_path, 'r') as f:
#                 reader = csv.DictReader(f)
#                 for row in reader:
#                     lane_id = int(row['lane_id'])
#                     if lane_id not in self.lane_csv_data:
#                         self.lane_csv_data[lane_id] = {
#                             'points': [],
#                             'lane_type': int(row['lane_type'])
#                         }

#                     point = {
#                         'x': float(row['x']),
#                         'y': float(row['y']),
#                         'z': float(row['z'])
#                     }
#                     self.lane_csv_data[lane_id]['points'].append(point)

#             rospy.loginfo(f"Loaded {len(self.lane_csv_data)} lanes from CSV")

#             # 중앙 차선들 계산
#             self._compute_center_lanes()

#         except Exception as e:
#             rospy.logerr(f"Failed to load lane CSV: {e}")
#             self.lane_csv_data = {}
#             self.center_lanes = {}

#     def _compute_center_lanes(self):
#         """차선 중앙선들 계산

#         3개의 차선(lane_id 0, 1, 2)이 있을 때:
#         - center_lane 1: lane 0과 1의 중점
#         - center_lane 2: lane 1과 2의 중점
#         """
#         self.center_lanes = {}

#         # lane_id 0, 1, 2가 모두 있는지 확인
#         if 0 not in self.lane_csv_data or 1 not in self.lane_csv_data or 2 not in self.lane_csv_data:
#             rospy.logwarn("Lane ID 0, 1, or 2 not found. Cannot compute center lanes.")
#             return

#         lane0_points = self.lane_csv_data[0]['points']
#         lane1_points = self.lane_csv_data[1]['points']
#         lane2_points = self.lane_csv_data[2]['points']

#         # Center lane 1: lane 0과 1 사이
#         self.center_lanes[1] = []
#         min_len_01 = min(len(lane0_points), len(lane1_points))
#         for i in range(min_len_01):
#             p0 = lane0_points[i]
#             p1 = lane1_points[i]
#             center_point = {
#                 'x': (p0['x'] + p1['x']) / 2.0,
#                 'y': (p0['y'] + p1['y']) / 2.0,
#                 'z': (p0['z'] + p1['z']) / 2.0
#             }
#             self.center_lanes[1].append(center_point)

#         # Center lane 2: lane 1과 2 사이
#         self.center_lanes[2] = []
#         min_len_12 = min(len(lane1_points), len(lane2_points))
#         for i in range(min_len_12):
#             p1 = lane1_points[i]
#             p2 = lane2_points[i]
#             center_point = {
#                 'x': (p1['x'] + p2['x']) / 2.0,
#                 'y': (p1['y'] + p2['y']) / 2.0,
#                 'z': (p1['z'] + p2['z']) / 2.0
#             }
#             self.center_lanes[2].append(center_point)

#         rospy.loginfo(
#             f"Computed center lanes: "
#             f"center_lane_1 ({len(self.center_lanes[1])} points), "
#             f"center_lane_2 ({len(self.center_lanes[2])} points)"
#         )

#     def callback_image(self, msg):
#         """카메라 이미지 콜백"""
#         try:
#             self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         except Exception as e:
#             rospy.logerr(f"Failed to convert image: {e}")

#     def callback_vehicle_state(self, msg):
#         """차량 상태 콜백"""
#         self.vehicle_state = msg

#     def callback_ground_truth_lanes(self, msg):
#         """Ground truth 차선 콜백"""
#         self.ground_truth_lanes = msg

#     def detect_lanes_from_csv(self):
#         """CSV 데이터로부터 차선 검출

#         미리 계산된 중앙 차선(center lane)을 사용하여
#         안정적인 차선 추종이 가능하도록 합니다.
#         파라미터로 지정된 center lane(1 또는 2)을 발행합니다.
#         """
#         if self.vehicle_state is None or not self.center_lanes:
#             return None

#         # 선택된 중앙 차선 확인
#         if self.target_center_lane not in self.center_lanes:
#             rospy.logwarn_throttle(
#                 5.0,
#                 f"Target center lane {self.target_center_lane} not found. "
#                 f"Available: {list(self.center_lanes.keys())}"
#             )
#             return None

#         vehicle_x = self.vehicle_state.pose.position.x
#         vehicle_y = self.vehicle_state.pose.position.y

#         # 차량의 yaw 각도 계산
#         quat = self.vehicle_state.pose.orientation
#         siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
#         cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
#         vehicle_yaw = np.arctan2(siny_cosp, cosy_cosp)

#         detected_lanes = LaneArray()
#         detected_lanes.header.stamp = rospy.Time.now()
#         detected_lanes.header.frame_id = "map"

#         # 선택된 중앙 차선을 단일 Lane으로 발행
#         lane_msg = Lane()
#         lane_msg.id = 900 + self.target_center_lane  # 901 또는 902
#         lane_msg.lane_type = 1  # solid line
#         lane_msg.header.stamp = rospy.Time.now()
#         lane_msg.header.frame_id = "map"

#         # 차량 전방의 중앙 차선 포인트만 선택
#         selected_center_lane = self.center_lanes[self.target_center_lane]
#         for point_data in selected_center_lane:
#             dx = point_data['x'] - vehicle_x
#             dy = point_data['y'] - vehicle_y
#             distance = np.sqrt(dx*dx + dy*dy)

#             # 차량 좌표계로 변환
#             local_x = dx * np.cos(vehicle_yaw) + dy * np.sin(vehicle_yaw)

#             # 전방 포인트만 포함 (약간의 후방 포함)
#             if distance < self.detection_range and local_x > -5.0:
#                 point = Point()
#                 point.x = point_data['x']
#                 point.y = point_data['y']
#                 point.z = point_data['z']
#                 lane_msg.lane_lines.append(point)

#         # 포인트가 있으면 추가
#         if len(lane_msg.lane_lines) > 0:
#             detected_lanes.lanes.append(lane_msg)

#         return detected_lanes

#     def detect_lanes_from_image(self):
#         """이미지로부터 차선 검출 (IPM + Sliding Window + 직접 좌표 변환)"""
#         if self.current_image is None or self.vehicle_state is None:
#             return None

#         height, width = self.current_image.shape[:2]

#         # ---------------------------------------------------------
#         # 1. BEV 변환 행렬 계산 (Pixel -> Pixel)
#         # ---------------------------------------------------------
#         # Source: 사다리꼴 (카메라 이미지)
#         src_pts = np.float32([
#             [width * 0.40, height * 0.60],  # TL
#             [width * 0.60, height * 0.60],  # TR
#             [width * 0.10, height * 0.95],  # BL
#             [width * 0.90, height * 0.95]   # BR
#         ])

#         # Destination: 직사각형 (BEV 이미지)
#         dst_pts = np.float32([
#             [200, 0],       # TL
#             [width-200, 0], # TR
#             [200, height],  # BL
#             [width-200, height] # BR
#         ])

#         # M: Image -> BEV 변환 행렬
#         M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
#         # Minv: BEV -> Image 변환 행렬 (시각화 및 복원용)
#         Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)

#         # ---------------------------------------------------------
#         # 2. 이미지 전처리 (색상 필터링)
#         # ---------------------------------------------------------
#         blur = cv2.GaussianBlur(self.current_image, (5, 5), 0)
#         hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)

#         lower_white = np.array([0, 200, 0])
#         upper_white = np.array([255, 255, 255])
#         mask_white = cv2.inRange(hls, lower_white, upper_white)

#         lower_yellow = np.array([15, 100, 100])
#         upper_yellow = np.array([35, 255, 255])
#         mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)
        
#         mask_combined = cv2.bitwise_or(mask_white, mask_yellow)

#         # ---------------------------------------------------------
#         # 3. BEV 변환 (Warp)
#         # ---------------------------------------------------------
#         warped = cv2.warpPerspective(mask_combined, M, (width, height))

#         # ---------------------------------------------------------
#         # 4. Sliding Window (차선 픽셀 찾기)
#         # ---------------------------------------------------------
#         # (이전 코드와 동일한 로직으로 픽셀 추출)
#         histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)
#         midpoint = int(histogram.shape[0] / 2)
        
#         leftx_base = np.argmax(histogram[:midpoint])
#         rightx_base = np.argmax(histogram[midpoint:]) + midpoint

#         nwindows = 9
#         window_height = int(height / nwindows)
#         margin = 80
#         minpix = 50

#         nonzero = warped.nonzero()
#         nonzeroy = np.array(nonzero[0])
#         nonzerox = np.array(nonzero[1])

#         left_lane_inds = []
#         right_lane_inds = []

#         leftx_current = leftx_base
#         rightx_current = rightx_base

#         for window in range(nwindows):
#             win_y_low = height - (window + 1) * window_height
#             win_y_high = height - window * window_height
#             win_xleft_low = leftx_current - margin
#             win_xleft_high = leftx_current + margin
#             win_xright_low = rightx_current - margin
#             win_xright_high = rightx_current + margin

#             good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
#                               (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
#             good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
#                                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

#             left_lane_inds.append(good_left_inds)
#             right_lane_inds.append(good_right_inds)

#             if len(good_left_inds) > minpix:
#                 leftx_current = int(np.mean(nonzerox[good_left_inds]))
#             if len(good_right_inds) > minpix:
#                 rightx_current = int(np.mean(nonzerox[good_right_inds]))

#         left_lane_inds = np.concatenate(left_lane_inds)
#         right_lane_inds = np.concatenate(right_lane_inds)

#         # ---------------------------------------------------------
#         # 5. 좌표 변환 및 메시지 생성 (수정된 부분)
#         # ---------------------------------------------------------
#         detected_lanes = LaneArray()
#         detected_lanes.header.stamp = rospy.Time.now()
#         detected_lanes.header.frame_id = "map"

#         # 여기서 Minv를 이용해 [BEV -> Image -> Vehicle -> Map] 과정을 수행합니다.
#         # "좌표 변환이 두 번 된 것 같다"는 느낌을 해소하기 위해
#         # 단계를 명확히 분리한 함수를 호출합니다.
        
#         if len(left_lane_inds) > 0:
#             lx = nonzerox[left_lane_inds]
#             ly = nonzeroy[left_lane_inds]
#             # lane_id 0: 좌측 차선
#             lane_msg = self._process_lane_pixels(lx, ly, Minv, lane_id=0)
#             if lane_msg: detected_lanes.lanes.append(lane_msg)

#         if len(right_lane_inds) > 0:
#             rx = nonzerox[right_lane_inds]
#             ry = nonzeroy[right_lane_inds]
#             # lane_id 2: 우측 차선
#             lane_msg = self._process_lane_pixels(rx, ry, Minv, lane_id=2)
#             if lane_msg: detected_lanes.lanes.append(lane_msg)

#         # 디버깅: BEV 이미지 시각화
#         out_img = np.dstack((warped, warped, warped)) * 255
#         out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
#         out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]
#         cv2.imshow("Lane_Detection_BEV", out_img)
#         cv2.waitKey(1)

#         return detected_lanes

#     def _process_lane_pixels(self, x_vals, y_vals, Minv, lane_id):
#         """
#         BEV 픽셀들로부터 차선을 피팅하고 Map 좌표로 변환하는 함수
#         프로세스: BEV 픽셀 -> (Minv) -> 원본 이미지 픽셀 -> (pixel_to_vehicle) -> 차량 3D 좌표 -> (Rotation) -> 지도 좌표
#         """
#         if len(x_vals) < 10: return None

#         # 1. BEV 상에서 2차 함수 피팅 (Curve Fitting)
#         # y값(세로)에 따른 x값(가로)의 변화를 모델링
#         fit = np.polyfit(y_vals, x_vals, 2)
#         poly = np.poly1d(fit)

#         # 2. 샘플링 (이미지 아래쪽 ~ 중간 정도까지)
#         # 너무 멀리 있는 점(y=0 근처)은 오차가 크므로 y=height*0.2 정도까지만 사용
#         plot_y = np.linspace(480, 100, 10) 
#         plot_x = poly(plot_y)

#         lane_msg = Lane()
#         lane_msg.id = lane_id
#         lane_msg.lane_type = 1
#         lane_msg.header.stamp = rospy.Time.now()
#         lane_msg.header.frame_id = "map"

#         # 3. 좌표 변환 체인 (Chain of Transformations)
#         # OpenCV 함수를 사용해 행렬 곱셈을 더 깔끔하게 처리
        
#         # (1) BEV Pixel -> Original Image Pixel
#         # perspectiveTransform은 (N, 1, 2) 형태의 입력을 받습니다.
#         bev_points = np.array([np.transpose(np.vstack([plot_x, plot_y]))], dtype=np.float32)
#         image_points = cv2.perspectiveTransform(bev_points, Minv) # 결과: (1, N, 2)
        
#         image_points = image_points[0] # (N, 2)

#         for (u, v) in image_points:
#             # (2) Original Image Pixel -> Vehicle Coordinate (Local Meter)
#             # 이전에 만든 pixel_to_vehicle 함수 사용 (Ray Casting)
#             vehicle_pt = self.pixel_to_vehicle(u, v)
            
#             if vehicle_pt is None:
#                 continue

#             # (3) Vehicle Coordinate -> Map Coordinate (Global Meter)
#             # 차량의 현재 위치와 방향을 더해줌
#             map_pt = self._vehicle_to_map_transform(vehicle_pt)
            
#             lane_msg.lane_lines.append(map_pt)
            
#         return lane_msg

#     def _vehicle_to_map_transform(self, vehicle_pt):
#         """차량 기준 좌표를 지도 좌표로 변환"""
#         vx, vy = vehicle_pt.x, vehicle_pt.y
        
#         vehicle_x = self.vehicle_state.pose.position.x
#         vehicle_y = self.vehicle_state.pose.position.y
        
#         quat = self.vehicle_state.pose.orientation
#         siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
#         cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
#         yaw = np.arctan2(siny_cosp, cosy_cosp)
        
#         # 회전 + 이동
#         map_x = vehicle_x + (vx * np.cos(yaw) - vy * np.sin(yaw))
#         map_y = vehicle_y + (vx * np.sin(yaw) + vy * np.cos(yaw))
        
#         point = Point()
#         point.x = map_x
#         point.y = map_y
#         point.z = 0.0
#         return point

#     def _fit_and_transform_bev(self, x_vals, y_vals, M, lane_id):
#         """
#         BEV 상의 픽셀들로부터 2차 함수 피팅 후 Map 좌표로 변환
        
#         중요: BEV 픽셀 좌표 -> 원본 이미지 픽셀 -> Map 좌표 순으로 변환하거나
#               BEV 픽셀 -> 실제 거리(Meter) 비율을 알면 바로 변환 가능합니다.
#               여기서는 정확성을 위해 [BEV -> 원본 -> Map] 방식을 사용합니다.
#         """
#         if len(x_vals) < 10: return None

#         # 2차 함수 피팅 (x = ay^2 + by + c) - 곡선 도로 대응
#         fit = np.polyfit(y_vals, x_vals, 2)
#         poly = np.poly1d(fit)

#         # 샘플링 (BEV 이미지의 y축 전체에 대해)
#         plot_y = np.linspace(0, 480 - 1, 10) # 10개 점만 추출
#         plot_x = poly(plot_y)

#         lane_msg = Lane()
#         lane_msg.id = lane_id
#         lane_msg.lane_type = 1
#         lane_msg.header.stamp = rospy.Time.now()
#         lane_msg.header.frame_id = "map"

#         # 역변환 행렬 (BEV -> Original Image)
#         Minv = np.linalg.inv(M)

#         for px, py in zip(plot_x, plot_y):
#             # 1. BEV Pixel -> Original Image Pixel
#             # 행렬 곱: P_orig = Minv @ P_bev
#             vec = np.array([px, py, 1.0])
#             orig_vec = np.dot(Minv, vec)
            
#             # 동차 좌표 정규화 (scale로 나누기)
#             if orig_vec[2] != 0:
#                 u = orig_vec[0] / orig_vec[2]
#                 v = orig_vec[1] / orig_vec[2]
#             else:
#                 continue

#             # 2. Original Image Pixel -> Map Coordinate
#             # (이전에 만든 pixel_to_map 함수 재사용!)
#             map_point = self._pixel_to_map(u, v)
            
#             if map_point:
#                 lane_msg.lane_lines.append(map_point)
        
#         return lane_msg

#     def pixel_to_vehicle(self, u, v):
#         """
#         이미지 픽셀 좌표(u, v)를 차량 기준 좌표(x, y)로 변환
        
#         가정:
#         1. 핀홀 카메라 모델 (왜곡 무시 혹은 보정된 이미지라 가정)
#         2. 지면은 평평하다 (Flat Ground Assumption, Z_vehicle = 0)
        
#         Args:
#             u: 이미지 x좌표 (가로)
#             v: 이미지 y좌표 (세로)
            
#         Returns:
#             point: geometry_msgs/Point (x, y, z=0) 차량 기준 좌표
#                    변환 불가능한 경우(하늘 등) None 반환
#         """
#         # 1. 카메라 파라미터 가져오기 (sensor_camera_node.py와 동일한 값 사용)
#         # 실제로는 config나 param server에서 가져와야 합니다.
#         f = 320.0  # focal_length
#         cx = 320.0 # image_width / 2
#         cy = 240.0 # image_height / 2
        
#         # 카메라 오프셋 및 회전 각도
#         offset_x = 1.1
#         offset_y = 0.0
#         offset_z = 1.4
        
#         roll_deg = 0.0
#         pitch_deg = -5.0
#         yaw_deg = 0.0

#         # 2. 픽셀 좌표 -> 정규화된 이미지 평면 좌표 (Normalized Image Plane)
#         # u = f * (x / z) + cx  => x/z = (u - cx) / f
#         # v = f * (y / z) + cy  => y/z = (v - cy) / f
        
#         if f <= 0: return None
        
#         # 카메라 좌표계 기준 (Right, Down, Forward)의 방향 벡터
#         # cam_z_forward = 1 로 가정했을 때의 x, y 값
#         n_x = (u - cx) / f  # Right
#         n_y = (v - cy) / f  # Down
#         n_z = 1.0           # Forward
        
#         # 3. 카메라 좌표계 -> 차량 좌표계 방향 벡터 변환
#         # sensor_camera_node.py의 world_to_camera_frame 역연산
        
#         # 코드상의 매핑 관계:
#         # camera_x_right   = -vehicle_y
#         # camera_y_down    = -vehicle_z
#         # camera_z_forward =  vehicle_x
        
#         # 이를 역으로 차량 좌표계 기준의 방향 벡터(ray)로 변환하면:
#         ray_x = n_z   # Forward -> Vehicle X
#         ray_y = -n_x  # Right   -> -Vehicle Y (Left)
#         ray_z = -n_y  # Down    -> -Vehicle Z (Up)
        
#         ray_vec = np.array([ray_x, ray_y, ray_z])
        
#         # 4. 카메라의 회전 행렬 적용 (Vehicle Frame 기준의 Ray 방향 계산)
#         # R_camera 행렬 생성 (sensor_camera_node.py 로직 그대로 사용)
#         cam_r_rad = math.radians(roll_deg)
#         cam_p_rad = math.radians(pitch_deg)
#         cam_y_rad = math.radians(yaw_deg)
        
#         rot_x = np.array([
#             [1, 0, 0],
#             [0, math.cos(cam_r_rad), -math.sin(cam_r_rad)],
#             [0, math.sin(cam_r_rad), math.cos(cam_r_rad)]
#         ])
#         rot_y = np.array([
#             [math.cos(cam_p_rad), 0, math.sin(cam_p_rad)],
#             [0, 1, 0],
#             [-math.sin(cam_p_rad), 0, math.cos(cam_p_rad)]
#         ])
#         rot_z = np.array([
#             [math.cos(cam_y_rad), -math.sin(cam_y_rad), 0],
#             [math.sin(cam_y_rad), math.cos(cam_y_rad), 0],
#             [0, 0, 1]
#         ])
        
#         # 회전 행렬 합성 R = Rz * Ry * Rx
#         R_camera = rot_z @ rot_y @ rot_x
        
#         # 회전된 레이 벡터 (차량 좌표계 기준의 실제 시선 방향)
#         rotated_ray = R_camera @ ray_vec
        
#         # 5. 지면 투영 (Flat Ground Assumption)
#         # 레이 방정식: P = Camera_Origin + t * Ray_Direction
#         # 우리는 P.z = 0 (지면)이 되는 t를 찾아야 함
#         # 0 = offset_z + t * rotated_ray.z
#         # t = -offset_z / rotated_ray.z
        
#         # 수평선 위쪽(하늘)을 클릭한 경우 처리
#         if rotated_ray[2] >= -1e-6: 
#             return None
            
#         t = -offset_z / rotated_ray[2]
        
#         # 차량 기준 좌표 계산 (z는 당연히 0)
#         vehicle_x = offset_x + t * rotated_ray[0]
#         vehicle_y = offset_y + t * rotated_ray[1]
        
#         # 전방에 있는 점만 유효하다고 가정 (선택 사항)
#         if vehicle_x < 0:
#             return None

#         point = Point()
#         point.x = vehicle_x
#         point.y = vehicle_y
#         point.z = 0.0
        
#         return point

#     def _draw_fitted_line(self, img, points, height, color):
#         """디버깅용: 피팅된 차선을 이미지에 그리는 헬퍼 함수"""
#         points = np.array(points)
#         x = points[:, 0]
#         y = points[:, 1]
        
#         if len(x) < 2: return

#         # 1차 함수 피팅 (x = ay + b)
#         fit = np.polyfit(y, x, 1)
#         poly = np.poly1d(fit)

#         # 그리기용 포인트 생성 (이미지 하단 ~ ROI 상단)
#         plot_y = np.linspace(height * 0.6, height, 10)
#         plot_x = poly(plot_y)
        
#         # 정수형 좌표로 변환하여 그리기
#         pts = np.array([np.transpose(np.vstack([plot_x, plot_y]))], np.int32)
#         cv2.polylines(img, pts, False, color, thickness=3)

#     def _create_lane_msg_from_points(self, points, img_height, lane_id):
#         """이미지 포인트들로부터 Lane 메시지 생성"""
#         if not points:
#             return None
            
#         # 다항식 피팅 (1차 함수: x = ay + b)
#         # y를 기준으로 x를 구하는 것이 수직선에 가까운 차선에서 더 안정적임
#         points = np.array(points)
#         x = points[:, 0]
#         y = points[:, 1]
        
#         if len(x) < 2: return None
        
#         fit = np.polyfit(y, x, 1) # x = fit[0]*y + fit[1]
#         poly = np.poly1d(fit)

#         # 샘플링 (이미지 하단에서 ROI 상단까지)
#         y_samples = np.linspace(img_height, img_height * 0.6, 10)
#         x_samples = poly(y_samples)

#         lane_msg = Lane()
#         lane_msg.id = lane_id
#         lane_msg.lane_type = 1 # solid
#         lane_msg.header.stamp = rospy.Time.now()
#         lane_msg.header.frame_id = "map"

#         # 좌표 변환 (Pixel -> Vehicle -> Map)
#         for vx, vy in zip(x_samples, y_samples):
#             map_point = self._pixel_to_map(vx, vy)
#             if map_point:
#                 lane_msg.lane_lines.append(map_point)
                
#         return lane_msg

#     def _pixel_to_map(self, u, v):
#         """픽셀 좌표 -> 차량 좌표 -> 지도 좌표 변환"""
        
#         # 1. 픽셀 -> 차량 좌표 (위에서 만든 함수 사용)
#         vehicle_point = self.pixel_to_vehicle(u, v)
#         if vehicle_point is None:
#             return None
            
#         local_x = vehicle_point.x
#         local_y = vehicle_point.y
        
#         # 2. 차량 좌표 -> 지도(Map) 좌표
#         # (기존 코드와 동일)
#         vehicle_x = self.vehicle_state.pose.position.x
#         vehicle_y = self.vehicle_state.pose.position.y
        
#         quat = self.vehicle_state.pose.orientation
#         siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
#         cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
#         yaw = np.arctan2(siny_cosp, cosy_cosp)
        
#         map_x = vehicle_x + (local_x * np.cos(yaw) - local_y * np.sin(yaw))
#         map_y = vehicle_y + (local_x * np.sin(yaw) + local_y * np.cos(yaw))
        
#         point = Point()
#         point.x = map_x
#         point.y = map_y
#         point.z = 0.0
        
#         return point

#     def callback_timer(self, event):
#         """타이머 콜백 - 주기적으로 차선 검출 수행"""

#         if self.vehicle_state is None:
#             return

#         # 검출 모드에 따라 차선 검출
#         if self.detection_mode == "csv":
#             detected_lanes = self.detect_lanes_from_csv()
#         else:  # "image"
#             detected_lanes = self.detect_lanes_from_image()

#         # 검출된 차선 발행
#         if detected_lanes is not None:
#             self.lane_pub.publish(detected_lanes)

#             # RViz용 marker publish
#             marker_array = self._to_marker_array(detected_lanes)
#             self.marker_pub.publish(marker_array)       


# def main():
#     """메인 함수"""
#     rospy.init_node("lane_detection_node")

#     try:
#         node = LaneDetectionNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass


# if __name__ == "__main__":
#     main()






#!/usr/bin/env python
"""
Lane Detection Node

이 노드는 카메라 이미지와 lane.csv 파일로부터 차선을 검출합니다.

입력:
- /sensors/camera/image_raw: 카메라 이미지
- /vehicle/state: 차량 상태

출력:
- /perception/lanes: 검출된 차선 정보
"""

import os
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dcas_msgs.msg import VehicleState, LaneArray, Lane
from geometry_msgs.msg import Point
import rospkg
import math

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class LaneDetectionNode:
    """차선 검출 노드

    카메라 이미지와 ground truth lane 데이터를 사용하여 차선을 검출합니다.
    """

    def __init__(self):
        """노드 초기화"""

        # ROS 파라미터
        self.detection_mode = rospy.get_param("~detection_mode", "image")  # "image" or "csv"
        self.detection_range = rospy.get_param("~detection_range", 50.0)  # 검출 범위 (m)
        self.target_center_lane = rospy.get_param("~target_center_lane", 1)  # 1: lane 0-1 사이, 2: lane 1-2 사이

        # Camera parameters (should match sensor_camera_node.py)
        self._load_camera_params()

        # CvBridge
        self.bridge = CvBridge()

        # 상태 변수
        self.current_image = None
        self.vehicle_state = None
        self.ground_truth_lanes = None

        # Ground truth lane 데이터 로드
        self._load_lane_csv()

        # Publisher
        self.lane_pub = rospy.Publisher(
            "/perception/lanes",
            LaneArray,
            queue_size=10
        )
        self.marker_pub = rospy.Publisher(
            "/perception/lanes_marker",
            MarkerArray,
            queue_size=10
        )


        # Subscribers
        rospy.Subscriber(
            "/sensors/camera/image_raw",
            Image,
            self.callback_image
        )
        rospy.Subscriber(
            "/vehicle/state",
            VehicleState,
            self.callback_vehicle_state
        )
        rospy.Subscriber(
            "/env/lanes",
            LaneArray,
            self.callback_ground_truth_lanes
        )

        # 타이머 (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.callback_timer)

        rospy.loginfo("Lane detection node initialized")

    def _load_camera_params(self):
        """Load camera intrinsics/extrinsics from the same ROS params used by sensor_camera_node.py."""
        # Intrinsics
        self.cam_f = float(rospy.get_param("/sensor_camera_node/focal_length", 320.0))
        self.cam_width = int(rospy.get_param("/sensor_camera_node/width", 640))
        self.cam_height = int(rospy.get_param("/sensor_camera_node/height", 480))
        # Allow explicit cx/cy override; otherwise use image center
        self.cam_cx = float(rospy.get_param("/sensor_camera_node/cx", self.cam_width / 2.0))
        self.cam_cy = float(rospy.get_param("/sensor_camera_node/cy", self.cam_height / 2.0))

        # Extrinsics (vehicle frame)
        self.cam_offset_x = float(rospy.get_param("/sensor_camera_node/offset_x", 1.1))
        self.cam_offset_y = float(rospy.get_param("/sensor_camera_node/offset_y", 0.0))
        self.cam_offset_z = float(rospy.get_param("/sensor_camera_node/offset_z", 1.4))

        self.cam_roll_deg = float(rospy.get_param("/sensor_camera_node/roll_deg", 0.0))
        self.cam_pitch_deg = float(rospy.get_param("/sensor_camera_node/pitch_deg", -5.0))
        self.cam_yaw_deg = float(rospy.get_param("/sensor_camera_node/yaw_deg", 0.0))

        # Precompute R_vehicle_to_camera (same convention as sensor_camera_node.py)
        r = math.radians(self.cam_roll_deg)
        p = math.radians(self.cam_pitch_deg)
        y = math.radians(self.cam_yaw_deg)

        rot_x = np.array([[1, 0, 0],
                          [0, math.cos(r), -math.sin(r)],
                          [0, math.sin(r),  math.cos(r)]], dtype=np.float64)
        rot_y = np.array([[ math.cos(p), 0, math.sin(p)],
                          [0,           1, 0],
                          [-math.sin(p), 0, math.cos(p)]], dtype=np.float64)
        rot_z = np.array([[math.cos(y), -math.sin(y), 0],
                          [math.sin(y),  math.cos(y), 0],
                          [0,            0,           1]], dtype=np.float64)

        self.R_vehicle_to_camera = rot_z @ rot_y @ rot_x
        self.R_camera_to_vehicle = self.R_vehicle_to_camera.T
    def _to_marker_array(self, lane_array_msg):
        ma = MarkerArray()

        for i, lane in enumerate(lane_array_msg.lanes):
            mk = Marker()
            mk.header = lane_array_msg.header         # frame_id="map" 유지
            mk.ns = "detected_lanes"
            mk.id = i
            mk.type = Marker.LINE_STRIP
            mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0
            mk.scale.x = 0.2  # 선 두께 (m)

            # 색상(차선별로 다르게)
            c = ColorRGBA()
            if lane.id == 0:
                c.r, c.g, c.b, c.a = 1.0, 0.0, 0.0, 1.0   # 빨강
            elif lane.id == 2:
                c.r, c.g, c.b, c.a = 0.0, 0.0, 1.0, 1.0   # 파랑
            else:
                c.r, c.g, c.b, c.a = 0.0, 1.0, 0.0, 1.0   # 초록
            mk.color = c

            # points 채우기 (Lane.lane_lines 가 geometry_msgs/Point 리스트라고 가정)
            for p in lane.lane_lines:
                gp = Point()
                gp.x, gp.y, gp.z = p.x, p.y, p.z
                mk.points.append(gp)

            mk.lifetime = rospy.Duration(0.2)  # 10Hz 갱신이면 짧게
            ma.markers.append(mk)

        return ma

    def _load_lane_csv(self):
        """lane.csv 파일 로드 및 중앙 차선 계산"""
        try:
            # ROS 패키지 경로 찾기
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('dcas_perception_sim')
            csv_path = os.path.join(pkg_path, 'maps', 'lanes.csv')

            if not os.path.exists(csv_path):
                rospy.logwarn(f"Lane CSV file not found: {csv_path}")
                self.lane_csv_data = {}
                self.center_lanes = {}
                return

            # CSV 파일 읽기
            import csv
            self.lane_csv_data = {}

            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lane_id = int(row['lane_id'])
                    if lane_id not in self.lane_csv_data:
                        self.lane_csv_data[lane_id] = {
                            'points': [],
                            'lane_type': int(row['lane_type'])
                        }

                    point = {
                        'x': float(row['x']),
                        'y': float(row['y']),
                        'z': float(row['z'])
                    }
                    self.lane_csv_data[lane_id]['points'].append(point)

            rospy.loginfo(f"Loaded {len(self.lane_csv_data)} lanes from CSV")

            # 중앙 차선들 계산
            self._compute_center_lanes()

        except Exception as e:
            rospy.logerr(f"Failed to load lane CSV: {e}")
            self.lane_csv_data = {}
            self.center_lanes = {}

    def _compute_center_lanes(self):
        """차선 중앙선들 계산

        3개의 차선(lane_id 0, 1, 2)이 있을 때:
        - center_lane 1: lane 0과 1의 중점
        - center_lane 2: lane 1과 2의 중점
        """
        self.center_lanes = {}

        # lane_id 0, 1, 2가 모두 있는지 확인
        if 0 not in self.lane_csv_data or 1 not in self.lane_csv_data or 2 not in self.lane_csv_data:
            rospy.logwarn("Lane ID 0, 1, or 2 not found. Cannot compute center lanes.")
            return

        lane0_points = self.lane_csv_data[0]['points']
        lane1_points = self.lane_csv_data[1]['points']
        lane2_points = self.lane_csv_data[2]['points']

        # Center lane 1: lane 0과 1 사이
        self.center_lanes[1] = []
        min_len_01 = min(len(lane0_points), len(lane1_points))
        for i in range(min_len_01):
            p0 = lane0_points[i]
            p1 = lane1_points[i]
            center_point = {
                'x': (p0['x'] + p1['x']) / 2.0,
                'y': (p0['y'] + p1['y']) / 2.0,
                'z': (p0['z'] + p1['z']) / 2.0
            }
            self.center_lanes[1].append(center_point)

        # Center lane 2: lane 1과 2 사이
        self.center_lanes[2] = []
        min_len_12 = min(len(lane1_points), len(lane2_points))
        for i in range(min_len_12):
            p1 = lane1_points[i]
            p2 = lane2_points[i]
            center_point = {
                'x': (p1['x'] + p2['x']) / 2.0,
                'y': (p1['y'] + p2['y']) / 2.0,
                'z': (p1['z'] + p2['z']) / 2.0
            }
            self.center_lanes[2].append(center_point)

        rospy.loginfo(
            f"Computed center lanes: "
            f"center_lane_1 ({len(self.center_lanes[1])} points), "
            f"center_lane_2 ({len(self.center_lanes[2])} points)"
        )

    def callback_image(self, msg):
        """카메라 이미지 콜백"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def callback_vehicle_state(self, msg):
        """차량 상태 콜백"""
        self.vehicle_state = msg

    def callback_ground_truth_lanes(self, msg):
        """Ground truth 차선 콜백"""
        self.ground_truth_lanes = msg

    def detect_lanes_from_csv(self):
        """CSV 데이터로부터 차선 검출

        미리 계산된 중앙 차선(center lane)을 사용하여
        안정적인 차선 추종이 가능하도록 합니다.
        파라미터로 지정된 center lane(1 또는 2)을 발행합니다.
        """
        if self.vehicle_state is None or not self.center_lanes:
            return None

        # 선택된 중앙 차선 확인
        if self.target_center_lane not in self.center_lanes:
            rospy.logwarn_throttle(
                5.0,
                f"Target center lane {self.target_center_lane} not found. "
                f"Available: {list(self.center_lanes.keys())}"
            )
            return None

        vehicle_x = self.vehicle_state.pose.position.x
        vehicle_y = self.vehicle_state.pose.position.y

        # 차량의 yaw 각도 계산
        quat = self.vehicle_state.pose.orientation
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        vehicle_yaw = np.arctan2(siny_cosp, cosy_cosp)

        detected_lanes = LaneArray()
        detected_lanes.header.stamp = rospy.Time.now()
        detected_lanes.header.frame_id = "map"

        # 선택된 중앙 차선을 단일 Lane으로 발행
        lane_msg = Lane()
        lane_msg.id = 900 + self.target_center_lane  # 901 또는 902
        lane_msg.lane_type = 1  # solid line
        lane_msg.header.stamp = rospy.Time.now()
        lane_msg.header.frame_id = "map"

        # 차량 전방의 중앙 차선 포인트만 선택
        selected_center_lane = self.center_lanes[self.target_center_lane]
        for point_data in selected_center_lane:
            dx = point_data['x'] - vehicle_x
            dy = point_data['y'] - vehicle_y
            distance = np.sqrt(dx*dx + dy*dy)

            # 차량 좌표계로 변환
            local_x = dx * np.cos(vehicle_yaw) + dy * np.sin(vehicle_yaw)

            # 전방 포인트만 포함 (약간의 후방 포함)
            if distance < self.detection_range and local_x > -5.0:
                point = Point()
                point.x = point_data['x']
                point.y = point_data['y']
                point.z = point_data['z']
                lane_msg.lane_lines.append(point)

        # 포인트가 있으면 추가
        if len(lane_msg.lane_lines) > 0:
            detected_lanes.lanes.append(lane_msg)

        # debug
        self.debug_print_lanes(detected_lanes)

        return detected_lanes

    def detect_lanes_from_image(self):
        """이미지로부터 차선 검출 (IPM + Sliding Window + 직접 좌표 변환) - 디버깅 강화 버전"""
        if self.current_image is None or self.vehicle_state is None:
            return None

        height, width = self.current_image.shape[:2]

        # ---------------------------------------------------------
        # 1. BEV 변환 행렬 계산 (Pixel -> Pixel)
        # ---------------------------------------------------------
        # Source: 사다리꼴 (카메라 이미지)
        src_pts = np.float32([
            [width * 0.40, height * 0.60],  # TL
            [width * 0.60, height * 0.60],  # TR
            [width * 0.10, height * 0.95],  # BL
            [width * 0.90, height * 0.95]   # BR
        ])

        # Destination: 직사각형 (BEV 이미지)
        dst_pts = np.float32([
            [200, 0],       # TL
            [width-200, 0], # TR
            [200, height],  # BL
            [width-200, height] # BR
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)

        # [DEBUG 1] 원본 이미지에 ROI 영역 그리기
        debug_src = self.current_image.copy()
        pts = src_pts.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(debug_src, [pts], True, (0, 0, 255), 3) # 빨간색 다각형
        cv2.imshow("1. Original with ROI", debug_src)

        # ---------------------------------------------------------
        # 2. 이미지 전처리 (색상 필터링)
        # ---------------------------------------------------------
        blur = cv2.GaussianBlur(self.current_image, (5, 5), 0)
        hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)

        lower_white = np.array([0, 200, 0])
        upper_white = np.array([255, 255, 255])
        mask_white = cv2.inRange(hls, lower_white, upper_white)

        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)
        
        mask_combined = cv2.bitwise_or(mask_white, mask_yellow)

        # [DEBUG 2] 색상 필터링 결과 확인
        cv2.imshow("2. Color Mask", mask_combined)

        # ---------------------------------------------------------
        # 3. BEV 변환 (Warp)
        # ---------------------------------------------------------
        warped = cv2.warpPerspective(mask_combined, M, (width, height))

        # [DEBUG 3] BEV 변환 결과 확인
        cv2.imshow("3. BEV Warped", warped)

        # ---------------------------------------------------------
        # 4. Sliding Window (차선 픽셀 찾기)
        # ---------------------------------------------------------
        histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        
        # Sliding Window 시각화를 위해 미리 출력 이미지 생성 (Color 변환)
        out_img = np.dstack((warped, warped, warped)) * 255

        def find_nearest_lane_base(hist_slice, offset_x):
            threshold = 50 
            potential_peaks = []
            window = 50 
            for x in range(0, len(hist_slice) - window, window):
                sub_hist = hist_slice[x : x+window]
                if np.max(sub_hist) > threshold:
                    peak_loc = x + np.argmax(sub_hist)
                    potential_peaks.append(peak_loc)
            
            if not potential_peaks:
                return np.argmax(hist_slice) + offset_x 
                
            absolute_peaks = np.array(potential_peaks) + offset_x
            distances = np.abs(absolute_peaks - midpoint)
            nearest_idx = np.argmin(distances)
            return absolute_peaks[nearest_idx]

        leftx_base = find_nearest_lane_base(histogram[:midpoint], 0)
        rightx_base = find_nearest_lane_base(histogram[midpoint:], midpoint)

        nwindows = 9
        window_height = int(height / nwindows)
        margin = 80
        minpix = 50

        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = []
        right_lane_inds = []

        leftx_current = leftx_base
        rightx_current = rightx_base

        for window in range(nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # [DEBUG 4-1] Sliding Window 박스 그리기 (녹색)
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # [DEBUG 4-2] 검출된 픽셀 색칠하기
        if len(left_lane_inds) > 0:
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255] # Red for Left
        if len(right_lane_inds) > 0:
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0] # Blue for Right

        # [DEBUG 4-3] 최종 결과 띄우기
        cv2.imshow("4. Sliding Windows & Lane Pixels", out_img)
        cv2.waitKey(1) # 화면 갱신을 위해 필수

        # ---------------------------------------------------------
        # 5. 좌표 변환 및 메시지 생성
        # ---------------------------------------------------------
        detected_lanes = LaneArray()
        detected_lanes.header.stamp = rospy.Time.now()
        detected_lanes.header.frame_id = "map"

        if len(left_lane_inds) > 0:
            lx = nonzerox[left_lane_inds]
            ly = nonzeroy[left_lane_inds]
            lane_msg = self._process_lane_pixels(lx, ly, Minv, lane_id=0)
            # if lane_msg: detected_lanes.lanes.append(lane_msg)

        if len(right_lane_inds) > 0:
            rx = nonzerox[right_lane_inds]
            ry = nonzeroy[right_lane_inds]
            lane_msg = self._process_lane_pixels(rx, ry, Minv, lane_id=2)
            # if lane_msg: detected_lanes.lanes.append(lane_msg)

        if len(left_lane_inds) * len(right_lane_inds) > 0:
            lx = nonzerox[left_lane_inds]
            ly = nonzeroy[left_lane_inds]
            rx = nonzerox[right_lane_inds]
            ry = nonzeroy[right_lane_inds]
            cx, cy = self.compute_centerline_from_lr(lx, ly, rx, ry)
            lane_msg = self._process_lane_pixels(cx, cy, Minv, lane_id=1)
            if lane_msg: detected_lanes.lanes.append(lane_msg)

        return detected_lanes
    
    def compute_centerline_from_lr(self, lx, ly, rx, ry, y_min=None, y_max=None, deg=3, n_samples=40):
        # 최소 점수 체크
        if lx is None or ly is None or rx is None or ry is None:
            return None
        if len(lx) < deg + 1 or len(rx) < deg + 1:
            return None

        # 1) x = f(y)로 피팅 (BEV에서 안정적)
        # cL, cR: x = a*y^deg + ... 형태 계수
        try:
            cL = np.polyfit(ly, lx, deg)
            cR = np.polyfit(ry, rx, deg)
        except Exception:
            return None

        # 2) y 공통 구간 선택 (둘 다 관측된 구간만)
        if y_min is None:
            y_min = int(max(np.min(ly), np.min(ry)))
        if y_max is None:
            y_max = int(min(np.max(ly), np.max(ry)))
        if y_max <= y_min:
            return None

        y_vals = np.linspace(y_min, y_max, n_samples)

        # 3) 같은 y에서 좌/우 x 평가 후 중점
        xL = np.polyval(cL, y_vals)
        xR = np.polyval(cR, y_vals)
        xC = 0.5 * (xL + xR)

        return xC, y_vals  # (x_center, y) in BEV pixel coords
    
    def fit_y_as_fn_of_x(self, x, y, deg=2):
        """
        y = a*x^2 + b*x + c 형태로 피팅.
        deg=2가 보통 차선에 안정적.
        """
        if len(x) < deg + 1:
            return None
        return np.polyfit(x, y, deg)  # returns [a,b,c]

    def eval_poly(self, coeff, x):
        return np.polyval(coeff, x)

    def debug_print_lanes(self, detected_lanes):
        if detected_lanes is None:
            rospy.loginfo("[LaneDebug] detected_lanes = None")
            return

        rospy.loginfo(
            "[LaneDebug] LaneArray | frame=%s | lanes=%d",
            detected_lanes.header.frame_id,
            len(detected_lanes.lanes)
        )

        for lane in detected_lanes.lanes:
            n_pts = len(lane.lane_lines)
            rospy.loginfo(
                "  Lane id=%d | type=%d | points=%d",
                lane.id, lane.lane_type, n_pts
            )

            if n_pts == 0:
                continue

            # --- 첫/끝 포인트 ---
            p0 = lane.lane_lines[0]
            pN = lane.lane_lines[-1]
            rospy.loginfo(
                "    start: (%.2f, %.2f, %.2f)",
                p0.x, p0.y, p0.z
            )
            rospy.loginfo(
                "    end  : (%.2f, %.2f, %.2f)",
                pN.x, pN.y, pN.z
            )

            # --- 거리 통계 ---
            dists = []
            vx = self.vehicle_state.pose.position.x
            vy = self.vehicle_state.pose.position.y
            for p in lane.lane_lines:
                dists.append(np.hypot(p.x - vx, p.y - vy))

            rospy.loginfo(
                "    dist[min/mean/max] = %.1f / %.1f / %.1f m",
                min(dists), sum(dists)/len(dists), max(dists)
            )

            # --- 샘플 포인트 (최대 5개) ---
            step = max(1, n_pts // 5)
            rospy.loginfo("    sample points:")
            for i in range(0, n_pts, step):
                p = lane.lane_lines[i]
                rospy.loginfo(
                    "      [%3d] (%.2f, %.2f, %.2f)",
                    i, p.x, p.y, p.z
                )


    def _process_lane_pixels(self, x_vals, y_vals, Minv, lane_id):
        """
        BEV 픽셀들로부터 차선을 피팅하고 Map 좌표로 변환하는 함수
        프로세스: BEV 픽셀 -> (Minv) -> 원본 이미지 픽셀 -> (pixel_to_vehicle) -> 차량 3D 좌표 -> (Rotation) -> 지도 좌표
        """
        if len(x_vals) < 10: return None

        # 1. BEV 상에서 2차 함수 피팅 (Curve Fitting)
        # y값(세로)에 따른 x값(가로)의 변화를 모델링
        fit = np.polyfit(y_vals, x_vals, 2)
        poly = np.poly1d(fit)

        # 2. 샘플링 (이미지 아래쪽 ~ 중간 정도까지)
        # 너무 멀리 있는 점(y=0 근처)은 오차가 크므로 y=height*0.2 정도까지만 사용
        plot_y = np.linspace(self.current_image.shape[0] * 0.95, self.current_image.shape[0] * 0.55, 12)
        plot_x = poly(plot_y)

        lane_msg = Lane()
        lane_msg.id = 900 + lane_id
        lane_msg.lane_type = 1
        lane_msg.header.stamp = rospy.Time.now()
        lane_msg.header.frame_id = "map"

        # 3. 좌표 변환 체인 (Chain of Transformations)
        # OpenCV 함수를 사용해 행렬 곱셈을 더 깔끔하게 처리
        
        # (1) BEV Pixel -> Original Image Pixel
        # perspectiveTransform은 (N, 1, 2) 형태의 입력을 받습니다.
        bev_points = np.array([np.transpose(np.vstack([plot_x, plot_y]))], dtype=np.float32)
        image_points = cv2.perspectiveTransform(bev_points, Minv) # 결과: (1, N, 2)
        
        image_points = image_points[0] # (N, 2)

        for (u, v) in image_points:
            # (2) Original Image Pixel -> Vehicle Coordinate (Local Meter)
            # 이전에 만든 pixel_to_vehicle 함수 사용 (Ray Casting)
            vehicle_pt = self.pixel_to_vehicle(u, v)

            if vehicle_pt is None:
                continue

            if vehicle_pt.x > self.detection_range:
                continue
            # (3) Vehicle Coordinate -> Map Coordinate (Global Meter)
            # 차량의 현재 위치와 방향을 더해줌
            map_pt = self._vehicle_to_map_transform(vehicle_pt)
            
            lane_msg.lane_lines.append(map_pt)
            
        return lane_msg

    def _vehicle_to_map_transform(self, vehicle_pt):
        """차량 기준 좌표를 지도 좌표로 변환"""
        vx, vy = vehicle_pt.x, vehicle_pt.y
        
        vehicle_x = self.vehicle_state.pose.position.x
        vehicle_y = self.vehicle_state.pose.position.y
        
        quat = self.vehicle_state.pose.orientation
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # 회전 + 이동
        map_x = vehicle_x + (vx * np.cos(yaw) - vy * np.sin(yaw))
        map_y = vehicle_y + (vx * np.sin(yaw) + vy * np.cos(yaw))
        
        point = Point()
        point.x = map_x
        point.y = map_y
        point.z = 0.0
        return point

    # def _fit_and_transform_bev(self, x_vals, y_vals, M, lane_id):
    #     """
    #     BEV 상의 픽셀들로부터 2차 함수 피팅 후 Map 좌표로 변환
        
    #     중요: BEV 픽셀 좌표 -> 원본 이미지 픽셀 -> Map 좌표 순으로 변환하거나
    #           BEV 픽셀 -> 실제 거리(Meter) 비율을 알면 바로 변환 가능합니다.
    #           여기서는 정확성을 위해 [BEV -> 원본 -> Map] 방식을 사용합니다.
    #     """
    #     if len(x_vals) < 10: return None

    #     # 2차 함수 피팅 (x = ay^2 + by + c) - 곡선 도로 대응
    #     fit = np.polyfit(y_vals, x_vals, 2)
    #     poly = np.poly1d(fit)

    #     # 샘플링 (BEV 이미지의 y축 전체에 대해)
    #     plot_y = np.linspace(0, 480 - 1, 10) # 10개 점만 추출
    #     plot_x = poly(plot_y)

    #     lane_msg = Lane()
    #     lane_msg.id = lane_id
    #     lane_msg.lane_type = 1
    #     lane_msg.header.stamp = rospy.Time.now()
    #     lane_msg.header.frame_id = "map"

    #     # 역변환 행렬 (BEV -> Original Image)
    #     Minv = np.linalg.inv(M)

    #     for px, py in zip(plot_x, plot_y):
    #         # 1. BEV Pixel -> Original Image Pixel
    #         # 행렬 곱: P_orig = Minv @ P_bev
    #         vec = np.array([px, py, 1.0])
    #         orig_vec = np.dot(Minv, vec)
            
    #         # 동차 좌표 정규화 (scale로 나누기)
    #         if orig_vec[2] != 0:
    #             u = orig_vec[0] / orig_vec[2]
    #             v = orig_vec[1] / orig_vec[2]
    #         else:
    #             continue

    #         # 2. Original Image Pixel -> Map Coordinate
    #         # (이전에 만든 pixel_to_map 함수 재사용!)
    #         map_point = self._pixel_to_map(u, v)
            
    #         if map_point:
    #             lane_msg.lane_lines.append(map_point)
        
    #     return lane_msg

    def pixel_to_vehicle(self, u, v):
        """
        이미지 픽셀 좌표(u, v)를 차량 기준 좌표(x, y)로 변환
        
        가정:
        1. 핀홀 카메라 모델 (왜곡 무시 혹은 보정된 이미지라 가정)
        2. 지면은 평평하다 (Flat Ground Assumption, Z_vehicle = 0)
        
        Args:
            u: 이미지 x좌표 (가로)
            v: 이미지 y좌표 (세로)
            
        Returns:
            point: geometry_msgs/Point (x, y, z=0) 차량 기준 좌표
                   변환 불가능한 경우(하늘 등) None 반환
        """
        # 1. 카메라 파라미터 (sensor_camera_node.py와 동일한 ROS params 사용)
        f = float(getattr(self, "cam_f", 320.0))
        cx = float(getattr(self, "cam_cx", 320.0))
        cy = float(getattr(self, "cam_cy", 240.0))

        # 카메라 오프셋 (vehicle frame)
        offset_x = float(getattr(self, "cam_offset_x", 1.1))
        offset_y = float(getattr(self, "cam_offset_y", 0.0))
        offset_z = float(getattr(self, "cam_offset_z", 1.4))
        # 2. 픽셀 좌표 -> 정규화된 이미지 평면 좌표 (Normalized Image Plane)
        # u = f * (x / z) + cx  => x/z = (u - cx) / f
        # v = f * (y / z) + cy  => y/z = (v - cy) / f
        
        if f <= 0: return None
        
        # 카메라 좌표계 기준 (Right, Down, Forward)의 방향 벡터
        # cam_z_forward = 1 로 가정했을 때의 x, y 값
        n_x = (u - cx) / f  # Right
        n_y = (v - cy) / f  # Down
        n_z = 1.0           # Forward
        
        # 3. 카메라 좌표계 -> 차량 좌표계 방향 벡터 변환
        # sensor_camera_node.py의 world_to_camera_frame 역연산
        
        # 코드상의 매핑 관계:
        # camera_x_right   = -vehicle_y
        # camera_y_down    = -vehicle_z
        # camera_z_forward =  vehicle_x
        
        # 이를 역으로 차량 좌표계 기준의 방향 벡터(ray)로 변환하면:
        ray_x = n_z   # Forward -> Vehicle X
        ray_y = -n_x  # Right   -> -Vehicle Y (Left)
        ray_z = -n_y  # Down    -> -Vehicle Z (Up)
        
        ray_vec = np.array([ray_x, ray_y, ray_z])
        
        # 4. 카메라 회전 적용
        # sensor_camera_node.py에서 R_camera는 'vehicle -> camera' 회전입니다.
        # 여기서는 지면으로 쏘는 ray를 차량 좌표계에서 쓰려면 역회전(전치)을 적용해야 합니다.
        rotated_ray = self.R_camera_to_vehicle @ ray_vec
        # 5. 지면 투영 (Flat Ground Assumption)
        # 레이 방정식: P = Camera_Origin + t * Ray_Direction
        # 우리는 P.z = 0 (지면)이 되는 t를 찾아야 함
        # 0 = offset_z + t * rotated_ray.z
        # t = -offset_z / rotated_ray.z
        
        # 수평선 위쪽(하늘)을 클릭한 경우 처리
        if rotated_ray[2] >= -1e-6: 
            return None
            
        t = -offset_z / rotated_ray[2]
        
        # 차량 기준 좌표 계산 (z는 당연히 0)
        vehicle_x = offset_x + t * rotated_ray[0]
        vehicle_y = offset_y + t * rotated_ray[1]
        
        # 전방에 있는 점만 유효하다고 가정 (선택 사항)
        if vehicle_x < 0:
            return None

        point = Point()
        point.x = vehicle_x
        point.y = vehicle_y
        point.z = 0.0
        
        return point

    # def _draw_fitted_line(self, img, points, height, color):
    #     """디버깅용: 피팅된 차선을 이미지에 그리는 헬퍼 함수"""
    #     points = np.array(points)
    #     x = points[:, 0]
    #     y = points[:, 1]
        
    #     if len(x) < 2: return

    #     # 1차 함수 피팅 (x = ay + b)
    #     fit = np.polyfit(y, x, 1)
    #     poly = np.poly1d(fit)

    #     # 그리기용 포인트 생성 (이미지 하단 ~ ROI 상단)
    #     plot_y = np.linspace(height * 0.6, height, 10)
    #     plot_x = poly(plot_y)
        
    #     # 정수형 좌표로 변환하여 그리기
    #     pts = np.array([np.transpose(np.vstack([plot_x, plot_y]))], np.int32)
    #     cv2.polylines(img, pts, False, color, thickness=3)

    # def _create_lane_msg_from_points(self, points, img_height, lane_id):
    #     """이미지 포인트들로부터 Lane 메시지 생성"""
    #     if not points:
    #         return None
            
    #     # 다항식 피팅 (1차 함수: x = ay + b)
    #     # y를 기준으로 x를 구하는 것이 수직선에 가까운 차선에서 더 안정적임
    #     points = np.array(points)
    #     x = points[:, 0]
    #     y = points[:, 1]
        
    #     if len(x) < 2: return None
        
    #     fit = np.polyfit(y, x, 1) # x = fit[0]*y + fit[1]
    #     poly = np.poly1d(fit)

    #     # 샘플링 (이미지 하단에서 ROI 상단까지)
    #     y_samples = np.linspace(img_height, img_height * 0.6, 10)
    #     x_samples = poly(y_samples)

    #     lane_msg = Lane()
    #     lane_msg.id = lane_id
    #     lane_msg.lane_type = 1 # solid
    #     lane_msg.header.stamp = rospy.Time.now()
    #     lane_msg.header.frame_id = "map"

    #     # 좌표 변환 (Pixel -> Vehicle -> Map)
    #     for vx, vy in zip(x_samples, y_samples):
    #         map_point = self._pixel_to_map(vx, vy)
    #         if map_point:
    #             lane_msg.lane_lines.append(map_point)
                
    #     return lane_msg

    # def _pixel_to_map(self, u, v):
    #     """픽셀 좌표 -> 차량 좌표 -> 지도 좌표 변환"""
        
    #     # 1. 픽셀 -> 차량 좌표 (위에서 만든 함수 사용)
    #     vehicle_point = self.pixel_to_vehicle(u, v)
    #     if vehicle_point is None:
    #         return None
            
    #     local_x = vehicle_point.x
    #     local_y = vehicle_point.y
        
    #     # 2. 차량 좌표 -> 지도(Map) 좌표
    #     # (기존 코드와 동일)
    #     vehicle_x = self.vehicle_state.pose.position.x
    #     vehicle_y = self.vehicle_state.pose.position.y
        
    #     quat = self.vehicle_state.pose.orientation
    #     siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
    #     cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    #     yaw = np.arctan2(siny_cosp, cosy_cosp)
        
    #     map_x = vehicle_x + (local_x * np.cos(yaw) - local_y * np.sin(yaw))
    #     map_y = vehicle_y + (local_x * np.sin(yaw) + local_y * np.cos(yaw))
        
    #     point = Point()
    #     point.x = map_x
    #     point.y = map_y
    #     point.z = 0.0
        
    #     return point

    def callback_timer(self, event):
        """타이머 콜백 - 주기적으로 차선 검출 수행"""

        if self.vehicle_state is None:
            return

        # 검출 모드에 따라 차선 검출
        if self.detection_mode == "csv":
            detected_lanes = self.detect_lanes_from_csv()
        else:  # "image"
            detected_lanes = self.detect_lanes_from_image()

        # 검출된 차선 발행
        if detected_lanes is not None:
            self.lane_pub.publish(detected_lanes)

            # RViz용 marker publish
            marker_array = self._to_marker_array(detected_lanes)
            self.marker_pub.publish(marker_array)       


def main():
    """메인 함수"""
    rospy.init_node("lane_detection_node")

    try:
        node = LaneDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
