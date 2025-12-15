#!/usr/bin/env python
"""
Evaluation Node

- lanes.csv 기반 GT 차선으로 CTE 계산
- image 기반 lane_detection 결과(/perception/lanes)의 middle lane(point 집합)을 받아
  GT 중앙선(center_lanes[1] = lane0-lane1 중점 polyline)과 비교하여 정확도 지표(RMSE 등) 계산
- metrics를 CSV 파일로 저장

입력:
- /vehicle/state: 차량 상태
- /perception/lanes: 이미지 기반 차선 검출 결과 (LaneArray)
- maps/lanes.csv: ground truth 차선 정보

출력:
- /evaluation/cte: Cross Track Error (m)
- logs/evaluation_metrics.csv: 평가 지표 로그
"""

import os
import rospy
import numpy as np
import math
import csv
from dcas_msgs.msg import VehicleState, LaneArray
from std_msgs.msg import Float32
import rospkg


class EvaluationNode:
    def __init__(self):
        # -------------------------
        # ROS 파라미터
        # -------------------------
        self.evaluation_rate = rospy.get_param("~evaluation_rate", 10.0)

        # lane_detection_node 기준: center lane id = 900+1 = 901
        self.detected_center_lane_id = int(rospy.get_param("~detected_center_lane_id", 901))

        # metrics 저장 파일
        self.metrics_csv_path = rospy.get_param("~metrics_csv_path", "")
        self.metrics_flush_every = int(rospy.get_param("~metrics_flush_every", 10))

        # -------------------------
        # 상태 변수
        # -------------------------
        self.vehicle_state = None
        self.lane_data = {}
        self.center_lanes = {}
        self.cte_history = []
        self.max_cte = 0.0

        # 최신 image 기반 차선 검출 결과
        self.latest_detected_lanes = None

        # lane accuracy history (detected vs GT center_lanes[1])
        self.lane_rmse_history = []
        self.max_lane_rmse = 0.0

        # CSV row buffer
        self._pending_rows = []

        # -------------------------
        # GT 로드 및 중앙선 계산
        # -------------------------
        self._load_lane_csv()

        # -------------------------
        # metrics csv 경로 초기화
        # -------------------------
        self._init_metrics_path()

        # -------------------------
        # Pub/Sub
        # -------------------------
        self.cte_pub = rospy.Publisher("/evaluation/cte", Float32, queue_size=10)

        rospy.Subscriber("/vehicle/state", VehicleState, self.callback_vehicle_state)
        rospy.Subscriber("/perception/lanes", LaneArray, self.callback_detected_lanes)

        # -------------------------
        # Timer / Shutdown
        # -------------------------
        timer_period = 1.0 / self.evaluation_rate
        self.timer = rospy.Timer(rospy.Duration(timer_period), self.callback_timer)

        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo("Evaluation node initialized")

    # =========================================================================
    # Init / Load
    # =========================================================================
    def _init_metrics_path(self):
        try:
            if self.metrics_csv_path and len(self.metrics_csv_path) > 0:
                out_path = self.metrics_csv_path
            else:
                rospack = rospkg.RosPack()
                pkg_path = rospack.get_path('dcas_perception_sim')
                logs_dir = os.path.join(pkg_path, 'logs')
                out_path = os.path.join(logs_dir, 'evaluation_metrics.csv')

            out_dir = os.path.dirname(out_path)
            if out_dir and (not os.path.exists(out_dir)):
                os.makedirs(out_dir, exist_ok=True)

            self.metrics_csv_path = out_path

            if not os.path.exists(self.metrics_csv_path):
                self._write_metrics_header()

            rospy.loginfo(f"Metrics CSV path: {self.metrics_csv_path}")

        except Exception as e:
            rospy.logerr(f"Failed to init metrics csv path: {e}")
            home_ros = os.path.join(os.path.expanduser("~"), ".ros")
            os.makedirs(home_ros, exist_ok=True)
            self.metrics_csv_path = os.path.join(home_ros, "evaluation_metrics.csv")
            if not os.path.exists(self.metrics_csv_path):
                self._write_metrics_header()

    def _write_metrics_header(self):
        header = [
            "t_sec",
            "cte_m",
            "abs_cte_m",
            "cte_mean_m",
            "cte_max_m",
            "det_lane_id",
            "gt_center_lane_id",  # here: center_lanes[1]
            "n_det_pts",
            "lane_rmse_m",
            "lane_mean_err_m",
            "lane_max_err_m",
        ]
        try:
            with open(self.metrics_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        except Exception as e:
            rospy.logerr(f"Failed to write metrics header: {e}")

    def _load_lane_csv(self):
        try:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('dcas_perception_sim')
            csv_path = os.path.join(pkg_path, 'maps', 'lanes.csv')

            if not os.path.exists(csv_path):
                rospy.logerr(f"Lane CSV file not found: {csv_path}")
                return

            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lane_id = int(row['lane_id'])
                    if lane_id not in self.lane_data:
                        self.lane_data[lane_id] = {'points': [], 'lane_type': int(row['lane_type'])}

                    p = np.array([float(row['x']), float(row['y']), float(row['z'])], dtype=np.float64)
                    self.lane_data[lane_id]['points'].append(p)

            for lane_id in self.lane_data:
                self.lane_data[lane_id]['points'] = np.array(self.lane_data[lane_id]['points'], dtype=np.float64)

            rospy.loginfo(f"Loaded {len(self.lane_data)} lanes from CSV")

            self._compute_center_lanes()

        except Exception as e:
            rospy.logerr(f"Failed to load lane CSV: {e}")

    def _compute_center_lanes(self):
        self.center_lanes = {}

        if 0 not in self.lane_data or 1 not in self.lane_data or 2 not in self.lane_data:
            rospy.logwarn("Lane ID 0, 1, or 2 not found. Cannot compute center lanes.")
            return

        lane0 = self.lane_data[0]['points']
        lane1 = self.lane_data[1]['points']
        lane2 = self.lane_data[2]['points']

        # center_lanes[1] = (lane0 + lane1)/2  (CTE/Accuracy GT 기준으로 사용)
        n01 = min(len(lane0), len(lane1))
        self.center_lanes[1] = (lane0[:n01] + lane1[:n01]) / 2.0

        # center_lanes[2] = (lane1 + lane2)/2 (CTE 계산 시 보조로도 사용)
        n12 = min(len(lane1), len(lane2))
        self.center_lanes[2] = (lane1[:n12] + lane2[:n12]) / 2.0

        rospy.loginfo(
            f"Computed center lanes: center_lane_1 ({len(self.center_lanes[1])} pts), "
            f"center_lane_2 ({len(self.center_lanes[2])} pts)"
        )

    # =========================================================================
    # Callbacks
    # =========================================================================
    def callback_vehicle_state(self, msg):
        self.vehicle_state = msg

    def callback_detected_lanes(self, msg):
        self.latest_detected_lanes = msg

    # =========================================================================
    # Geometry helpers
    # =========================================================================
    def find_closest_point_on_lane(self, vehicle_pos, lane_points):
        if lane_points is None or len(lane_points) < 2:
            return None, None, float('inf')

        min_distance = float('inf')
        closest_point = None
        closest_segment_idx = 0

        vehicle_xy = vehicle_pos[:2]

        for i in range(len(lane_points) - 1):
            p1 = lane_points[i][:2]
            p2 = lane_points[i + 1][:2]

            seg = p2 - p1
            seg_len2 = float(np.dot(seg, seg))

            if seg_len2 < 1e-9:
                proj = p1
            else:
                t = float(np.dot(vehicle_xy - p1, seg) / seg_len2)
                t = float(np.clip(t, 0.0, 1.0))
                proj = p1 + t * seg

            d = float(np.linalg.norm(vehicle_xy - proj))
            if d < min_distance:
                min_distance = d
                closest_point = np.array([proj[0], proj[1], 0.0], dtype=np.float64)
                closest_segment_idx = i

        return closest_point, closest_segment_idx, min_distance

    def point_to_polyline_distance(self, p_xy, polyline_points):
        if polyline_points is None or len(polyline_points) < 2:
            return float('inf')

        p = np.asarray(p_xy, dtype=np.float64)
        min_d = float('inf')

        for i in range(len(polyline_points) - 1):
            a = polyline_points[i][:2]
            b = polyline_points[i + 1][:2]
            ab = b - a
            denom = float(np.dot(ab, ab))

            if denom < 1e-9:
                proj = a
            else:
                t = float(np.dot(p - a, ab) / denom)
                t = float(np.clip(t, 0.0, 1.0))
                proj = a + t * ab

            d = float(np.linalg.norm(p - proj))
            if d < min_d:
                min_d = d

        return min_d

    # =========================================================================
    # Metrics: CTE
    # =========================================================================
    def calculate_cte(self):
        if self.vehicle_state is None:
            return None
        if not self.center_lanes:
            rospy.logwarn_throttle(5.0, "Center lanes not available")
            return None

        vehicle_x = self.vehicle_state.pose.position.x
        vehicle_y = self.vehicle_state.pose.position.y
        vehicle_pos = np.array([vehicle_x, vehicle_y, 0.0], dtype=np.float64)

        min_cte = None
        min_dist = float('inf')

        for _, center_pts in self.center_lanes.items():
            closest_point, seg_idx, dist = self.find_closest_point_on_lane(vehicle_pos, center_pts)
            if closest_point is None:
                continue

            if dist < min_dist:
                min_dist = dist

                # 차선 방향 벡터
                if seg_idx < len(center_pts) - 1:
                    p1 = center_pts[seg_idx][:2]
                    p2 = center_pts[seg_idx + 1][:2]
                    lane_vec = p2 - p1
                elif seg_idx > 0:
                    p1 = center_pts[seg_idx - 1][:2]
                    p2 = center_pts[seg_idx][:2]
                    lane_vec = p2 - p1
                else:
                    min_cte = dist
                    continue

                nv = float(np.linalg.norm(lane_vec))
                if nv < 1e-9:
                    min_cte = dist
                    continue

                lane_dir = lane_vec / nv
                v2l = vehicle_pos[:2] - closest_point[:2]
                cross_z = float(lane_dir[0] * v2l[1] - lane_dir[1] * v2l[0])
                min_cte = math.copysign(dist, cross_z)

        return min_cte

    def update_cte_metrics(self, cte):
        if cte is None:
            return
        self.cte_history.append(abs(cte))
        if abs(cte) > self.max_cte:
            self.max_cte = abs(cte)

    # =========================================================================
    # Metrics: Lane accuracy (detected vs GT center_lanes[1])
    # =========================================================================
    def extract_detected_center_lane_points(self):
        if self.latest_detected_lanes is None:
            return None

        for lane in self.latest_detected_lanes.lanes:
            if int(lane.id) == int(self.detected_center_lane_id):
                pts = []
                for p in lane.lane_lines:
                    pts.append([p.x, p.y, p.z])
                if len(pts) < 2:
                    return None
                return np.asarray(pts, dtype=np.float64)

        return None

    def get_gt_center_lane_points_for_eval(self):
        """GT 기준: center_lanes[1] (lane0-lane1 중앙선)"""
        if 1 in self.center_lanes and self.center_lanes[1] is not None and len(self.center_lanes[1]) >= 2:
            return self.center_lanes[1]
        return None

    def calculate_lane_accuracy_metrics(self):
        det_pts = self.extract_detected_center_lane_points()
        gt_center = self.get_gt_center_lane_points_for_eval()

        if det_pts is None or gt_center is None:
            return None

        dists = []
        for p in det_pts:
            d = self.point_to_polyline_distance(p[:2], gt_center)
            if np.isfinite(d):
                dists.append(d)

        if len(dists) == 0:
            return None

        dists = np.asarray(dists, dtype=np.float64)
        rmse = float(np.sqrt(np.mean(dists ** 2)))
        mean_err = float(np.mean(dists))
        max_err = float(np.max(dists))

        return {
            "n_det": int(len(dists)),
            "rmse": rmse,
            "mean": mean_err,
            "max": max_err
        }

    def update_lane_accuracy_history(self, lane_metrics):
        if lane_metrics is None:
            return
        rmse = float(lane_metrics["rmse"])
        self.lane_rmse_history.append(rmse)
        if rmse > self.max_lane_rmse:
            self.max_lane_rmse = rmse

    # =========================================================================
    # CSV logging
    # =========================================================================
    def append_metrics_row(self, t_sec, cte, lane_metrics):
        cte_mean = float(np.mean(self.cte_history)) if len(self.cte_history) > 0 else float('nan')
        cte_max = float(self.max_cte) if len(self.cte_history) > 0 else float('nan')

        if lane_metrics is None:
            n_det = 0
            lane_rmse = float('nan')
            lane_mean = float('nan')
            lane_max = float('nan')
        else:
            n_det = int(lane_metrics["n_det"])
            lane_rmse = float(lane_metrics["rmse"])
            lane_mean = float(lane_metrics["mean"])
            lane_max = float(lane_metrics["max"])

        row = [
            float(t_sec),
            float(cte) if cte is not None else float('nan'),
            float(abs(cte)) if cte is not None else float('nan'),
            cte_mean,
            cte_max,
            int(self.detected_center_lane_id),
            1,  # gt_center_lane_id = center_lanes[1]
            int(n_det),
            float(lane_rmse),
            float(lane_mean),
            float(lane_max),
        ]
        self._pending_rows.append(row)

        if len(self._pending_rows) >= max(1, self.metrics_flush_every):
            self.flush_metrics_rows()

    def flush_metrics_rows(self):
        if len(self._pending_rows) == 0:
            return
        try:
            with open(self.metrics_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self._pending_rows)
            self._pending_rows = []
        except Exception as e:
            rospy.logerr(f"Failed to flush metrics rows: {e}")

    def on_shutdown(self):
        self.flush_metrics_rows()
        rospy.loginfo("Evaluation node shutdown: metrics flushed")

    # =========================================================================
    # Timer
    # =========================================================================
    def callback_timer(self, event):
        if self.vehicle_state is None:
            return

        # 1) CTE 계산 및 publish
        cte = self.calculate_cte()
        if cte is not None:
            self.cte_pub.publish(Float32(cte))
            self.update_cte_metrics(cte)

        # 2) detected lane vs GT center lane accuracy
        lane_metrics = self.calculate_lane_accuracy_metrics()
        self.update_lane_accuracy_history(lane_metrics)

        # 3) CSV 로깅
        t_sec = rospy.Time.now().to_sec()
        self.append_metrics_row(t_sec, cte, lane_metrics)

        # 4) 주기적 로그
        if len(self.cte_history) > 0 and (len(self.cte_history) % 100 == 0):
            mean_cte = float(np.mean(self.cte_history))
            std_cte = float(np.std(self.cte_history))
            rospy.loginfo(f"[CTE] Mean: {mean_cte:.3f}m, Std: {std_cte:.3f}m, Max: {self.max_cte:.3f}m")

            if len(self.lane_rmse_history) > 0:
                mean_rmse = float(np.mean(self.lane_rmse_history))
                rospy.loginfo(
                    f"[LaneAcc vs GT center_lanes[1]] RMSE Mean: {mean_rmse:.3f}m, "
                    f"RMSE Max: {self.max_lane_rmse:.3f}m (det_lane_id={self.detected_center_lane_id})"
                )


def main():
    rospy.init_node("evaluation_node")
    try:
        node = EvaluationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
