#!/usr/bin/env python
import math
import numpy as np
from typing import Optional

import rospy
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker, MarkerArray
import tf
from std_msgs.msg import Float32
from dcas_msgs.msg import VehicleState
try:
    from vehicleModel import VehicleModel as PhysicsVehicleModel
except Exception:
    # Fallback: add script directory to path and retry
    import os, sys
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    from vehicleModel import VehicleModel as PhysicsVehicleModel


def YawToQuaternion(yaw: float) -> Quaternion:
    half = yaw * 0.5
    return Quaternion(0.0, 0.0, math.sin(half), math.cos(half))


## KinematicBicycle removed: replaced by VehicleDynamicsNode using VehicleModel


class VehicleDynamicsNode:
    """ROS node that runs the 7-DOF VehicleModel and publishes VehicleState.

    Inputs:
      - /vehicle/cmd/steer: steering wheel angle in degrees
      - /vehicle/cmd/accel: longitudinal acceleration command (m/s^2)

    The acceleration command is mapped to a throttle command in [-1, 1].
    """

    def __init__(self) -> None:
        # Parameters
        self.dt: float = rospy.get_param("~dt", 0.02)
        self.max_accel: float = rospy.get_param("~max_accel", 3.0)
        self.max_decel: float = rospy.get_param("~max_decel", -6.0)

        # Model and state [x, y, psi, vx, vy, yaw_rate, ax, ay]
        model = PhysicsVehicleModel()
        # Override model params from ROS if provided (use current model defaults otherwise)
        for key in [
            'm','Iz','lf','lr','width','h','Iw','Rw','Caf','Car','mu','rho','Cd','A','g'
        ]:
            param_name = f"~model/{key}"
            default_val = getattr(model, key)
            setattr(model, key, rospy.get_param(param_name, default_val))
        self.model = model
        self.state = np.array([
            0.0,  # x
            1.750,  # y
            0.0,  # psi
            0.0,  # vx
            0.0,  # vy
            0.0,  # yaw_rate
            0.0,  # ax
            0.0,  # ay
        ], dtype=float)

        # Command inputs
        self.steer_deg_cmd: float = 0.0
        self.accel_cmd: float = 0.0

        # IO
        self.pub_state = rospy.Publisher("/vehicle/state", VehicleState, queue_size=10)
        self.pub_viz = rospy.Publisher("/viz/vehicle", MarkerArray, queue_size=10)
        self.tf_br = tf.TransformBroadcaster()
        rospy.Subscriber("/vehicle/cmd/steer", Float32, self.CallbackSteer)
        rospy.Subscriber("/vehicle/cmd/accel", Float32, self.CallbackAccel)

        # Timer
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.CallbackTimer)

    def CallbackSteer(self, msg: Float32) -> None:
        self.steer_deg_cmd = float(msg.data)

    def CallbackAccel(self, msg: Float32) -> None:
        self.accel_cmd = float(msg.data)

    def _accel_to_throttle(self, accel_cmd: float) -> float:
        # Map acceleration command (m/s^2) to throttle [-1, 1]
        if accel_cmd >= 0.0:
            denom = max(1e-6, self.max_accel)
            return max(0.0, min(1.0, accel_cmd / denom))
        else:
            denom = max(1e-6, abs(self.max_decel))
            return -max(0.0, min(1.0, abs(accel_cmd) / denom))

    def CallbackTimer(self, _evt) -> None:
        # Build inputs for dynamics
        delta = math.radians(self.steer_deg_cmd)
        throttle = self._accel_to_throttle(self.accel_cmd)

        # Integrate one step
        self.state = np.array(self.model.euler_step(self.state, [delta, throttle], self.dt), dtype=float)
        # Enforce non-negative forward speed (no reverse)
        if self.state[3] < 0.0:
            self.state[3] = 0.0

        # Unpack state
        x, y, psi = float(self.state[0]), float(self.state[1]), float(self.state[2])
        vx, vy, yaw_rate = float(self.state[3]), float(self.state[4]), float(self.state[5])
        ax, ay = float(self.state[6]), float(self.state[7])

        # Publish VehicleState
        msg = VehicleState()
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation = YawToQuaternion(psi)
        msg.twist.linear.x = max(0.0, vx)
        msg.twist.linear.y = vy
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = yaw_rate
        msg.acceleration.x = ax
        msg.acceleration.y = ay
        msg.acceleration.z = 9.81
        msg.steering_wheel_angle_deg = self.steer_deg_cmd
        self.pub_state.publish(msg)

        # Publish TF: map -> ego_frame
        self.tf_br.sendTransform(
            (x, y, 0.0),
            (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ),
            rospy.Time.now(),
            "ego_frame",
            "map",
        )

        # Publish visualization markers (mesh + text)
        arr = MarkerArray()
        mesh = Marker()
        mesh.header.frame_id = "ego_frame"
        mesh.header.stamp = rospy.Time.now()
        mesh.ns = "vehicle"
        mesh.id = 0
        mesh.type = Marker.MESH_RESOURCE
        mesh.action = Marker.ADD
        mesh.mesh_resource = "package://dcas_perception_sim/rviz/meshes/Ioniq5.stl"
        mesh.scale.x = 1.0
        mesh.scale.y = 1.0
        mesh.scale.z = 1.0
        mesh.color.r, mesh.color.g, mesh.color.b, mesh.color.a = 1.0, 1.0, 1.0, 1.0
        mesh.pose.orientation.z = 1.0
        mesh.pose.orientation.w = 0.0
        arr.markers.append(mesh)

        text = Marker()
        text.header.frame_id = "ego_frame"
        text.header.stamp = rospy.Time.now()
        text.ns = "vehicle_text"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.z = 2.2
        text.scale.z = 0.7
        text.text = f"v={vx:.1f} m/s, steer={self.steer_deg_cmd:.1f} deg, thr={throttle:.2f}"
        text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 1.0, 1.0
        arr.markers.append(text)

        self.pub_viz.publish(arr)

def Main() -> None:
    rospy.init_node("vehicle_kinematic_node")
    VehicleDynamicsNode()
    rospy.loginfo("vehicle_kinematic_node: VehicleModel dynamics started -> /vehicle/state")
    rospy.spin()


if __name__ == "__main__":
    Main()
