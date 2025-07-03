import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import tf_transformations
import numpy as np


class TransformChainListener(Node):
    def __init__(self):
        super().__init__("transform_chain_listener")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_timer(2.0, self.print_transform_once)
        self.transform_printed = False

        # (AP)Define hardcoded transforms
        self.hardcoded_transforms = {
            # From Cloud Compare alignment
            # -0.998947799206 -0.045859057456 0.000430255546 0.023267621174
            # -0.007639066316 0.175637781620 0.984425246716 -0.657857954502
            # -0.045220382512 0.983386158943 -0.175803303719 0.833961486816
            # 0.000000000000 0.000000000000 0.000000000000 1.000000000000
            (
                "gripper_right_left_camera_optical_frame",
                "robot_base_left_camera_optical_frame",
            ): np.array(
                [
                    [-0.998947799206, -0.045859057456, 0.000430255546, 0.023267621174],
                    [-0.007639066316, 0.175637781620, 0.984425246716, -0.657857954502],
                    [-0.045220382512, 0.983386158943, -0.175803303719, 0.833961486816],
                    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000],
                ]
            ),
            # From URDF of zed2i camera. Needed because until we run this script,
            # the zed2i may not be in the URDF, and thus can't be looked up on tf.
            (
                "robot_base_left_camera_optical_frame",
                "robot_base_camera_link",
            ): np.array(
                [
                    [0.000, -1.000, 0.000, 0.060],
                    [0.000, 0.000, -1.000, 0.015],
                    [1.000, 0.000, 0.000, 0.010],
                    [0.000, 0.000, 0.000, 1.000],
                ]
            ),
        }

    def print_transform_once(self):
        if not self.transform_printed:
            try:
                self.print_transform_chain()
                self.transform_printed = True
                self.get_logger().info("Transform printed. Press Ctrl+C to exit.")
            except Exception as e:
                self.get_logger().warn(f"Failed to print transform, will retry: {e}")

    def print_transform_chain(self):
        # (AP) Chain can be visualized in the urdf file using Rviz.
        frames = [
            ("torso1", "torso2"),
            ("torso2", "LA_base_link"),
            ("LA_base_link", "LA_ee_link"),
            ("LA_ee_link", "gripper_right_left_camera_optical_frame"),
            (
                "gripper_right_left_camera_optical_frame",
                "robot_base_left_camera_optical_frame",
            ),
            ("robot_base_left_camera_optical_frame", "robot_base_camera_link"),
        ]

        composed_matrix = np.identity(4)

        for parent, child in frames:
            # (AP) Get hardcoded transforms
            if (parent, child) in self.hardcoded_transforms:
                mat = self.hardcoded_transforms[(parent, child)]
                self.get_logger().info(f"Using hardcoded transform {parent} -> {child}")
            else:
                # (AP) Use TF2 lookup for all other transforms
                tf: TransformStamped = self.tf_buffer.lookup_transform(
                    parent,
                    child,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0),
                )
                mat = self.transform_to_matrix(tf.transform)
                self.get_logger().info(f"Using TF2 lookup for {parent} -> {child}")

            composed_matrix = np.dot(composed_matrix, mat)

        self.get_logger().info("========================================")
        self.get_logger().info(
            f"Composed transform from torso1 to robot_base_camera_link:"
        )
        for row in composed_matrix:
            self.get_logger().info(
                f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}"
            )
        self.get_logger().info("========================================")

        # Extract and print XYZ and RPY values
        xyz = composed_matrix[:3, 3]
        rpy = tf_transformations.euler_from_matrix(composed_matrix[:3, :3], "sxyz")
        rpy_degrees = np.degrees(rpy)

        self.get_logger().info("Translation (XYZ):")
        self.get_logger().info(f"X: {xyz[0]:.6f}, Y: {xyz[1]:.6f}, Z: {xyz[2]:.6f}")
        self.get_logger().info("Rotation (RPY) in radians:")
        self.get_logger().info(
            f"Roll: {rpy[0]:.6f}, Pitch: {rpy[1]:.6f}, Yaw: {rpy[2]:.6f}"
        )
        self.get_logger().info("Rotation (RPY) in degrees:")
        self.get_logger().info(
            f"Roll: {rpy_degrees[0]:.6f}, Pitch: {rpy_degrees[1]:.6f}, Yaw: {rpy_degrees[2]:.6f}"
        )

        # Format for URDF joint specification
        self.get_logger().info("========================================")
        self.get_logger().info("For URDF joint specification:")
        self.get_logger().info(
            f'<origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}" />'
        )
        self.get_logger().info("========================================")

        self.get_logger().info("Transformation Matrix:")
        matrix_str = "np.array([\n"
        for i in range(4):
            matrix_str += f"    [{composed_matrix[i,0]:.12f}, {composed_matrix[i,1]:.12f}, {composed_matrix[i,2]:.12f}, {composed_matrix[i,3]:.12f}],\n"
        matrix_str += "])"
        self.get_logger().info(matrix_str)

    def transform_to_matrix(self, transform):
        trans = transform.translation
        rot = transform.rotation
        translation = tf_transformations.translation_matrix([trans.x, trans.y, trans.z])
        rotation = tf_transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        return np.dot(translation, rotation)


def main():
    rclpy.init()
    node = TransformChainListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
