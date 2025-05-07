#!/usr/bin/env python

import rospy
import tf2_ros
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from commander.srv import SavePointCloud, SavePointCloudResponse

class PointCloudSaver:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.latest_msg = None
        rospy.Subscriber('/points2', PointCloud2, self.pc_callback)
        self.service = rospy.Service('save_pointcloud', SavePointCloud, self.handle_save)

    def pc_callback(self, msg):
        self.latest_msg = msg

    def handle_save(self, req):
        if self.latest_msg is None:
            return SavePointCloudResponse(False, "No point cloud received yet.")

        try:
            # Get the transform from the point cloud frame to base_link
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                self.latest_msg.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0)
            )

            # Build the 4x4 transformation matrix
            t = transform.transform.translation
            q = transform.transform.rotation
            T = self.transform_to_matrix(t, q)

            # Read the original point cloud
            points = list(pc2.read_points(self.latest_msg, field_names=["x", "y", "z"], skip_nans=True))
            if not points:
                return SavePointCloudResponse(False, "Point cloud is empty.")

            np_points = np.array(points, dtype=np.float32)

            # Apply transform to all points (homogeneous coordinates)
            ones = np.ones((np_points.shape[0], 1), dtype=np.float32)
            homogenous_points = np.hstack([np_points[:, :3], ones])
            transformed_points = (T @ homogenous_points.T).T[:, :3]

            # Convert to Open3D and save
            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(transformed_points)
            o3d.io.write_point_cloud(req.filename, o3d_cloud)

            return SavePointCloudResponse(True, "Saved to {}".format(req.filename))

        except Exception as e:
            return SavePointCloudResponse(False, str(e))

    def transform_to_matrix(self, trans, rot):
        import tf.transformations as tft
        trans_vec = [trans.x, trans.y, trans.z]
        rot_quat = [rot.x, rot.y, rot.z, rot.w]
        T = tft.quaternion_matrix(rot_quat)
        T[0:3, 3] = trans_vec
        return T

if __name__ == '__main__':
    rospy.init_node('pointcloud_save_service')
    saver = PointCloudSaver()
    rospy.loginfo("Ready to save point clouds via service.")
    rospy.spin()