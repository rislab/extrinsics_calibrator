#!/usr/bin/env python2.7
from __future__ import division
import roslib
import rospy
import tf
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np


class GT_cleaner:

    def __init__(self):

        self.last_time = rospy.Time.now()
        self.init = False
        self.broadcaster = tf.TransformBroadcaster()
        self.clean_pub = rospy.Publisher(
            '/gt_clean_odom', Odometry, queue_size=10)
        self.sub = rospy.Subscriber(
            "/mocap/odom", Odometry, self.callback)
        self.first_quat = None
        self.first_pos = np.array([0, 0, 0])
        self.prev_frame = np.eye(4)

    def callback(self, msg):
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        quat = np.array([q.x, q.y, q.z, q.w])
        pos = np.array([p.x, p.y, p.z])
        if self.init == False:
            self.last_time = msg.header.stamp
            self.init = True

            self.first_frame = tf.transformations.quaternion_matrix(quat)
            self.first_frame[:3, 3] = pos
            self.first_frame_inv = np.linalg.inv(self.first_frame)

            return

        dt = (msg.header.stamp - self.last_time).to_sec()
        self.last_time = msg.header.stamp
        frame = tf.transformations.quaternion_matrix(quat)
        frame[:3, 3] = pos

        frame_in_first = np.dot(self.first_frame_inv, frame)

        # add to path
        odom = Odometry()

        odom.header.frame_id = msg.header.frame_id
        odom.pose.pose.position.x = frame_in_first[0, 3]
        odom.pose.pose.position.y = frame_in_first[1, 3]
        odom.pose.pose.position.z = frame_in_first[2, 3]
        q = tf.transformations.quaternion_from_matrix(frame_in_first)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.header.stamp = msg.header.stamp

        #Now time for the velocities
        # Get the delta transform to obtain the velocities
        delta_frame = np.dot(np.linalg.inv(self.prev_frame), frame_in_first)
        self.prev_frame = frame_in_first
        # Linear part is easy
        odom.twist.twist.linear.x = delta_frame[0,3]/dt
        odom.twist.twist.linear.y = delta_frame[1,3]/dt
        odom.twist.twist.linear.z = delta_frame[2,3]/dt
        # For the angular velocity, we compute the angle axis
        result = tf.transformations.rotation_from_matrix(delta_frame)
        angle = result[0]
        direction = result[1]
        omega = direction * angle/dt
        odom.twist.twist.angular.x = omega[0]
        odom.twist.twist.angular.y = omega[1]
        odom.twist.twist.angular.z = omega[2]


        self.clean_pub.publish(odom)

        #Just for lolz publish the body fixed level frame
        level_frame= np.eye(4)
        level_frame[:3,:3] = frame[:3,:3]
        #Only set the rotation to be a rotation about z; so zero out any cross component wrt z
        # level_frame[2,:3] = [0,0,1]
        # level_frame[:3,2] = [0,0,1]
        # Normalize the first two columns
        # level_frame[:3,0] /= np.linalg.norm(level_frame[:3,0])
        # level_frame[:3,1] /= np.linalg.norm(level_frame[:3,1])
        
        # q = tf.transformations.quaternion_from_matrix(level_frame)

        # print level_frame
        level_q = quat
        # Zero out rotation about x and y
        level_q[0] = 0
        level_q[1] = 0
        level_q /= np.linalg.norm(level_q)
        self.broadcaster.sendTransform(frame[:3,3], level_q , msg.header.stamp, 'level', 'world')


if __name__ == '__main__':
    rospy.init_node('gt_cleaner', anonymous=True)
    cleaner_obj = GT_cleaner()
    rospy.spin()
