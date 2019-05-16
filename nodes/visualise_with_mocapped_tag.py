#!/usr/bin/env python2.7
import numpy as np
import sys
import cv2
import tf
import pdb
import yaml

import rosbag
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

from apriltag_tracker._AprilTagTracker import AprilTagTracker
from apriltag_tracker.msg import Apriltags

datatype = np.float32
np.set_printoptions(precision=2, suppress=True)

path = '/media/data/bagfiles/kinect_one/extrinsic_calib.bag'
config_file = '../config/extrinsics_calib.yaml'

with open(config_file, 'r') as f:
    node = yaml.load(f)['apriltag_tracker_params']
    tag_size = node['tag_size']
    K = np.array(node['K']).reshape(3, 3)

tracker = AprilTagTracker(config_file)

rospy.init_node('broadcaster')
bridge = CvBridge()
broadcaster = tf.TransformBroadcaster()
img_pub = rospy.Publisher('debug_img', Image)


def transform_matrix_from_odom(msg):
    translation = np.array(
        [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
    quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
    # translation = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
    # quaternion = np.array([msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w])
    T = tf.transformations.quaternion_matrix(quaternion)
    T[:3, 3] = translation
    return T


def got_tuple(img_msg, cam_odom, board_odom):
    img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    body_to_world = transform_matrix_from_odom(cam_odom)
    board_to_world = transform_matrix_from_odom(board_odom)

    # Get detection from tracker
    tag_in_cam = np.eye(4).astype(datatype)
    if tracker.track(img, tag_in_cam):
        cam_to_body = np.array([[0.998634, -0.0329651, -0.0405292, 0.001775],
                                [0.0332441, 0.999428, 0.00622975, 0.0235],
                                [0.0403007, -0.00756861, 0.999159, -0.034787],
                                [0, 0, 0, 1]])
        cam_to_world = np.dot(body_to_world, cam_to_body)

        tag_in_world = np.dot(cam_to_world, tag_in_cam)
        board_in_cam = np.dot(np.linalg.inv(cam_to_world), board_to_world)

        s = tag_size/2.0
        tag_in_board = np.array(
            [[0, -1, 0, s], [1, 0, 0, s], [0, 0, 1, 0], [0, 0, 0, 1]])

        print 'tag_in_cam:: '
        print tag_in_cam
        print 'board_in_cam:: '
        print board_in_cam
        print 'tag_in_world:: '
        print tag_in_world
        print 'board_in_world:: '
        print board_to_world

        broadcaster.sendTransform(body_to_world[:3, 3],
                                  tf.transformations.quaternion_from_matrix(
                                      body_to_world),
                                  rospy.Time.now(),
                                  'body',
                                  "world")

        broadcaster.sendTransform(board_to_world[:3, 3],
                                  tf.transformations.quaternion_from_matrix(
                                      board_to_world),
                                  rospy.Time.now(),
                                  'board',
                                  "world")

        broadcaster.sendTransform(cam_to_body[:3, 3],
                                  tf.transformations.quaternion_from_matrix(
                                      cam_to_body),
                                  rospy.Time.now(),
                                  'cam',
                                  "body")

        broadcaster.sendTransform(tag_in_cam[:3, 3],
                                  tf.transformations.quaternion_from_matrix(
                                      tag_in_cam),
                                  rospy.Time.now(),
                                  'tag',
                                  "cam")

        broadcaster.sendTransform(tag_in_board[:3, 3],
                                  tf.transformations.quaternion_from_matrix(
                                      tag_in_board),
                                  rospy.Time.now(),
                                  'tag_gt',
                                  "board")

        # Now see if the 3D points projected make sense.
        tag_pts = np.array([[-s, -s, 0, 1], [s, -s, 0, 1],
                            [s, s, 0, 1], [-s, s, 0, 1]]).transpose()
        tag_pts_in_world = np.dot(
            board_to_world, np.dot(tag_in_board, tag_pts))
        tag_pts_in_cam = np.dot(np.linalg.inv(cam_to_world), tag_pts_in_world)

        projections = np.dot(K, tag_pts_in_cam[:3, :])
        projections /= projections[2]
        projections = projections[:2].transpose()

        pixels = []
        debug_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        pixels = tracker.detect_tag(img, debug_img)
        pixels = np.array(pixels).reshape(4, 2)

        # Draw these pixels
        cv2.polylines(debug_img, np.int32([projections]), 1, (0, 255, 0), 3)
        img_pub.publish(bridge.cv2_to_imgmsg(debug_img))
        # pdb.set_trace()

    # rospy.spinOnce()


topics_to_parse = ['/kinect2/qhd/image_color_rect',
                   '/kinect_one/vicon_odom', '/apriltag_27_board/vicon_odom']
use_bag = False

subs = []
subs.append(Subscriber(topics_to_parse[0], Image))
subs.append(Subscriber(topics_to_parse[1], Odometry))
subs.append(Subscriber(topics_to_parse[2], Odometry))
synchronizer = ApproximateTimeSynchronizer(subs, 10, 0.05)

synchronizer.registerCallback(got_tuple)

if use_bag:
    with rosbag.Bag(path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics_to_parse):
            if topic in topics_to_parse:
                index = topics_to_parse.index(topic)
                subs[index].signalMessage(msg)
else:
    rospy.Subscriber(topics_to_parse[0], Image,
                     lambda msg: subs[0].signalMessage(msg))
    rospy.Subscriber(topics_to_parse[1], Odometry,
                     lambda msg: subs[1].signalMessage(msg))
    rospy.Subscriber(topics_to_parse[2], Odometry,
                     lambda msg: subs[2].signalMessage(msg))
    rospy.spin()
