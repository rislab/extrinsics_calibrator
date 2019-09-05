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

from geometry import SE3, se3

datatype = np.float32
np.set_printoptions(precision=4, suppress=True)


config_file = '../config/extrinsics_calib.yaml'

with open(config_file, 'r') as f:
    file_node = yaml.load(f)
    node = file_node['apriltag_tracker_params']
    tag_size = node['tag_size']
    s = tag_size/2.0
    K = np.array(node['K']).reshape(3, 3)
    path = file_node['extrinsics_calib_params']['bag_file']

tracker = AprilTagTracker(config_file)

rospy.init_node('broadcaster')
bridge = CvBridge()
broadcaster = tf.TransformBroadcaster()
img_pub = rospy.Publisher('debug_img', Image)

data_tuples = []
use_bag = True
visualize = True

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

def cost_function( cam_to_body_log):
    cam_to_body = SE3.group_from_algebra(se3.algebra_from_vector(cam_to_body_log))
    error = 0
    for measurement, body_to_world, board_to_world in data_tuples:
        cam_to_world = np.dot(body_to_world, cam_to_body)
        tag_pts = np.array([[-s, -s, 0, 1], [s, -s, 0, 1],
                                [s, s, 0, 1], [-s, s, 0, 1]]).transpose()
        tag_in_board = np.array(
                [[0, -1, 0, s], [1, 0, 0, s], [0, 0, 1, 0], [0, 0, 0, 1]])
        tag_pts_in_world = np.dot(
            board_to_world, np.dot(tag_in_board, tag_pts))
        tag_pts_in_cam = np.dot(np.linalg.inv(cam_to_world), tag_pts_in_world)

        projections = np.dot(K, tag_pts_in_cam[:3, :])
        projections /= projections[2]
        projections = projections[:2].transpose()

        error += np.linalg.norm(measurement - projections)
    return error

buffer_size = 100

def cost_function_tuple_offset( params):
    cam_to_body = SE3.group_from_algebra(se3.algebra_from_vector(params[:6]))
    tuple_offset = int(params[6]*100)
    # Use a central region of data tuples +- 100 
    # The offset determines the start of the measurement offset
    # pdb.set_trace()
    # measurements_offset = data_tuples[buffer_size + tuple_offset: -buffer_size + tuple_offset, 0]
    # bodys_to_world_tuple_offset = data_tuples[buffer_size:-buffer_size, 1]
    # boards_to_world_tuple_offset = data_tuples[buffer_size:-buffer_size, 2]
    # offset_tuples = np.concatenate(measurements_offset, bodys_to_world_offset, boards_to_world_offset, axis=1)

    error = 0
    for i in range(len(data_tuples) - buffer_size*2):
        measurement = data_tuples[i + buffer_size + tuple_offset][0]
        body_to_world = data_tuples[i + buffer_size][1]
        board_to_world = data_tuples[i + buffer_size][2]

        cam_to_world = np.dot(body_to_world, cam_to_body)
        tag_pts = np.array([[-s, -s, 0, 1], [s, -s, 0, 1],
                                [s, s, 0, 1], [-s, s, 0, 1]]).transpose()
        tag_in_board = np.array(
                [[0, -1, 0, s], [1, 0, 0, s], [0, 0, 1, 0], [0, 0, 0, 1]])
        tag_pts_in_world = np.dot(
            board_to_world, np.dot(tag_in_board, tag_pts))
        tag_pts_in_cam = np.dot(np.linalg.inv(cam_to_world), tag_pts_in_world)

        projections = np.dot(K, tag_pts_in_cam[:3, :])
        projections /= projections[2]
        projections = projections[:2].transpose()

        error += np.linalg.norm(measurement - projections)
    return error


def got_tuple(img_msg, cam_odom, board_odom):
    img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    body_to_world = transform_matrix_from_odom(cam_odom)
    board_to_world = transform_matrix_from_odom(board_odom)

    # Get detection from tracker
    pixels = []
    debug_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    pixels = tracker.detect_tag(img, debug_img)
    pixels = np.array(pixels)
    if pixels.shape[0] > 0:
        pixels = pixels.reshape(4, 2)
        data_tuples.append([pixels, body_to_world, board_to_world])

    # Get detection from tracker
    if visualize:
        tag_in_cam = np.eye(4).astype(datatype)
        if tracker.track(img, tag_in_cam):
            # cam_to_body = np.array([[0.998634, -0.0329651, -0.0405292, 0.013017],#0.001775],
            #                         [0.0332441, 0.999428, 0.00622975, 0.00547],#0.0235],
            #                         [0.0403007, -0.00756861, 0.999159, -0.0230],#-0.034787],
            #                         [0, 0, 0, 1]])
            cam_to_body = np.load('cam_in_body.npy')
            cam_to_world = np.dot(body_to_world, cam_to_body)

            tag_in_world = np.dot(cam_to_world, tag_in_cam)
            board_in_cam = np.dot(np.linalg.inv(cam_to_world), board_to_world)

            tag_in_board = np.array(
                [[0, -1, 0, s], [1, 0, 0, s], [0, 0, 1, 0], [0, 0, 0, 1]])

            # print 'tag_in_cam:: '
            # print tag_in_cam
            # print 'board_in_cam:: '
            # print board_in_cam
            # print 'tag_in_world:: '
            # print tag_in_world
            # print 'board_in_world:: '
            # print board_to_world

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


topics_to_parse = ['/kinect2/qhd/image_color_rect',
                   '/kinect_one/vicon_odom', '/apriltag_27_board/vicon_odom']

subs = []
subs.append(Subscriber(topics_to_parse[0], Image))
subs.append(Subscriber(topics_to_parse[1], Odometry))
subs.append(Subscriber(topics_to_parse[2], Odometry))
synchronizer = ApproximateTimeSynchronizer(subs, 10, 0.05)

synchronizer.registerCallback(got_tuple)

if use_bag:
    with rosbag.Bag(path, 'r') as bag:
        counter = 0
        for topic, msg, t in bag.read_messages(topics_to_parse):
            if topic in topics_to_parse:
                index = topics_to_parse.index(topic)
                subs[index].signalMessage(msg)
                counter += 1
                if counter%1000 == 0:
                    print 'Read {0} tuples'.format(counter)
    
    # Try to use a black box optimizer
    print 'Starting optimization...'
    from scipy.optimize import minimize
    initial_guess = np.array([0,0,0,0,0,0,-0.1]) # Since initial guess is pretty close to unity
    result = minimize(cost_function_tuple_offset, initial_guess, bounds=np.array([[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1], [-1, 1]]))
    print 'Done, results is'
    print result
    print SE3.group_from_algebra(se3.algebra_from_vector(result.x[:6]))
    print result.x[6]
    pdb.set_trace()

else:
    rospy.Subscriber(topics_to_parse[0], Image,
                     lambda msg: subs[0].signalMessage(msg))
    rospy.Subscriber(topics_to_parse[1], Odometry,
                     lambda msg: subs[1].signalMessage(msg))
    rospy.Subscriber(topics_to_parse[2], Odometry,
                     lambda msg: subs[2].signalMessage(msg))
    rospy.spin()
