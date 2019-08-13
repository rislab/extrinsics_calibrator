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
from sensor_msgs.msg import CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

from apriltag_tracker._AprilTagTracker import AprilTagTracker
from apriltag_tracker.msg import Apriltags

import glob

from geometry import SE3, se3

datatype = np.float32
np.set_printoptions(precision=4, suppress=True)

# path = '/media/icoderaven/Dumps/bagfiles/depth_calib/kinect_one/depth_char_1_27_05_19.bag'
# path = '/media/icoderaven/958c5fed-c873-4414-9ed4-c0662983e711/3dv_data/depth_calib/ext_calib_rs_kinect.bag'
# path = '/media/icoderaven/958c5fed-c873-4414-9ed4-c0662983e711/3dv_data/extrinsics_calib_plus_demo/ext_calib_rs_kinect_board.bag'
# path = '/media/icoderaven/SHENEXTDRIVE/calib_data/extrinsics_calib/ext_calib_kinect_small_board.bag'
path = '/media/icoderaven/958c5fed-c873-4414-9ed4-c0662983e711/3dv_data/ext_calib_kinect_big_board.bag'
big_board_params = {}
big_board_params['s'] = 0.13
big_board_params['rows'] = 6
big_board_params['cols'] = 7

small_board_params = {}
small_board_params['s'] = 0.08
small_board_params['rows'] = 4
small_board_params['cols'] = 6
cam_types = ['kinect', 'realsense']

cam = cam_types[0]
params = big_board_params

K = None
K_msg = None
cam_to_body = None

# We work with the assumption that the mocap frame for the checkerboard is
# coincident with the frame for PnP. Visualise on RViz to confirm
s = params['s']
rows = params['rows']
cols = params['cols']

tag_in_board = np.array(
    [[0.0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

rospy.init_node('mocap_cam_extrinsic_calibrator')
bridge = CvBridge()
broadcaster = tf.TransformBroadcaster()
img_pub = rospy.Publisher('/cam/debug_img', Image)
cam_info_pub = rospy.Publisher('/cam/camera_info', CameraInfo)

data_tuples = []
img_tuples = []
use_bag = True
visualize = True

objp = np.zeros((cols*rows, 3), np.float32)
objp[:, :2] = s * np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

diffs_vector = []


def transform_matrix_from_odom(msg):
    translation = np.array(
        [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
    quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
    T = tf.transformations.quaternion_matrix(quaternion)
    T[:3, 3] = translation
    return T


def cost_function(c_to_body_log_plus_offset):
    c_to_body = SE3.group_from_algebra(
        se3.algebra_from_vector(c_to_body_log_plus_offset[:6]))
    tag_in_board_offset = c_to_body_log_plus_offset[6:]
    t_in_board = tag_in_board.copy()
    # t_in_board[:3,3] += tag_in_board_offset
    t_in_board = np.dot(t_in_board, SE3.group_from_algebra(
        se3.algebra_from_vector(tag_in_board_offset)))
    error = 0
    img_count = 0
    for measurement, body_to_world, board_to_world, tag_in_cam in data_tuples:
        cam_to_world = np.dot(body_to_world, c_to_body)
        tag_pts = np.concatenate(
            (objp, np.ones((objp.shape[0], 1))), axis=1).transpose()
        tag_pts_in_world = np.dot(
            board_to_world, np.dot(t_in_board, tag_pts))
        tag_pts_in_cam = np.dot(np.linalg.inv(cam_to_world), tag_pts_in_world)

        projections = np.dot(K, tag_pts_in_cam[:3, :])
        projections /= projections[2]
        projections = projections[:2].transpose()

        projections.shape = (projections.shape[0], 1, projections.shape[1])
        projections = projections.astype(np.float32)

        # cv2.drawChessboardCorners(debug_img, (rows, cols), projections, True)

        projections.shape = (projections.shape[0], projections.shape[-1])
        measurement.shape = (measurement.shape[0], measurement.shape[-1])

        if img_count == 0:
            debug_img = img_tuples[img_count].copy()
            for i in range(projections.shape[0]):
                cv2.circle(
                    debug_img, (projections[i, 0], projections[i, 1]), 2, (255, 0, 0), 1)
                cv2.circle(
                    debug_img, (measurement[i, 0], measurement[i, 1]), 2, (0, 0, 255), 1)
                cv2.line(debug_img, (measurement[i, 0], measurement[i, 1]),
                         (projections[i, 0], projections[i, 1]), (0, 255, 0))
            cv2.imshow('cost function img', debug_img)
            cv2.waitKey(10)
            # pdb.set_trace()

        img_count += 1
        # np.linalg.norm(measurement - projections)
        error += np.sum((measurement - projections) ** 2)
    return error


buffer_size = 100


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    try:
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 5)
    except:
        print 'oof'
        pdb.set_trace()
    return img
    


def got_tuple(img_msg, cam_odom, board_odom):
    img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    body_to_world = transform_matrix_from_odom(cam_odom)
    board_to_world = transform_matrix_from_odom(board_odom)

    # Get detection from tracker
    pixels = []
    debug_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")

    gray = cv2.cvtColor(debug_img, cv2.COLOR_BGR2GRAY)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    axis = np.float32([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]).reshape(-1, 3)

    ret, corners = cv2.findChessboardCorners(
        gray, (cols, rows), None, cv2.CALIB_CB_FILTER_QUADS)
    cv2.drawChessboardCorners(debug_img, (cols, rows), corners, ret)
    cv2.imshow('PreRefinement', debug_img)

    if ret == True:
        if K is None:
            print 'K not initialized yet!'
            return
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        corners2.shape = (corners2.shape[0], corners2.shape[-1])
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv2.solvePnP(
            objp, corners2, K, np.array([[0, 0, 0, 0]], dtype=np.float32))
        if ret == True:
            rodRotMat = cv2.Rodrigues(rvecs)
            tag_in_cam = np.eye(4)
            tag_in_cam[:3, :3] = rodRotMat[0]
            tag_in_cam[:3, 3] = tvecs[:, 0]
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(
                axis, rvecs, tvecs, K, np.zeros((1, 4)))
            if visualize:
                cv2.drawChessboardCorners(debug_img, (cols, rows), corners2, ret)
                img_msg = bridge.cv2_to_imgmsg(debug_img)
                img_msg.header.frame_id = 'cam'
                img_pub.publish(img_msg)
                K_msg.header.stamp = img_msg.header.stamp
                cam_info_pub.publish(K_msg)

                debug_img = draw(debug_img, corners2, imgpts)
                
                print ret

                cam_to_world = np.dot(body_to_world, cam_to_body)

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
                # tag_in_cam = np.eye(4).astype(datatype)
                # # Now see if the 3D points projected make sense.

                # tag_pts = np.concatenate((objp, np.ones((objp.shape[0], 1))), axis=1).transpose()
                # tag_pts_in_world = np.dot(
                #     board_to_world, np.dot(tag_in_board, tag_pts))
                # tag_pts_in_cam = np.dot(np.linalg.inv(cam_to_world), tag_pts_in_world)

                # projections = np.dot(K, tag_pts_in_cam[:3, :])
                # projections /= projections[2]
                # projections = projections[:2].transpose()

                # pixels = []
                # Draw these pixels
                
                
                # pdb.set_trace()
            tag_in_cam_mocap_approx = np.dot(np.linalg.inv(cam_to_world), np.dot(board_to_world, tag_in_board))
            diff = np.dot(np.linalg.inv(tag_in_cam), tag_in_cam_mocap_approx)

            diff = se3.vector_from_algebra(SE3.algebra_from_group(diff))
            diffs_vector.append(diff)
            print diff
            print np.linalg.norm(diff[:3])
            # I'm curious to see the projected mocap frame in the image too
            pts = np.eye(4)
            pts[3,:] = 1

            origin_in_cam = np.dot(tag_in_cam_mocap_approx, pts)
            projections = np.dot(K, origin_in_cam[:3, :])
            projections /= projections[2]
            projections = projections.astype(np.float32)
            debug_img = cv2.line(debug_img, tuple(projections[:2,3]), tuple(projections[:2,0]), (0, 0, 127), 1)
            debug_img = cv2.line(debug_img, tuple(projections[:2,3]), tuple(projections[:2,1]), (0, 127, 0), 1)
            debug_img = cv2.line(debug_img, tuple(projections[:2,3]), tuple(projections[:2,2]), (127, 0, 0), 1)
            cv2.imshow('img', debug_img)
            cv2.waitKey(-1)

            data_tuples.append([corners2, body_to_world, board_to_world, tag_in_cam])
            img_tuples.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

if __name__ == "__main__":
    if params == small_board_params:
        checkerboard_topic = '/small_checkerboard/vicon_odom'
    else:
        checkerboard_topic = '/checkerboard/vicon_odom'

    if cam == 'kinect':
        topics_to_parse = ['/kinect2/qhd/image_color_rect',
                        '/kinect_one/vicon_odom', checkerboard_topic, '/kinect2/qhd/camera_info']
    elif cam == 'realsense':
        topics_to_parse = ['/camera/color/image_raw',
                        '/realsense_rig/vicon_odom', checkerboard_topic, '/camera/color/camera_info']

    subs = []
    subs.append(Subscriber(topics_to_parse[0], Image))
    subs.append(Subscriber(topics_to_parse[1], Odometry))
    subs.append(Subscriber(topics_to_parse[2], Odometry))
    synchronizer = ApproximateTimeSynchronizer(subs, 1, 0.05)

    synchronizer.registerCallback(got_tuple)

    # See if cam_in_body exists, if not initialize to identity transform
    save_name = path.split('.')[0] + '_' + cam + '_cam_to_body.npy'
    file_path = glob.glob(save_name)
    if not file_path:
        cam_to_body = np.eye(4)
    else:
        cam_to_body = np.load(save_name)

    if use_bag:
        with rosbag.Bag(path, 'r') as bag:
            counter = 0
            for topic, msg, t in bag.read_messages(topics_to_parse):
                if topic in topics_to_parse:
                    index = topics_to_parse.index(topic)
                    if topic in ['/kinect2/qhd/camera_info', '/camera/color/camera_info']:
                        # Assign the K here
                        K = np.array(msg.K).astype(datatype).reshape(3, 3)
                        print 'K is {0}'.format(K)
                        K_msg = msg
                        K_msg.header.frame_id = 'cam'
                        del topics_to_parse[index]
                    else:
                        subs[index].signalMessage(msg)
                        counter += 1
                        if counter % 1000 == 0:
                            print 'Read {0} tuples'.format(counter)

        # Try to use a black box optimizer
        print 'Starting optimization...'
        from scipy.optimize import minimize
        # Since initial guess is pretty close to identity
        initial_guess = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # pdb.set_trace()
        result = minimize(cost_function, initial_guess, bounds=np.array(
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-0.1,0.1], [-0.1,0.1], [-0.1,0.1], [-0.1,0.1], [-0.1,0.1], [-0.1,0.1]]), method = 'L-BFGS-B')
        print 'Done, results is'
        print result
        cam_to_body = SE3.group_from_algebra(se3.algebra_from_vector(result.x[:6]))
        tag_in_board_offset = result.x[6:]
        print cam_to_body
        print tag_in_board_offset

        # Perform validation visualisations
        i = 0
        t_in_board = tag_in_board.copy()
        t_in_board = np.dot(t_in_board, SE3.group_from_algebra(
        se3.algebra_from_vector(tag_in_board_offset)))
        # t_in_board[:3,3] += tag_in_board_offset
        for measurement, body_to_world, board_to_world, tag_in_cam in data_tuples:
            cam_to_world = np.dot(body_to_world, cam_to_body)
            tag_pts = np.concatenate(
                (objp, np.ones((objp.shape[0], 1))), axis=1).transpose()
            tag_pts_in_world = np.dot(
                board_to_world, np.dot(t_in_board, tag_pts))
            tag_pts_in_cam = np.dot(np.linalg.inv(cam_to_world), tag_pts_in_world)

            projections = np.dot(K, tag_pts_in_cam[:3, :])
            projections /= projections[2]
            projections = projections[:2].transpose()

            debug_img = img_tuples[i]
            projections.shape = (projections.shape[0], 1, projections.shape[1])
            projections = projections.astype(np.float32)

            cv2.drawChessboardCorners(debug_img, (rows, cols), projections, True)
            cv2.imshow('validation', debug_img)
            img_msg = bridge.cv2_to_imgmsg(debug_img)
            img_msg.header.frame_id = 'cam'
            img_pub.publish(img_msg)
            K_msg.header.stamp = img_msg.header.stamp
            cam_info_pub.publish(K_msg)
            
            # And the tfs

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
            
            
            cv2.waitKey(-1)
            
            i += 1

        if raw_input('Save? y/n') in ['y', 'Y']:
            print 'saving to '+save_name
            np.save(save_name, cam_to_body)

    else:
        rospy.Subscriber(topics_to_parse[0], Image,
                        lambda msg: subs[0].signalMessage(msg))
        rospy.Subscriber(topics_to_parse[1], Odometry,
                        lambda msg: subs[1].signalMessage(msg))
        rospy.Subscriber(topics_to_parse[2], Odometry,
                        lambda msg: subs[2].signalMessage(msg))

        rospy.spin()
