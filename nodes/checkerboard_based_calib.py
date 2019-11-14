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

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import glob

from geometry import SE3, se3

# A yaml constructor is for loading from a yaml node.
# This is taken from: http://stackoverflow.com/a/15942429


def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat


yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix",
                     opencv_matrix_constructor)

# A yaml representer is for dumping structs into a yaml node.
# So for an opencv_matrix type (to be compatible with c++'s FileStorage) we save the rows, cols, type and flattened-data


def opencv_matrix_representer(dumper, mat):
    mapping = {'rows': mat.shape[0], 'cols': mat.shape[1],
               'dt': 'd', 'data': mat.reshape(-1).tolist()}
    return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)


yaml.add_representer(np.ndarray, opencv_matrix_representer)

datatype = np.float32
np.set_printoptions(precision=4, suppress=True)

# path = '/media/icoderaven/Dumps/bagfiles/depth_calib/kinect_one/depth_char_1_27_05_19.bag'
# path = '/media/icoderaven/958c5fed-c873-4414-9ed4-c0662983e711/3dv_data/depth_calib/ext_calib_rs_kinect.bag'
# path = '/media/icoderaven/958c5fed-c873-4414-9ed4-c0662983e711/3dv_data/extrinsics_calib_plus_demo/ext_calib_rs_kinect_board.bag'
# path = '/media/icoderaven/SHENEXTDRIVE/calib_data/extrinsics_calib/ext_calib_kinect_small_board.bag'
# path = '/media/icoderaven/958c5fed-c873-4414-9ed4-c0662983e711/3dv_data/ext_calib_kinect_big_board.bag'
# path = '/media/icoderaven/958c5fed-c873-4414-9ed4-c0662983e711/3dv_data/icra20/ext_calib_realsense_new_small_board.bag'
path = '/media/icoderaven/Dumps/bagfiles/ext_calib_realsense_new_small_board.bag'
big_board_params = {}
big_board_params['s'] = 0.13
big_board_params['rows'] = 6
big_board_params['cols'] = 7

small_board_params = {}
small_board_params['s'] = 0.08
small_board_params['rows'] = 4
small_board_params['cols'] = 6
cam_types = ['kinect', 'realsense']

cam = cam_types[1]
params = small_board_params
use_gtsam = True
do_temporal_offset_detection = False

if use_gtsam:
    from extrinsics_calibrator.ExtrinsicCalibPyModules import CheckerboardExtrinsicCalibration
    calibrator = CheckerboardExtrinsicCalibration(
        '../config/checkerboard_extrinsics_calib.yaml')

offset_dt = 0.0
if do_temporal_offset_detection:
    offset_found = False
    offset_running = True
    buffer_size = 50
    prev_pnp = np.eye(4)
    prev_mocap = np.eye(4)
    sampled_pnp_odom = []
    sampled_pnp_timestamps = [0.0]
    sampled_mocap_odom = []
    sampled_mocap_timestamps = []
    offset_idx = 0

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
marker_pub = rospy.Publisher('/markers', Marker)

data_tuples = []
img_tuples = []
use_bag = True
visualize = True
use_camodocal_prealign = False

objp = np.zeros((cols*rows, 3), np.float32)
objp[:, :2] = s * np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
tag_pts = np.concatenate(
    (objp, np.ones((objp.shape[0], 1))), axis=1).transpose()

diffs_vector = []


def transform_matrix_from_odom(msg):
    translation = np.array(
        [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
    quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
    T = tf.transformations.quaternion_matrix(quaternion)
    T[:3, 3] = translation
    return T


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
    global prev_mocap, prev_pnp, sampled_mocap_odom, sampled_pnp_odom, sampled_pnp_timestamps, sampled_mocap_timestamps, offset_idx, offset_dt, offset_found
    img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    body_to_world = transform_matrix_from_odom(cam_odom)
    board_to_world = transform_matrix_from_odom(board_odom)

    # Check if board_to_world has flipped
    board_rel_disp = 0.0
    if len(data_tuples) > 0:
        board_rel_disp = np.linalg.norm(se3.vector_from_algebra(SE3.algebra_from_group(
            np.dot(np.linalg.inv(data_tuples[-1][2]), board_to_world))))
    if board_rel_disp > 0.1:
        tang = se3.vector_from_algebra(SE3.algebra_from_group(
            np.dot(np.linalg.inv(data_tuples[-1][2]), board_to_world)))
        print 'Board rot disp {0} trans disp {1}'.format(
            np.linalg.norm(tang[:3]), np.linalg.norm(tang[3:]))
        return

    # Get detection from tracker
    pixels = []
    debug_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")

    gray = cv2.cvtColor(debug_img, cv2.COLOR_BGR2GRAY)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    axis = np.float32([[s*cols, 0, 0], [0, s*rows, 0],
                       [0, 0, 0.5]]).reshape(-1, 3)

    ret, corners = cv2.findChessboardCorners(
        gray, (cols, rows), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    cv2.drawChessboardCorners(debug_img, (cols, rows), corners, ret)
    cv2.imshow('PreRefinement', debug_img)

    if ret == False:
        cv2.waitKey(10)
        return

    if ret == True:
        if K is None:
            print 'K not initialized yet!'
            return
        # Use a radius of half the minimum distance between corners
        min_colwise_dist = np.min(
            np.sum(np.diff(corners.reshape(rows, cols, 2), axis=0)**2, axis=2))
        min_rowwise_dist = np.min(
            np.sum(np.diff(corners.reshape(rows, cols, 2), axis=1)**2, axis=2))
        min_dist = np.sqrt(min(min_colwise_dist, min_rowwise_dist))
        radius = int(np.ceil(min_dist*0.5))

        corners2 = cv2.cornerSubPix(
            gray, corners, (radius, radius), (-1, -1), criteria)
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
                cv2.drawChessboardCorners(
                    debug_img, (cols, rows), corners2, ret)
                debug_img_msg = bridge.cv2_to_imgmsg(debug_img)
                debug_img_msg.header.frame_id = 'cam'
                img_pub.publish(debug_img_msg)
                K_msg.header.stamp = img_msg.header.stamp
                cam_info_pub.publish(K_msg)

                debug_img = draw(debug_img, corners2, imgpts)
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

            tag_in_cam_mocap_approx = np.dot(np.linalg.inv(
                cam_to_world), np.dot(board_to_world, tag_in_board))
            diff = np.dot(np.linalg.inv(tag_in_cam), tag_in_cam_mocap_approx)

            diff = se3.vector_from_algebra(SE3.algebra_from_group(diff))
            diffs_vector.append(diff)
            # I'm curious to see the projected mocap frame in the image too
            pts = np.eye(4)
            pts[3, :] = 1

            origin_in_cam = np.dot(tag_in_cam_mocap_approx, pts)
            projections = np.dot(K, origin_in_cam[:3, :])
            projections /= projections[2]
            projections = projections.astype(np.float32)
            debug_img = cv2.line(debug_img, tuple(projections[:2, 3]), tuple(
                projections[:2, 0]), (0, 0, 127), 1)
            debug_img = cv2.line(debug_img, tuple(projections[:2, 3]), tuple(
                projections[:2, 1]), (0, 127, 0), 1)
            debug_img = cv2.line(debug_img, tuple(projections[:2, 3]), tuple(
                projections[:2, 2]), (127, 0, 0), 1)
            cv2.imshow('img', debug_img)
            cv2.waitKey(10)

            if do_temporal_offset_detection:
                if not offset_found:
                    # Compute pnp odom and corresponding mocap odom
                    # We compare the z differences

                    # Get dt, if it's too large, don't append odom

                    dt = img_msg.header.stamp.to_sec() - \
                        sampled_pnp_timestamps[-1]
                    # if dt < 0.5:
                    cam_in_world_through_tag = np.dot(board_to_world,  np.dot(
                        tag_in_board, np.linalg.inv(tag_in_cam)))
                    sampled_pnp_odom.append(
                        tf.transformations.euler_from_matrix(cam_in_world_through_tag)[1])
                    # cam_in_world_through_tag[2,3])
                    # np.dot(prev_pnp, np.linalg.inv(tag_in_cam)))
                    # else:
                    # sampled_pnp_odom.append(np.eye(4))
                    sampled_pnp_timestamps.append(
                        img_msg.header.stamp.to_sec())
                    prev_pnp = tag_in_cam
                    prev_mocap = body_to_world
                    if len(sampled_pnp_odom) >= buffer_size:
                        import matplotlib.pyplot as plt
                        plt.figure()
                        sampled_mocap_odom = sampled_mocap_odom[1:]
                        sampled_pnp_odom = sampled_pnp_odom[1:]
                        sampled_mocap_timestamps = sampled_mocap_timestamps[1:]
                        sampled_pnp_timestamps = sampled_pnp_timestamps[2:]

                        # Mini optimisation to find offset
                        def mini_cost(dt):
                            sampled_odoms = np.interp(
                                np.array(sampled_pnp_timestamps) + dt, np.array(sampled_mocap_timestamps), np.array(sampled_mocap_odom))
                            return np.sum(
                                (np.array(sampled_pnp_odom) - np.array(sampled_odoms))**2)

                        sample_pts_for_mini = np.linspace(-0.1, 0.1, 200)
                        mini_costs = [mini_cost(dt)
                                      for dt in sample_pts_for_mini]

                        plt.plot(mini_costs)
                        min_idx_mini_cost = np.argmin(mini_costs)
                        print 'mini_costs min is {0} corresonding to dt of {1}'.format(
                            min_idx_mini_cost, sample_pts_for_mini[min_idx_mini_cost])
                        offset_dt = sample_pts_for_mini[min_idx_mini_cost]

                        # # Find longest sequence of continuous pnp odom
                        # discont_ids = np.array(
                        #     np.where(np.diff(sampled_pnp_timestamps) > 0.5))[0]
                        # longest_idx = np.argmax(np.diff(discont_ids))
                        # sampled_pnp_odom = sampled_pnp_odom[discont_ids[longest_idx]: discont_ids[longest_idx+1]]
                        # sampled_pnp_timestamps = sampled_pnp_timestamps[
                        #     discont_ids[longest_idx]: discont_ids[longest_idx+1]]

                        # # Okay, now let's look at a 1s buffer of mocap on either end
                        # first_mocap_idx = max(0, np.searchsorted(
                        #     sampled_mocap_timestamps, sampled_pnp_timestamps[0]) - 50)
                        # last_mocap_idx = min(len(sampled_mocap_timestamps), np.searchsorted(
                        #     sampled_mocap_timestamps, sampled_pnp_timestamps[-1], 'right') + 50)
                        # sampled_mocap_timestamps = sampled_mocap_timestamps[
                        #     first_mocap_idx:last_mocap_idx]
                        # sampled_mocap_odom = sampled_mocap_odom[first_mocap_idx:last_mocap_idx]

                        # # Use numpy's corr to determine offset
                        # corr = np.correlate(
                        #     np.array(sampled_mocap_odom) -
                        #     sampled_mocap_odom[0],
                        #     np.array(sampled_pnp_odom) - sampled_pnp_odom[0], mode='valid')

                        # # The size of corr is len(arg1) + len(arg2), and an ideal alignment
                        # # should give a value of 0 offset_idx
                        # # see https://stackoverflow.com/questions/49372282/find-the-best-lag-from-the-numpy-correlate-output

                        # # Additional check required to find argmax only in central
                        # offset_idx = corr.argmax() - (len(sampled_mocap_odom) - 1)
                        # # This is the offset in index, what does it mean for time?
                        # offset_dt = offset_idx * \
                        #     np.mean(np.diff(sampled_mocap_timestamps))
                        # # Finally, plot the aligned data. Shift mocap data.
                        # shifted_mocap_timestamps = []
                        # shifted_mocap_odom = []
                        # if offset_idx >= 0:
                        #     shifted_mocap_timestamps = sampled_mocap_timestamps[:-offset_idx]
                        #     shifted_mocap_odom = sampled_mocap_odom[:-offset_idx]
                        # else:
                        #     shifted_mocap_timestamps = sampled_mocap_timestamps[:offset_idx]
                        #     shifted_mocap_odom = sampled_mocap_odom[-offset_idx:]

                        # plt.plot(sampled_mocap_timestamps,
                        #          sampled_mocap_odom, '--', label='original_mocap')
                        # plt.plot(shifted_mocap_timestamps,
                        #          shifted_mocap_odom, label='shifted_mocap')
                        # plt.plot(sampled_pnp_timestamps,
                        #          sampled_pnp_odom, label='pnp')
                        # plt.legend()
                        # plt.figure()
                        # plt.plot(corr)
                        # plt.vlines(len(sampled_mocap_odom), 0.0, np.max(corr))

                        # print 'Offset idx is {0} and dt is {1} s'.format(
                        #     offset_idx, offset_dt)
                        # plt.vlines(len(sampled_mocap_odom) +
                        #            offset_idx, 0.0, np.max(corr), 'b')
                        plt.show()
                        pdb.set_trace()
                        offset_found = True
                    else:
                        # Only add to data_tuples until we have an offset
                        return
                elif offset_running:
                    # Don't add anything to calibrator or data_tuples
                    return

            if use_gtsam:
                body_to_board = np.dot(
                    np.linalg.inv(board_to_world), body_to_world)
                calibrator.add_measurement(
                    body_to_world, board_to_world, corners2.astype(np.float))

            data_tuples.append(
                [corners2, body_to_world, board_to_world, tag_in_cam])
            img_tuples.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))


if __name__ == "__main__":
    if params == small_board_params:
        checkerboard_topic = '/small_checkerboard/vicon_odom'
    else:
        checkerboard_topic = '/checkerboard/vicon_odom'

    if cam == 'kinect':
        topics_to_parse = ['/kinect2/qhd/image_color_rect',
                           '/kinect_one_new/vicon_odom', checkerboard_topic]
    elif cam == 'realsense':
        topics_to_parse = ['/camera/infra1/image_rect_raw',
                           '/realsense_rig_new/vicon_odom', checkerboard_topic]

    subs = []
    subs.append(Subscriber(topics_to_parse[0], Image))
    subs.append(Subscriber(topics_to_parse[1], Odometry))
    subs.append(Subscriber(topics_to_parse[2], Odometry))
    synchronizer = ApproximateTimeSynchronizer(subs, 100, 0.02)

    synchronizer.registerCallback(got_tuple)

    # See if cam_in_body exists, if not initialize to identity transform
    extrinsics_save_name = path.split('.')[0] + '_' + cam + '_cam_to_body.npy'
    board_save_name = path.split('.')[0] + '_' + cam + '_tag_to_board.npy'
    file_path = glob.glob(extrinsics_save_name)
    # if not file_path:
    with open('../config/checkerboard_extrinsics_calib.yaml', 'r') as f:
        ext_calib_node = yaml.load(f)['extrinsics_calib_params']
        cam_to_body = np.array(
            ext_calib_node['extrinsics_guess']).reshape(4, 4)
        K = np.eye(3)
        K_node = ext_calib_node['intrinsics_guess']
        K[0, 0] = K_node[0]
        K[1, 1] = K_node[1]
        K[0, 2] = K_node[3]
        K[1, 2] = K_node[4]
        K_msg = CameraInfo()
        K_msg.K = K.flatten()
        K_msg.R = np.eye(3).flatten()
        K_msg.height = ext_calib_node['height']
        K_msg.width = ext_calib_node['width']
        # CameraInfo wants k1, k2, p1, p2, k3
        K_msg.D = np.array(
            [K_node[7], K_node[8], K_node[5], K_node[6], K_node[9]])
        P = np.eye(4)
        P[:3, :3] = K
        K_msg.P = P[:3, :].flatten()
        K_msg.distortion_model = 'plumb_bob'

    print 'cam_to_body loaded {0}'.format(cam_to_body)
    # else:
    # cam_to_body = np.load(extrinsics_save_name)

    if use_bag:
        with rosbag.Bag(path, 'r') as bag:
            counter = 0
            if do_temporal_offset_detection:
                # We run through the bag twice, first time to determine temporal offset,
                # then second time applying this offset to the image data for
                # approx synchronizer
                for topic, msg, t in bag.read_messages(topics_to_parse):
                    index = topics_to_parse.index(topic)

                    if offset_found:
                        break
                    if index == 1:
                        body_to_world = transform_matrix_from_odom(msg)
                        sampled_mocap_odom.append(
                            tf.transformations.euler_from_matrix(body_to_world)[1])
                        sampled_mocap_timestamps.append(
                            msg.header.stamp.to_sec())
                        prev_mocap = body_to_world
                    # TODO: Avoid race conditions?
                    subs[index].signalMessage(msg)
                # Ok, we now have the offset! Reset approxsynchronizer
                subs = []
                subs.append(Subscriber(topics_to_parse[0], Image))
                subs.append(Subscriber(topics_to_parse[1], Odometry))
                subs.append(Subscriber(topics_to_parse[2], Odometry))
                synchronizer = ApproximateTimeSynchronizer(subs, 2, 0.01)
                synchronizer.registerCallback(got_tuple)
                # Reset data_tuples
                data_tuples = []
                img_tuples = []
                # Reset calibrator
                calibrator = CheckerboardExtrinsicCalibration(
                    '../config/checkerboard_extrinsics_calib.yaml')
                offset_running = False
                print 'Offset found, restarting!'

            for topic, msg, t in bag.read_messages(topics_to_parse):
                if topic in topics_to_parse:
                    index = topics_to_parse.index(topic)
                    if index in [1, 2]:
                        msg.header.stamp = rospy.Time(
                            msg.header.stamp.to_sec() - offset_dt)

                    subs[index].signalMessage(msg)
                    counter += 1

        print 'Done reading bag!'
        tangents = np.array([se3.vector_from_algebra(SE3.algebra_from_group(np.dot(np.linalg.inv(
            a), b))) for a, b in zip(np.array(data_tuples)[:-1, 2], np.array(data_tuples)[1:, 2])])
        rot_norms = np.linalg.norm(tangents[:, :3], axis=1)
        trans_norms = np.linalg.norm(tangents[:, 3:], axis=1)
        print 'BOARD ODOM:: Rotations Norm::{0} Std::{1} Translations Norm::{2} Std::{3}'.format(
            np.mean(rot_norms), np.std(rot_norms), np.mean(trans_norms), np.std(trans_norms))
        if use_gtsam:
            t_in_board = np.eye(4)
            cam_to_body = np.eye(4)
            landmark_pts = np.zeros((rows*cols, 3))
            cam_poses = np.zeros((len(data_tuples), 16))
            K_calib = np.zeros((3, 3))
            K_copy = K.copy()
            dist_coeffs = np.array(
                [[K_msg.D[0], K_msg.D[1], K_msg.D[2], K_msg.D[3]]])
            points_2d = np.array([t[0] for t in data_tuples])
            # print 'Calibrating camera...'
            # Uncomment if using distortion
            # result = cv2.calibrateCamera(np.array([objp]*len(data_tuples)), points_2d.reshape(len(data_tuples), rows*cols, 2), (640, 480), K_copy, dist_coeff, flags=cv2.CALIB_RATIONAL_MODEL)

            # result = cv2.calibrateCamera(np.array([objp]*len(data_tuples)), points_2d.reshape(len(data_tuples), rows*cols, 2), (640, 480), K_copy, dist_coeff, flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
            # K = result[1]
            # dist_coeffs = result[2]

            calibrator.solve(cam_to_body, t_in_board,
                             cam_poses, landmark_pts, K_calib)
            if not np.allclose(K_calib, np.zeros((3, 3))):
                K = K_calib

            for i in range(len(data_tuples)):
                # Send tf frame
                pose = cam_poses[i].reshape(4, 4).T
                broadcaster.sendTransform(
                    pose[:3, 3],
                    tf.transformations.quaternion_from_matrix(pose),
                    rospy.Time.now(),
                    'x'+str(i),
                    "board")

            broadcaster.sendTransform(
                cam_to_body[:3, 3],
                tf.transformations.quaternion_from_matrix(cam_to_body),
                rospy.Time.now(),
                'cam',
                "body")

            marker = Marker()
            marker.header.frame_id = 'board'
            marker.id = 0
            marker.type = marker.POINTS
            marker.action = marker.ADD
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            for i in range(rows*cols):
                marker.points.append(
                    Point(landmark_pts[i, 0], landmark_pts[i, 1], landmark_pts[i, 2]))
            marker_pub.publish(marker)
            pdb.set_trace()
        # Perform validation visualisations
        i = 0
        px_counter = 0
        # t_in_board = np.dot(t_in_board, SE3.group_from_algebra(
        # se3.algebra_from_vector(tag_in_board_offset)))
        # t_in_board[:3,3] += tag_in_board_offset
        error = 0

        for measurement, body_to_world, board_to_world, tag_in_cam in data_tuples:
            cam_to_board = np.dot(np.linalg.inv(
                board_to_world), np.dot(body_to_world, cam_to_body))
            landmark_pts_in_cam = np.dot(np.linalg.inv(cam_to_board), np.concatenate(
                (landmark_pts, np.ones((rows*cols, 1))), axis=1).T)
            # projections = np.dot(K, landmark_pts_in_cam[:3,:])
            # projections /= projections[2, :]
            # projections = projections[:2, :].T

            projections, jac = cv2.projectPoints(
                landmark_pts_in_cam.T[:, :3], np.zeros((3, 1)), np.zeros((3, 1)), K, dist_coeffs)
            projections.shape = (projections.shape[0], projections.shape[-1])
            debug_img = img_tuples[i]
            error += (np.sqrt(np.sum((measurement - projections)**2, axis=1))).sum()

            projections = projections.astype(int)

            for j in range(projections.shape[0]):
                # if(px_counter in inliers):
                cv2.circle(
                    debug_img, (projections[j, 0], projections[j, 1]), 5, (0, 255, 0), 2)
                cv2.circle(
                    debug_img, (measurement[j, 0], measurement[j, 1]), 5, (0, 0, 255), 2)
                cv2.line(debug_img, (measurement[j, 0], measurement[j, 1]),
                         (projections[j, 0], projections[j, 1]), (0, 255, 0))
                # px_counter += 1

            pts = np.eye(4)
            pts[3, :] = 1
            origin_in_cam = np.dot(tag_in_cam, pts)
            projections = np.dot(K, origin_in_cam[:3, :])
            projections /= projections[2]
            projections = projections.astype(np.float32)
            debug_img = cv2.line(debug_img, tuple(projections[:2, 3]), tuple(
                projections[:2, 0]), (0, 0, 127), 1)
            debug_img = cv2.line(debug_img, tuple(projections[:2, 3]), tuple(
                projections[:2, 1]), (0, 127, 0), 1)
            debug_img = cv2.line(debug_img, tuple(projections[:2, 3]), tuple(
                projections[:2, 2]), (127, 0, 0), 1)

            # origin_in_cam = np.dot(board_in_cam, pts)
            # projections = np.dot(K, origin_in_cam[:3, :])
            # projections /= projections[2]
            # projections = projections.astype(np.float32)
            # debug_img = cv2.line(debug_img, tuple(projections[:2, 3]), tuple(
            #     projections[:2, 0]), (0, 0, 127), 1)
            # debug_img = cv2.line(debug_img, tuple(projections[:2, 3]), tuple(
            #     projections[:2, 1]), (0, 127, 0), 1)
            # debug_img = cv2.line(debug_img, tuple(projections[:2, 3]), tuple(
            #     projections[:2, 2]), (127, 0, 0), 1)

            cv2.imshow('Validation ({0}/{1})'.format(i,
                                                     len(data_tuples)-1), debug_img)
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
            # pdb.set_trace()
            i += 1
        print 'Final error is {0}'.format(error)
        if raw_input('Save? y/n') in ['y', 'Y']:
            print 'saving to '+extrinsics_save_name
            np.save(extrinsics_save_name, cam_to_body)
            print 'saving to '+board_save_name
            np.save(board_save_name, t_in_board)

    else:
        rospy.Subscriber(topics_to_parse[0], Image,
                         lambda msg: subs[0].signalMessage(msg))
        rospy.Subscriber(topics_to_parse[1], Odometry,
                         lambda msg: subs[1].signalMessage(msg))
        rospy.Subscriber(topics_to_parse[2], Odometry,
                         lambda msg: subs[2].signalMessage(msg))

        rospy.spin()
