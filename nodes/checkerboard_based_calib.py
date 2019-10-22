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
path = '/home/icoderaven/ext_calib_realsense_new_small_board.bag'
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

if use_gtsam:
    from extrinsics_calibrator.ExtrinsicCalibPyModules import CheckerboardExtrinsicCalibration
    calibrator = CheckerboardExtrinsicCalibration(
        '../config/checkerboard_extrinsics_calib.yaml')

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


def cost_function(theta):
    body_to_c = SE3.group_from_algebra(
        se3.algebra_from_vector(theta[:6]))

    # tag_in_board_offset = theta[6:]
    t_in_board = tag_in_board.copy()
    # t_in_board[:3,3] += tag_in_board_offset
    # t_in_board = np.dot(t_in_board, SE3.group_from_algebra(
    # se3.algebra_from_vector(tag_in_board_offset)))
    error = 0
    img_count = 0
    residual_vectors = []
    for measurement, body_to_world, board_to_world, tag_in_cam in data_tuples:
        tag_pts_in_world = np.dot(
            board_to_world, np.dot(t_in_board, tag_pts))
        tag_pts_in_body = np.dot(np.linalg.inv(
            body_to_world), tag_pts_in_world)
        tag_pts_in_cam = np.dot(body_to_c, tag_pts_in_body)

        projections, jac = cv2.projectPoints(
            tag_pts_in_body.T[:, :3], theta[:3], theta[3:6], K, np.zeros((1, 4)))
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
        error += (np.sqrt(np.sum((measurement - projections)**2, axis=1))).sum()
        residual_vectors.append((measurement - projections).ravel())
        # np.linalg.norm(measurement - projections)
        # error += np.sum((measurement - projections), axis=0)
    return error  # np.array(residual_vectors).ravel()


def jac_function(theta):
    body_to_c = SE3.group_from_algebra(
        se3.algebra_from_vector(theta[:6]))

    # tag_in_board_offset = theta[6:]
    t_in_board = tag_in_board.copy()
    # t_in_board[:3,3] += tag_in_board_offset
    # t_in_board = np.dot(t_in_board, SE3.group_from_algebra(
    # se3.algebra_from_vector(tag_in_board_offset)))
    error = 0
    img_count = 0
    jacs = []
    for measurement, body_to_world, board_to_world, tag_in_cam in data_tuples:
        tag_pts_in_world = np.dot(
            board_to_world, np.dot(t_in_board, tag_pts))
        tag_pts_in_body = np.dot(np.linalg.inv(
            body_to_world), tag_pts_in_world)
        tag_pts_in_cam = np.dot(body_to_c, tag_pts_in_body)

        projections, jac = cv2.projectPoints(
            tag_pts_in_body.T[:, :3], theta[:3], theta[3:6], K, np.zeros((1, 4)))

        jacs.append(jac[:, :6])
    return np.vstack(np.array(jacs))


def meta_cost_function(t_offset_theta, use_ransac=False):
    points_3d = []
    points_2d = []

    delta = np.eye(4)
    delta[:3, :3] = cv2.Rodrigues(t_offset_theta[:3])[0]
    delta[:3, 3] = t_offset_theta[3:]

    t_in_board = np.dot(tag_in_board, delta)
    body_to_cam = np.linalg.inv(cam_to_body)
    error = 0
    for measurement, body_to_world, board_to_world, tag_in_cam in data_tuples:
        board_in_body = np.dot(np.linalg.inv(
            body_to_world), board_to_world)
        board_in_cam = np.dot(body_to_cam, board_in_body)
        tag_in_cam = np.dot(board_in_cam, t_in_board)
        tag_pts_in_cam = np.dot(tag_in_cam, tag_pts)[:3, :]

        measurement.shape = (measurement.shape[0], measurement.shape[-1])

        points_3d.append(tag_pts_in_cam)
        points_2d.append(measurement)

    points_3d = np.concatenate(points_3d, axis=1).astype(np.float32).T
    points_2d = np.concatenate(points_2d, axis=0).astype(np.float32)
    D = np.array([[0, 0, 0, 0]], dtype=np.float32)

    # Refine K
    K_copy = K.copy()
    # result = cv2.calibrateCamera(np.array([objp]*30), points_2d.reshape(30, 42, 2), (960, 540), K_copy, np.array(
    #     [[0, 0, 0, 0]], dtype=np.float32), flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    # K_copy = result[1]
    # D = result[2]
    if use_ransac:
        result = cv2.solvePnPRansac(points_3d, points_2d, K_copy, D)
    else:
        result = cv2.solvePnP(points_3d, points_2d, K_copy, D)
    rv = result[1]
    tv = result[2]
    error = 0
    for measurement, body_to_world, board_to_world, tag_in_cam in data_tuples:

        tag_pts_in_world = np.dot(
            board_to_world, np.dot(t_in_board, tag_pts))
        tag_pts_in_body = np.dot(np.linalg.inv(
            body_to_world), tag_pts_in_world)
        tag_pts_in_body = tag_pts_in_body[:3, :]
        measurement.shape = (measurement.shape[0], measurement.shape[-1])
        projections, jac = cv2.projectPoints(
            tag_pts_in_body.T[:, :3], rv, tv, K, np.zeros((1, 4)))
        projections.shape = (projections.shape[0], projections.shape[-1])

        error += (np.sqrt(np.sum((measurement - projections)**2, axis=1))).sum()

    return result, error


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
    axis = np.float32([[s*cols, 0, 0], [0, s*rows, 0],
                       [0, 0, 0.5]]).reshape(-1, 3)

    ret, corners = cv2.findChessboardCorners(
        gray, (cols, rows), None, cv2.CALIB_CB_FILTER_QUADS)
    cv2.drawChessboardCorners(debug_img, (cols, rows), corners, ret)
    cv2.imshow('PreRefinement', debug_img)

    if ret == False:
        print 'yikes!'
        cv2.waitKey(-1)
        return

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

            if use_gtsam:
                body_to_board = np.dot(
                    np.linalg.inv(board_to_world), body_to_world)
                # calibrator.add_measurement(body_to_board.astype(
                #     np.float), corners2.astype(np.float))
                # calibrator.add_measurement(body_to_world.astype(
                #     np.float), corners2.astype(np.float))
                calibrator.add_measurement(
                    body_to_world, board_to_world, corners2.astype(np.float))

            if visualize:
                cv2.drawChessboardCorners(
                    debug_img, (cols, rows), corners2, ret)
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

            tag_in_cam_mocap_approx = np.dot(np.linalg.inv(
                cam_to_world), np.dot(board_to_world, tag_in_board))
            diff = np.dot(np.linalg.inv(tag_in_cam), tag_in_cam_mocap_approx)

            diff = se3.vector_from_algebra(SE3.algebra_from_group(diff))
            diffs_vector.append(diff)
            print diff
            print np.linalg.norm(diff[:3])
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
                           '/kinect_one_new/vicon_odom', checkerboard_topic, '/kinect2/qhd/camera_info']
    elif cam == 'realsense':
        topics_to_parse = ['/camera/color/image_raw',
                           '/realsense_rig_new/vicon_odom', checkerboard_topic, '/camera/color/camera_info']

    subs = []
    subs.append(Subscriber(topics_to_parse[0], Image))
    subs.append(Subscriber(topics_to_parse[1], Odometry))
    subs.append(Subscriber(topics_to_parse[2], Odometry))
    synchronizer = ApproximateTimeSynchronizer(subs, 10, 0.1)

    synchronizer.registerCallback(got_tuple)

    # See if cam_in_body exists, if not initialize to identity transform
    extrinsics_save_name = path.split('.')[0] + '_' + cam + '_cam_to_body.npy'
    board_save_name = path.split('.')[0] + '_' + cam + '_tag_to_board.npy'
    file_path = glob.glob(extrinsics_save_name)
    # if not file_path:
    with open('../config/checkerboard_extrinsics_calib.yaml', 'r') as f:
        cam_to_body = np.array(
            yaml.load(f)['extrinsics_calib_params']['extrinsics_guess']).reshape(4, 4)
    print 'cam_to_body loaded {0}'.format(cam_to_body)
    # else:
    # cam_to_body = np.load(extrinsics_save_name)

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
        if use_gtsam:
            t_in_board = np.eye(4)
            cam_to_body = np.eye(4)
            landmark_pts = np.zeros((rows*cols, 3))
            cam_poses = np.zeros((len(data_tuples), 16))
            K_calib = np.zeros((3,3))
            K_copy = K.copy()
            dist_coeff = np.array([[0, 0, 0, 0]], dtype=np.float32)
            points_2d = np.array([t[0] for t in data_tuples])
            # Uncomment if using distortion
            # result = cv2.calibrateCamera(np.array([objp]*len(data_tuples)), points_2d.reshape(len(data_tuples), rows*cols, 2), (640, 480), K_copy, dist_coeff, flags=cv2.CALIB_RATIONAL_MODEL)

            result = cv2.calibrateCamera(np.array([objp]*len(data_tuples)), points_2d.reshape(len(data_tuples), rows*cols, 2), (640, 480), K_copy, dist_coeff, flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)

            calibrator.solve(cam_to_body, t_in_board, cam_poses, landmark_pts, K_calib)
            print 'In python K is '
            print K_calib
            if not np.allclose(K_calib , np.zeros((3,3))):
                K = K_calib
            K = result[1]
            dist_coeffs = result[2]
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
        else:
            # Try to use a black box optimizer
            print 'Starting optimization...'
            from scipy.optimize import minimize, least_squares
            # Since initial guess is pretty close to identity
            initial_guess = np.array([0.0, 0, 0, 0, 0, 0])
            # if use_camodocal_prealign:
            #     # Save the tuple pairs in opencv format
            #     # example
            #     with open('TransformPairsInput.yml', 'w') as f:
            #         f.write("%YAML:1.0\n")
            #         f.write("frameCount: " + str(len(data_tuples)) + "\n")
            #         ind = 0
            #         for measurement, body_to_world, board_to_world, tag_in_cam in data_tuples:
            #             tag_in_world = np.dot(
            #                 board_to_world, tag_in_board)
            #             tag_in_body = np.dot(np.linalg.inv(
            #                 body_to_world), tag_in_world)
            #             yaml.dump({"T1_"+str(ind): tag_in_cam,
            #                        "T2_"+str(ind): tag_in_body}, f)
            #             ind += 1
            #     print 'Written to TransformPairsInput.yml, replace in calib package and continue when CalibratedTransform.yml is copied into cwd...'
            #     pdb.set_trace()
            #     with open('CalibratedTransform.yml', 'r') as f:
            #         f.readline()
            #         T_init_guess = yaml.load(
            #             f)['ArmTipToMarkerTagTransform'].reshape(4, 4)
            #         # T_init_guess = np.linalg.inv(T_init_guess)
            #         rvec = cv2.Rodrigues(T_init_guess[:3, :3])[0]
            #         tvec = T_init_guess[:3, 3]
            #     initial_guess = se3.vector_from_algebra(
            #         SE3.algebra_from_group(np.linalg.inv(T_init_guess)))
            #     pdb.set_trace()

            # The delta transform can just be solved by using solvePnP again
            # result = minimize(lambda x: meta_cost_function(x)[1], initial_guess)
            result, error = meta_cost_function(initial_guess, True)

            # res_lsq = least_squares(cost_function, initial_guess)
            # rvec = res_lsq.x[:3]
            # tvec = res_lsq.x[3:]
            # pdb.set_trace()

            # result = minimize(cost_function, initial_guess)
            # pdb.set_trace()
            # pdb.set_trace()
            # rvec = result.x[:3]
            # tvec = result.x[3:]
            # print 'error was {0}'.format(error)
            rvec = result[1]
            tvec = result[2]
            inliers = result[3]
            body_to_cam = np.eye(4)
            body_to_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
            body_to_cam[:3, 3] = tvec.ravel()
            cam_to_body = np.linalg.inv(body_to_cam)

            # print 'Done, results is'
            # print rvec, tvec

            # tag_in_board_offset = result.x[6:]
            print 'cam_in_body found was {0}'.format(cam_to_body)

            # result = minimize(lambda x: meta_cost_function(x)
            #                   [1], np.zeros((6, 1)))
            # delta = np.eye(4)
            # delta[:3, :3] = cv2.Rodrigues(result.x[:3])[0]
            # delta[:3, 3] = result.x[3:]

            # t_in_board = np.dot(tag_in_board, delta)
            # print 'tag_in_board was {0}'.format(t_in_board)
            # print tag_in_board_offset
            t_in_board = tag_in_board.copy()

        # Perform validation visualisations
        i = 0
        px_counter = 0
        # t_in_board = np.dot(t_in_board, SE3.group_from_algebra(
        # se3.algebra_from_vector(tag_in_board_offset)))
        # t_in_board[:3,3] += tag_in_board_offset
        error = 0

        for measurement, body_to_world, board_to_world, tag_in_cam in data_tuples:
            cam_to_board = np.dot( np.linalg.inv(board_to_world) ,np.dot(body_to_world, cam_to_body))
            landmark_pts_in_cam = np.dot( np.linalg.inv(cam_to_board), np.concatenate((landmark_pts, np.ones((rows*cols,1))), axis=1).T)
            # projections = np.dot(K, landmark_pts_in_cam[:3,:])
            # projections /= projections[2, :]
            # projections = projections[:2, :].T

            projections, jac = cv2.projectPoints(
                landmark_pts_in_cam.T[:, :3], np.zeros((3,1)), np.zeros((3,1)) , K, dist_coeffs)
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
