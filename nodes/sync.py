#!/usr/bin/env python

import cv2
import sys
import select
import os
import readline
import rosbag
import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

from pynput.keyboard import Key, Listener


class Synchronize:
    def __init__(self, topics, out_filename):
        self.subs = []
        self.out_bag = rosbag.Bag(out_filename, "w")
        self.topics = topics
        self.bridge = CvBridge()
        self.data_count = 0
        self.latest_sync_tuple = []
        self.updated = False
        self.init_sub()

    def spin(self):
        rospy.Subscriber(self.topics[0], Image,
                         lambda msg: self.subs[0].signalMessage(msg))
        rospy.Subscriber(self.topics[1], Odometry,
                         lambda msg: self.subs[1].signalMessage(msg))
        rospy.Subscriber(self.topics[2], Odometry,
                         lambda msg: self.subs[2].signalMessage(msg))
        rospy.Subscriber(self.topics[3], Image,
                         lambda msg: self.subs[3].signalMessage(msg))
        rospy.Subscriber(self.topics[4], CameraInfo,
                         lambda msg: self.subs[4].signalMessage(msg))
        # rospy.Subscriber(self.topics[5], Image,
        #                 lambda msg: self.subs[5].signalMessage(msg))
        # rospy.Subscriber(self.topics[6], Image,
        #                 lambda msg: self.subs[6].signalMessage(msg))
        # rospy.Subscriber(self.topics[7], CameraInfo,
        #                 lambda msg: self.subs[7].signalMessage(msg))
        # rospy.Subscriber(self.topics[8], Odometry,
        #                 lambda msg: self.subs[8].signalMessage(msg))

        self.kb_listener = Listener(on_release=self.on_release)
        self.kb_listener.start()
        # listener.join()
        print 'Ready to spin: Press s to save and Esc to quit'
        # try:
        rospy.spin()
        # rospy.spinOnce()
        # except:
        # print("Error")
        # self.out_bag.close()
        print'toodles!'

    def on_release(self, key):
        # string_in = str(key).split("au'")[1].split("'")[0]
        string_in = str(key)
        if key == Key.esc:
            print "here"
            self.out_bag.close()
            rospy.signal_shutdown('tood')
            exit(0)
        elif string_in == "u\'s\'":
            print 'saving'+str(self.data_count)
            for idx, topic in enumerate(self.topics):
                self.out_bag.write(
                    topic, self.latest_sync_tuple[idx], self.latest_sync_tuple[0].header.stamp)
            # self.out_bag.write(
            #    self.topics[0], self.latest_sync_tuple[0], self.latest_sync_tuple[0].header.stamp)
            # self.out_bag.write(
            #    self.topics[1], self.latest_sync_tuple[1], self.latest_sync_tuple[0].header.stamp) #self.out_bag.write(
            #    self.topics[2], self.latest_sync_tuple[2], self.latest_sync_tuple[0].header.stamp)
            # self.out_bag.write(
            #    self.topics[3], self.latest_sync_tuple[3], self.latest_sync_tuple[0].header.stamp)
            # self.out_bag.write(
            #    self.topics[4], self.latest_sync_tuple[4], self.latest_sync_tuple[0].header.stamp)
            # self.out_bag.write(
            #    self.topics[5], self.latest_sync_tuple[5], self.latest_sync_tuple[0].header.stamp)
            # self.out_bag.write(
            #    self.topics[6], self.latest_sync_tuple[6], self.latest_sync_tuple[0].header.stamp)
            # self.out_bag.write(
            #    self.topics[7], self.latest_sync_tuple[7], self.latest_sync_tuple[0].header.stamp)
            # self.out_bag.write(
            #    self.topics[8], self.latest_sync_tuple[8], self.latest_sync_tuple[0].header.stamp)
            self.updated = False
            self.data_count += 1

    # def callback(self, img_msg, kinect_odom_msg, tag_odom_msg, depth_img_msg, kinect_camera_info, rs_depth_msg, rs_color_msg, rs_camera_info, rs_odom):
    def callback(self, img_msg, kinect_odom_msg, tag_odom_msg, depth_img_msg, kinect_camera_info):
            # print 'asdasd'
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, 'passthrough')
        # cv2.imshow("color_img", cv_img)
        # cv2.waitKey(20)
        # self.latest_sync_tuple = [
        #    img_msg, kinect_odom_msg, tag_odom_msg, depth_img_msg, kinect_camera_info, rs_depth_msg, rs_color_msg, rs_camera_info, rs_odom]
        self.latest_sync_tuple = [img_msg, kinect_odom_msg,
                                  tag_odom_msg, depth_img_msg, kinect_camera_info]
        self.updated = True

    def init_sub(self):
        self.subs = []
        self.subs.append(Subscriber(self.topics[0], Image))
        self.subs.append(Subscriber(self.topics[1], Odometry))
        self.subs.append(Subscriber(self.topics[2], Odometry))
        self.subs.append(Subscriber(self.topics[3], Image))

        self.subs.append(Subscriber(self.topics[4], CameraInfo))
        #self.subs.append(Subscriber(self.topics[5], Image))
        #self.subs.append(Subscriber(self.topics[6], Image))
        #self.subs.append(Subscriber(self.topics[7], CameraInfo))
        #self.subs.append(Subscriber(self.topics[8], Odometry))
        self.synchronizer = ApproximateTimeSynchronizer(self.subs, 1, 0.1)

        self.synchronizer.registerCallback(self.callback)


def main():
    rospy.init_node("kinect_sync_node")
    # topics_to_parse = ['/kinect2/qhd/image_color_rect',
    #                   '/kinect_one/vicon_odom', '/small_checkerboard/vicon_odom', '/kinect2/qhd/image_depth_rect', '/kinect2/qhd/camera_info', '/camera/aligned_depth_to_color/image_raw', '/camera/color/image_raw', '/camera/color/camera_info', '/realsense_rig/vicon_odom']
    topics_to_parse = ['/kinect2/qhd/image_color_rect',
                       '/kinect_one/vicon_odom', '/checkerboard/vicon_odom',
                       '/kinect2/qhd/image_depth_rect', '/kinect2/qhd/camera_info']
    filename = "/home/aditya/bagfiles/kinect_one/ext_calib_kinect_big_board.bag"

    sync_obj = Synchronize(topics_to_parse, filename)
    sync_obj.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
