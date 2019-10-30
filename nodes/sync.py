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
    def __init__(self, topics, topic_types, out_filename):
        self.subs = []
        self.out_bag = rosbag.Bag(out_filename, "w")
        self.topics = topics
        self.topic_types = topic_types
        self.bridge = CvBridge()
        self.data_count = 0
        self.latest_sync_tuple = []
        self.updated = False
        self.init_sub()

    def spin(self):
        for i in range(len(self.topics)):
            rospy.Subscriber(self.topics[i], self.topic_types[i],
                             lambda msg: self.subs[i].signalMessage(msg), tcp_nodelay=True)

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
                    topic, self.latest_sync_tuple[idx], self.latest_sync_tuple[idx].header.stamp)
            self.updated = False
            self.data_count += 1

    # def callback(self, img_msg, kinect_odom_msg, tag_odom_msg, depth_img_msg, kinect_camera_info, rs_depth_msg, rs_color_msg, rs_camera_info, rs_odom):
    # img_msg, kinect_odom_msg, tag_odom_msg, depth_img_msg, kinect_camera_info):
    def callback(self, *args):
        sync_tuple = []
        # for i,msg in enumerate(args):
        #     if self.topic_types[i] == Image:
        #         cv_img = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

        # cv2.imshow("color_img", cv_img)
        # cv2.waitKey(20)
        self.latest_sync_tuple = args
        self.updated = True

    def init_sub(self):
        self.subs = []
        for i in range(len(self.topics)):
            self.subs.append(Subscriber(self.topics[i], self.topic_types[i]))
        self.synchronizer = ApproximateTimeSynchronizer(self.subs, 10, 0.1)

        self.synchronizer.registerCallback(self.callback)


def main():
    rospy.init_node("kinect_sync_node")
    topics_to_parse = ['/camera/infra1/image_rect_raw',
                       '/realsense_rig_new/vicon_odom',
                       '/small_checkerboard/vicon_odom',
                       '/camera/infra2/image_rect_raw',
                       '/camera/color/image_raw',
                       '/camera/infra1/camera_info',
                       '/camera/infra2/camera_info',
                       '/camera/color/camera_info']
    topic_types = [Image, Odometry, Odometry, Image,
                   Image, CameraInfo, CameraInfo, CameraInfo]
    filename = "/home/al/bagfiles/ext_calib_small_all.bag"

    sync_obj = Synchronize(topics_to_parse, topic_types, filename)
    sync_obj.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
