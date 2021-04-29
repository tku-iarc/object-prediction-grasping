#!/usr/bin/env python3
import cv2
import sys
sys.path.insert(1, "/home/chien/.local/lib/python3.6/site-packages/")
import rospy
from get_rs_image import Get_image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

if __name__ == '__main__':
    rospy.init_node('123')
    sub_img = Get_image()
    cv2.imshow("123",sub_img.cv_image)
    rospy.spin()