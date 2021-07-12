#!/usr/bin/env python
# use moveit_commander (the Python MoveIt user interfaces )
from math import pi
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from ur_move_command.msg import obj_infomsg
from ur_move_command.msg import obj_array
## END_SUB_TUTORIAL

def all_close(goal, actual, tolerance):

  all_equal = True
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True

class MoveGroupTutorial(object):
  """MoveGroupTutorial"""
  def __init__(self):
    super(MoveGroupTutorial, self).__init__()
    moveit_commander.roscpp_initialize(sys.argv)
    #rospy.init_node('move_group_tutorial_ur5', anonymous=True)
 
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "manipulator" 
    group = moveit_commander.MoveGroupCommander(group_name)
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)
    reference_frame = 'base_link'
    group.set_pose_reference_frame(reference_frame)
    ee_link = group.get_end_effector_link()
    
    group.set_end_effector_link(ee_link)
    group.set_max_acceleration_scaling_factor(0.1)
    group.set_max_velocity_scaling_factor(0.1)
    
    self.box_name = ''
    self.robot = robot
    self.scene = scene
    self.group = group
    self.reference_frame = reference_frame
    self.end_effector_link = ee_link

    self.origin_degree = [0, 0, 0]
  
  def origin_pose(self):
    group = self.group
    joint_goal = group.get_current_joint_values()
    joint_goal[0] = -pi * 0.5
    joint_goal[1] = -pi * 0.5
    joint_goal[2] = -pi * 0.5
    joint_goal[3] = -pi * 0.5
    joint_goal[4] = pi * 0.5
    joint_goal[5] = pi * 0.5    
    group.go(joint_goal, wait=True)
    group.stop()
    group.clear_pose_targets()
    current_joints = group.get_current_joint_values()
    origin_orientation =  group.get_current_pose().pose.orientation
    origindegree =  euler_from_quaternion([origin_orientation.x, origin_orientation.y, origin_orientation.z, origin_orientation.w]) 

    self.origin_degree[0] = origindegree[0]/3.14*180.0
    self.origin_degree[1] = origindegree[1]/3.14*180.0
    self.origin_degree[2] = origindegree[2]/3.14*180.0
    return all_close(joint_goal, current_joints, 0.01)

  def camera_pose(self):
    '''
    group = self.group
    joint_goal = group.get_current_joint_values()   #(x,y,z) = (0.11096, 0.46508, 0.5) (rx,ry,rz) = (2.31,2.304,0.195)
    joint_goal[0] = -pi * 0.5
    joint_goal[1] = -pi * 0.5
    joint_goal[2] = -pi * (4.0/9.0) #-77.46
    joint_goal[3] = -pi * (11.0/18.0) #-112.115
    joint_goal[4] = pi * 0.5
    joint_goal[5] = pi * 0.5    
    group.go(joint_goal, wait=True)
    group.stop()
    group.clear_pose_targets()
    current_joints = group.get_current_joint_values()
    origin_orientation =  group.get_current_pose().pose.orientation
    origindegree =  euler_from_quaternion([origin_orientation.x, origin_orientation.y, origin_orientation.z, origin_orientation.w])
    return all_close(joint_goal, current_joints, 0.01)
    '''
    group = self.group
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = 0.11096
    pose_goal.position.y = 0.46508
    pose_goal.position.z = 0.5002
    pose_goal.orientation.x = -0.06103
    pose_goal.orientation.y = 0.70449
    pose_goal.orientation.z = -0.06157
    pose_goal.orientation.w = 0.70439
    group.set_pose_target(pose_goal, self.end_effector_link)
    plan = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()


  def camera_pose2(self):

    group = self.group
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = 0.477477
    pose_goal.position.y = -0.023858
    pose_goal.position.z = 0.500234
    pose_goal.orientation.x = -0.447114
    pose_goal.orientation.y = -0.533706
    pose_goal.orientation.z = 0.54758
    pose_goal.orientation.w = -0.464116
    group.set_pose_target(pose_goal, self.end_effector_link)
    plan = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()



  def first_pose(self):
    group = self.group
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = 0.11096
    pose_goal.position.y = 0.46508
    pose_goal.position.z = 0.5002
    
    quaternion = quaternion_from_euler(np.radians(self.origin_degree[0]+10),np.radians(self.origin_degree[1]-10), np.radians(self.origin_degree[2]))   #roll_angle, pitch_angle, yaw_angle  
    


    pose_goal.orientation.x = quaternion[0]
    pose_goal.orientation.y = quaternion[1]
    pose_goal.orientation.z = quaternion[2]
    pose_goal.orientation.w = quaternion[3]

    group.set_pose_target(pose_goal, self.end_effector_link)
    plan = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

  def up(self):
    group = self.group
    pose_goal = group.get_current_pose().pose
    pose_goal.position.z += 0.016

    '''
    quaternion = quaternion_from_euler(0, np.radians(90.), 0)  
    pose_goal.orientation.x = quaternion[0]
    pose_goal.orientation.y = quaternion[1]
    pose_goal.orientation.z = quaternion[2]
    pose_goal.orientation.w = quaternion[3]
    '''

    group.go(pose_goal, wait=True)
    pose_msg = group.get_current_pose().pose.orientation
    euler_angle = euler_from_quaternion([pose_msg.x, pose_msg.y, pose_msg.z, pose_msg.w])
    print('----------------------------------')
    print(group.get_current_pose().pose)
    print(np.rad2deg(euler_angle))
    group.stop()

curr_list = obj_array()
def get_array(obj_array):
    global curr_list
    if (len(obj_array.Obj_list)>0):
        curr_list = obj_array.Obj_list
    else:
        curr_list = []


def main():
  '''
  while not rospy.is_shutdown():
    if curr_list:
      print(curr_list)
  '''    
  try:
    print("============ Press `Enter` to begin the tutorial by setting up the moveit_commander (press ctrl-d to exit) ...")
    tutorial = MoveGroupTutorial()
    print('start')
    
    raw_input()
    
    print("origin_pose")
    tutorial.origin_pose()
    print('finish')
    raw_input()
    print('================================')
    print("camera_pose")
    tutorial.camera_pose()
    print(curr_list)
    
    #raw_input()
    #print('================================')
    #print("camera_pose")
    #tutorial.camera_pose2()
    

    print("============ Python tutorial demo complete!")
    
  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return
   
if __name__ == '__main__':
  rospy.init_node('Strategy')
  rospy.Subscriber("info_topic", obj_array, get_array)
  main()