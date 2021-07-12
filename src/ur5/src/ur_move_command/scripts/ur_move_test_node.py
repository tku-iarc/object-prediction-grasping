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
import roslib; roslib.load_manifest('robotiq_2f_gripper_control')
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output  as outputMsg
import time

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
    self.target_angle = 0
    self.target_x = 0
    self.target_y = 0

  def origin_pose(self):
    group = self.group
    group.set_max_acceleration_scaling_factor(0.1)
    group.set_max_velocity_scaling_factor(0.1)
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
    group = self.group
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = 0.11096
    pose_goal.position.y = 0.51508     #0.56508
    pose_goal.position.z = 0.5002
    pose_goal.orientation.x = -0.06103
    pose_goal.orientation.y = 0.70449
    pose_goal.orientation.z = -0.06157
    pose_goal.orientation.w = 0.70439
    group.set_pose_target(pose_goal, self.end_effector_link)
    plan = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()


  def camera_pose2(self):   #neeed x or y +0.05

    group = self.group
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = 0.524307
    pose_goal.position.y = 0.058145
    pose_goal.position.z = 0.489706
    pose_goal.orientation.x = -0.45554
    pose_goal.orientation.y = -0.53682
    pose_goal.orientation.z = 0.54192
    pose_goal.orientation.w = -0.45894
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
    
    quaternion = quaternion_from_euler(np.radians(0),np.radians(90.), np.radians(0))   #roll_angle, pitch_angle, yaw_angle  
    


    pose_goal.orientation.x = quaternion[0]
    pose_goal.orientation.y = quaternion[1]
    pose_goal.orientation.z = quaternion[2]
    pose_goal.orientation.w = quaternion[3]

    group.set_pose_target(pose_goal, self.end_effector_link)
    plan = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()
  
  def move_to_obj(self):
    group = self.group
    pose_goal = group.get_current_pose().pose
    
    move_x = (self.target_x - 355)*0.001
    move_y = (298 - self.target_y)*0.001
    move_angle = self.target_angle
    '''
    move_x = (320 - 355)*0.001
    move_y = (298 - 240)*0.001
    move_angle =0
    '''
    pose_goal.position.x += move_x
    pose_goal.position.y += move_y
    pose_goal.position.z -= 0.356
    
    group.set_max_acceleration_scaling_factor(0.9)  #real0.7 1.98s  m0.6
    group.set_max_velocity_scaling_factor(0.9)

    quaternion = quaternion_from_euler(np.radians(move_angle), np.radians(90.), 0)  
    pose_goal.orientation.x = quaternion[0]
    pose_goal.orientation.y = quaternion[1]
    pose_goal.orientation.z = quaternion[2]
    pose_goal.orientation.w = quaternion[3]
    group.go(pose_goal, wait=True)
    group.stop()

  def move_slide_obj(self):
    group = self.group
    pose_goal = group.get_current_pose().pose
    
    move_y = (355 - self.target_x)*0.001
    move_x = (298 - self.target_y)*0.001
    move_angle = self.target_angle
    print(move_x, move_y)
    pose_goal.position.x += move_x
    pose_goal.position.y += move_y
    pose_goal.position.z -= 0.356
    
    group.set_max_acceleration_scaling_factor(0.9) 
    group.set_max_velocity_scaling_factor(0.9)

    quaternion = quaternion_from_euler(np.radians(90.), np.radians(90.), np.radians(move_angle))  
    pose_goal.orientation.x = quaternion[0]
    pose_goal.orientation.y = quaternion[1]
    pose_goal.orientation.z = quaternion[2]
    pose_goal.orientation.w = quaternion[3]
    group.go(pose_goal, wait=True)
    group.stop()

  def down(self):
    group = self.group
    pose_goal = group.get_current_pose().pose
    pose_goal.position.z -= 0.08    #bottle 0.08
    group.go(pose_goal, wait=True)
    group.stop()
   
  def down1(self):
    group = self.group
    pose_goal = group.get_current_pose().pose

    #move_x = (320 - 355)*0.001
    #move_y = (298 - 240)*0.001
    move_y = (355 - 320)*0.001
    move_x = (298 - 240)*0.001
    pose_goal.position.x += move_x
    pose_goal.position.y += move_y
    pose_goal.position.z -= 0.436                  #0.356  0.436 
    #pose_goal.position.x += 0.05
    group.set_max_acceleration_scaling_factor(0.1)
    group.set_max_velocity_scaling_factor(0.1)

    quaternion = quaternion_from_euler(np.radians(90.), np.radians(90.), np.radians(-30.))  
    pose_goal.orientation.x = quaternion[0]
    pose_goal.orientation.y = quaternion[1]
    pose_goal.orientation.z = quaternion[2]
    pose_goal.orientation.w = quaternion[3]
    group.go(pose_goal, wait=True)
    group.stop()

  def up(self):
    group = self.group
    pose_goal = group.get_current_pose().pose
    pose_goal.position.z += 0.1   #0.356
    quaternion = quaternion_from_euler(0, np.radians(90.), 0)  
    pose_goal.orientation.x = quaternion[0]
    pose_goal.orientation.y = quaternion[1]
    pose_goal.orientation.z = quaternion[2]
    pose_goal.orientation.w = quaternion[3]
    group.go(pose_goal, wait=True)
    group.stop()

  def up1(self):
    group = self.group
    pose_goal = group.get_current_pose().pose
    pose_goal.position.x -= 0.1   #0.356
    pose_goal.position.z += 0.25   #0.356
    quaternion = quaternion_from_euler(np.radians(90.), np.radians(90.), 0)  
    pose_goal.orientation.x = quaternion[0]
    pose_goal.orientation.y = quaternion[1]
    pose_goal.orientation.z = quaternion[2]
    pose_goal.orientation.w = quaternion[3]
    group.go(pose_goal, wait=True)
    group.stop()

  def box(self):
    group = self.group
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = 0.44383
    pose_goal.position.y = 0.22829     
    pose_goal.position.z = 0.431812
    pose_goal.orientation.x = 0.29971
    pose_goal.orientation.y = 0.64045
    pose_goal.orientation.z = -0.2998
    pose_goal.orientation.w = 0.6404
    group.go(pose_goal, wait=True)
    group.stop()

  def mid_point(self):
    group = self.group
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = 0.40897
    pose_goal.position.y = 0.24259     
    pose_goal.position.z = 0.40175
    pose_goal.orientation.x = 0.34362
    pose_goal.orientation.y = 0.62447
    pose_goal.orientation.z = -0.33783
    pose_goal.orientation.w = 0.61468
    group.go(pose_goal, wait=True)
    group.stop()

curr_list = obj_array()
def get_array(obj_array):
    global curr_list
    if (len(obj_array.Obj_list)>0):
        curr_list = obj_array.Obj_list
    else:
        curr_list = []


def res():
    command = outputMsg.Robotiq2FGripper_robot_output()
    command.rACT = 0
    print(command)
    return command
def test():
    print('bbb')
    command = outputMsg.Robotiq2FGripper_robot_output()
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 150
    return command

def catch11():
    #command = outputMsg.Robotiq2FGripper_robot_output()
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 150
    command.rPR = 255
    print(command)
    return command

def loose11():
    #command = outputMsg.Robotiq2FGripper_robot_output()
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 150
    command.rPR = 0
    print(command)
    return command

def main():
  try:
    print("============ Press `Enter` to begin the tutorial by setting up the moveit_commander (press ctrl-d to exit) ...")
    tutorial = MoveGroupTutorial()
    print('start')
    '''
    #raw_input()
    start = time.time()
    command = res()
    pub.publish(command)            
    rospy.sleep(0.1)
    '''
    
    '''
    print("origin_pose")
    tutorial.origin_pose()
    end = time.time()
    print('finish')
    print(end-start)
    '''
    '''
    raw_input()
    print('================================')
    print("camera_pose2")
    tutorial.camera_pose2()
    
    raw_input()
    print('================================')
    print("down1")
    tutorial.down1()
    raw_input()
    print('================================')
    print("camera_pose2")
    tutorial.camera_pose2()
    '''
    raw_input()
    print('================================')
    print("origin_pose")
    tutorial.origin_pose()
    
    
    print("============ Python tutorial demo complete!")
    
  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return
idle  = 0
busy  = 1
first = 2
camera = 3
down  = 4
gcatch = 5
up = 6
gopen = 7
get_info = 8
move_obj = 9
origin = 10
box = 11
stop = 12

slide_first = 13
camera1 = 14
get_slideinfo = 15
move_obj1 = 16
down1 = 17
gcatch1 = 18
up1 = 19
box1 = 20
gopen1 = 21
gomidpoint = 22
class Task:
  def __init__(self):
    self.state = slide_first
    self.nextstate = idle
    self.status = idle
    

  def process(self):
    global command
    if self.state == first:
      print('aaaaaa')
      command = res()
      pub.publish(command)
      rospy.sleep(1)
      #tutorial.origin_pose()
      self.state = camera
      print('finish')
    elif self.state == slide_first:
      print('bbbbbb')
      command = res()
      pub.publish(command)
      rospy.sleep(1)
      #tutorial.origin_pose()
      self.state = camera1
      print('finish')

    elif self.state == camera:  
      print('================================')
      print("camera_pose")
      command.rACT = 1
      command.rGTO = 1
      command.rSP  = 255
      command.rFR  = 150
      print(command)
      pub.publish(command)
      rospy.sleep(1)
      tutorial.camera_pose()
      self.state = get_info

    elif self.state == get_info:
      #print(curr_list)
      a = raw_input('Enter target name(Block, Danboard, Tetra Pak, Bottle) :')
      if a == 'q':
        self.state = stop
      else:  
        for i in range(len(curr_list)): 
          if curr_list[i].object_name == a:
            tutorial.target_x = curr_list[i].x
            tutorial.target_y = curr_list[i].y
            if curr_list[i].degree > 0:
              tutorial.target_angle = curr_list[i].degree-90
            else:
              tutorial.target_angle = curr_list[i].degree+90
            self.state = move_obj
            break
          else:
            print('no object')
            self.state = get_info

    elif self.state == move_obj:
      print('================================')
      print("move_obj")
      start = time.time()
      tutorial.move_to_obj()
      self.state = down
      end = time.time()
      print('finish')
      print(end-start)

    elif self.state == down:
      print('================================')
      print("down")
      start = time.time()
      tutorial.down()
      self.state = gcatch
      end = time.time()
      print('finish')
      print(end-start)
    elif self.state == gcatch:
      command = catch11()
      pub.publish(command)
      rospy.sleep(0.7)
      self.state = up

    elif self.state == up:
      print('================================')
      print("up")
      tutorial.up()
      self.state = origin

    elif self.state == origin:
      print('================================')
      print("origin_pose")
      tutorial.origin_pose()
      self.state = box

    elif self.state == box:
      print('================================')
      print("box")
      tutorial.box()
      self.state = gopen

    elif self.state == gopen:
      command = loose11()
      pub.publish(command)
      rospy.sleep(0.8)
      self.state = camera

    elif self.state == stop:
      print('stop')
      self.state = stop  

    elif self.state == camera1:
      print('================================')
      print("camera_pose2")
      command.rACT = 1
      command.rGTO = 1
      command.rSP  = 255
      command.rFR  = 150
      print(command)
      pub.publish(command)
      rospy.sleep(1)
      tutorial.camera_pose2()
      self.state = get_slideinfo

    elif self.state == get_slideinfo:
      #print(curr_list)
      a = raw_input('Enter target name(Block, Danboard, Tetra Pak, Bottle) :')
      if a == 'q':
        self.state = stop
      else:  
        for i in range(len(curr_list)): 
          if curr_list[i].object_name == a:
            tutorial.target_x = curr_list[i].x
            tutorial.target_y = curr_list[i].y
            
            if curr_list[i].object_name == 'Bottle':
              tutorial.target_x = curr_list[i].x - 17
            elif curr_list[i].object_name == 'Block':
              tutorial.target_x = curr_list[i].x - 10   

            elif curr_list[i].object_name == 'Tetra Pak':
              tutorial.target_x = curr_list[i].x - 18 
            if curr_list[i].degree > 0:
              tutorial.target_angle = 90 - curr_list[i].degree
            else:
              tutorial.target_angle = -90 - curr_list[i].degree
            self.state = move_obj1
            print(tutorial.target_angle)
            break
          else:
            print('no object')
            self.state = get_slideinfo
    
    elif self.state == move_obj1:
      print('================================')
      print("move_obj1")
      start = time.time()
      tutorial.move_slide_obj()
      self.state = down1
      end = time.time()
      print('finish')
      print(end-start)

    elif self.state == down1:
      print('================================')
      print("down")
      start = time.time()
      tutorial.down()
      self.state = gcatch1
      end = time.time()
      print('finish')
      print(end-start)
    elif self.state == gcatch1:
      command = catch11()
      pub.publish(command)
      rospy.sleep(0.7)
      self.state = up1

    elif self.state == up1:
      print('================================')
      print("up")
      tutorial.up1()
      self.state = gomidpoint

    elif self.state == gomidpoint:
      print('================================')
      print("up")
      tutorial.mid_point()
      self.state = box1

    elif self.state == box1:
      print('================================')
      print("box1")
      tutorial.origin_pose()
      self.state = gopen1
    elif self.state == gopen1:
      command = loose11()
      pub.publish(command)
      rospy.sleep(0.8)
      self.state = camera1  
if __name__ == '__main__':
  
  rospy.init_node('Strategy')
  pub = rospy.Publisher('Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output)
  command = outputMsg.Robotiq2FGripper_robot_output()
  main()
  '''
  rospy.init_node('Strategy')
  rospy.Subscriber("info_topic", obj_array, get_array)

  pub = rospy.Publisher('Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output)
  command = outputMsg.Robotiq2FGripper_robot_output()
  tutorial = MoveGroupTutorial()
  test = Task()
  while not rospy.is_shutdown():
    try:
        test.process()
    except rospy.ROSInterruptException:
        print('error')
        pass
        break
  '''