#!/usr/bin/env python2
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from openpyxl import Workbook
# from universal_robot_kinematics import invKine
# from forward_kinematics import fwd_kin
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
print("Environment Ready")

class IntelRealsense:
    def __init__(self):
        # initiate the pipeline
        self.shift_x = 300
        self.shift_y = 0
        self.pp_k = 0
        self.name = 1
        self.colorizer = rs.colorizer()
        self.pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        profile = self.pipe.start(config)
        i = profile.get_stream(rs.stream.depth)
        self.intr = i.as_video_stream_profile().get_intrinsics()
        # extr = i.as_video_stream_profile().get_extrinsics_to()
        # print('intr %s' %self.intr)
        s = profile.get_device().query_sensors()[1]
        s.set_option(rs.option.enable_auto_exposure, False)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    def reverse_perspective_projection(self, pp_x, pp_y, pp_depth):
        intr = self.intr
        image_plane = np.array([  [pp_x * pp_depth],
                                  [pp_y * pp_depth],
                                  [1 * pp_depth]      ])
        # print('Image Plane \n %s \n' %image_plane)
        T_perspective_projection = np.array([   [intr.fx, 0,        intr.ppx,      0],
                                                [0,       intr.fy,  intr.ppy,      0],
                                                [0,       0,        1,             0] ])
        # print('T_perspective_projection: \n%s\n' %T_perspective_projection)
        # T_perspective_projection * vec(world_coordinates) = image_plane
        answer = np.linalg.lstsq(T_perspective_projection, image_plane, rcond=None)
        return np.array(answer[0])

    def perspective_projection(self, real_x, real_y, real_depth):
        intr = self.intr
        real_world = np.array([   [real_x],
                                  [real_y],
                                  [real_depth],
                                  [1]       ])
        # print('Image Plane \n %s \n' %image_plane)
        T_perspective_projection = np.array([   [intr.fx, 0,        intr.ppx,      0],
                                                [0,       intr.fy,  intr.ppy,      0],
                                                [0,       0,        1,             0] ])
        # T_perspective_projection * vec(world_coordinates) = image_plane
        answer =( T_perspective_projection.dot(real_world) )/real_depth
        return np.array(answer)

    # def transformation_image2camera(self, pp_x, pp_y, pp_depth, pp_k, shift_x=0, shift_y=0):
    #     pp_x -= 424; pp_y -= 240;
    #     print('pp camera frame (%s, %s)' %(pp_x, pp_y))
    #     pingpong_camera = np.array([  [pp_x * pp_k *100],
    #                                   [pp_y * pp_k *100],
    #                                   [pp_depth * 100],
    #                                   [1]  ])
    #     return pingpong_camera

    def pingpong_detection(self, shift_x=0, shift_y=100, scope_side=60, scope_depth=50, display=True):
        # pp_area = 0.001256 # meters
        capture = time.time()
        frameset = self.pipe.wait_for_frames()
        depth_frame = frameset.get_depth_frame()
        depth = np.asanyarray(depth_frame.get_data())
        depth_shape = depth.shape
        depth_crop = depth[0:depth_shape[0]-shift_y, 0+shift_x:depth_shape[1]-200]
        min = depth_crop[depth_crop > 10].min()     # Depth Values
        # print('depth : %f' %(min))
        if min > 700 and min < 2100:
            min_pt = np.where(depth_crop == min)
            depth_scope = depth_crop[int(min_pt[0][0]-scope_side/2):int(min_pt[0][0]+scope_side/2), int(min_pt[1][0]-scope_side/2): int(min_pt[1][0]+scope_side/2)]
            numpix = 0; sumx = 0; sumy = 0
            for row in range(0,depth_scope.shape[0]):
                for col in range(0,depth_scope.shape[1]):
                    if depth_scope[row,col] < min+scope_depth and depth_scope[row,col] > min-scope_depth:numpix+=1; sumx += col; sumy += row;
            if numpix != 0:ppscope_x = sumx/numpix; ppscope_y = sumy/numpix
            else:ppscope_x = 0; ppscope_y = 0
            pp_x = ppscope_x+shift_x+min_pt[1][0]-scope_side/2
            pp_y = ppscope_y+min_pt[0][0]-scope_side/2
            # min_depth_color = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            # min_x = shift_x+min_pt[1][0]
            # min_y = min_pt[0][0]
            # cv2.circle(min_depth_color, (int(min_x),int(min_y)), (int)(1),(0,255,0),-1)
            # cv2.imwrite('/home/s/catkin_ws/src/ur_modern_driver/images/'+'Argmin.png',min_depth_color)
            pp_depth = depth[int(pp_y),int(pp_x)] * self.depth_scale
        else:
            pp_x=0; pp_y=0; pp_depth=0
        if display == True:
            # cv2.imwrite('/home/s/catkin_ws/src/ur_modern_driver/images/'+'Depth_Original.png',depth)
            depth_color = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            # cv2.imwrite('/home/s/catkin_ws/src/ur_modern_driver/images/'+'Depth_Colormap.png',depth_color)
            # depth_scope = depth_color[0:depth_shape[0]-shift_y, 0+shift_x:depth_shape[1]-0]
            # depth_scope = depth_scope[int(min_pt[0][0]-scope_side/2):int(min_pt[0][0]+scope_side/2), int(min_pt[1][0]-scope_side/2): int(min_pt[1][0]+scope_side/2)] #**
            # cv2.imwrite('/home/s/catkin_ws/src/ur_modern_driver/images/'+'Ping-Pong_Scope1.png',depth_scope)
            # cv2.circle(depth_scope, (int(ppscope_x),int(ppscope_y)), (int)(1),(0,255,0),-1)
            # cv2.imwrite('/home/s/catkin_ws/src/ur_modern_driver/images/'+'Ping-Pong_Scope2.png',depth_scope)

            cv2.line(depth_color, (shift_x,0), (shift_x, depth_shape[0]), (0,0,255), 1)
            cv2.line(depth_color, (0,depth_shape[0]-shift_y), (depth_shape[1], depth_shape[0]-shift_y), (0,0,255), 1)
            cv2.circle(depth_color, (int(pp_x),int(pp_y)), (int)(1),(0,255,0),-1)
            # cv2.circle(depth_color, (int(np.size(depth,1)/2),int(np.size(depth,0)/2)), (int)(0),(0,0,255),5)
            cv2.namedWindow('Object Detectiom using Depth Image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Object Detectiom using Depth Image', depth_color)
            # cv2.imwrite('/home/s/catkin_ws/src/ur_modern_driver/images/'+'Ping-Pong.png',depth_color)
            # if pp_x != 0 and pp_y != 0:
            #     filename = 'pingpong'+str(self.name)+'.png'
            #     cv2.imwrite('/home/s/catkin_ws/src/ur_modern_driver/images/'+filename,depth_color)
            #     self.name+=1
            cv2.waitKey(0)
            # key = cv2.waitKey(1)
            # if key & 0xFF == ord('q') or key == 27:
            #     cv2.destroyAllWindows()
        return pp_x, pp_y, pp_depth, capture

    def pingpong_velocity(self, STATE, pp_x, pp_y, pp_depth, capture, display=True):
        v_x = 999; v_y = 999; v_depth = 999;a_x =999; a_y = 999; a_depth = 999
        if pp_x == -1.1 and pp_y == 1.5:
            STATE = 'NONE'
        else:
            if STATE == 'NONE':STATE='INITIAL'
            if STATE == 'INITIAL':
                self.lastpp_x = pp_x; self.lastpp_y = pp_y; self.lastcapture = capture
                self.lastpp_depth = pp_depth; STATE = 'VELOCITY'
            elif STATE == 'VELOCITY':
                delt = capture-self.lastcapture; self.lastcapture = capture
                v_x = (pp_x - self.lastpp_x)/(delt); v_y = (pp_y - self.lastpp_y)/(delt)
                v_depth = (pp_depth - self.lastpp_depth)/(delt)
                self.lastv_x = v_x; self.lastv_y = v_y
                self.lastv_depth = v_depth
                self.lastpp_x = pp_x; self.lastpp_y = pp_y
                self.lastpp_depth = pp_depth
                STATE = 'KALMAN'
            elif STATE == 'KALMAN':
                delt = capture-self.lastcapture; self.lastcapture = capture
                # print('Delta t %s' %delt)
                v_x = (pp_x - self.lastpp_x)/(delt); v_y = (pp_y - self.lastpp_y)/(delt)
                v_depth = (pp_depth - self.lastpp_depth)/(delt)
                a_x = (v_x - self.lastv_x)/delt; a_y = (v_y - self.lastv_y)/delt
                a_depth = (v_depth - self.lastv_depth)/(delt)
                if display != True:
                    self.lastv_x = v_x; self.lastv_y = v_y
                    self.lastv_depth = v_depth
                    self.lastpp_x = pp_x; self.lastpp_y = pp_y
                    self.lastpp_depth = pp_depth
                    self.lasta_x = a_x;self.lasta_y = a_y
                    self.lasta_depth = a_depth
        if display == True and  STATE == 'KALMAN' :
            frameset = self.pipe.wait_for_frames()
            depth_frame = frameset.get_depth_frame()
            depth_color = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            predpp_x = self.lastpp_x + self.lastv_x*delt
            # predpp_y = self.lastpp_y + self.lastv_y*delt + 0.5*9.8*(delt**2)
            self.lastv_x = v_x; self.lastv_y = v_y
            self.lastpp_x = pp_x; self.lastpp_y = pp_y
            cv2.line(depth_color, (int(predpp_x), 0), (int(predpp_x), depth_color.shape[0]), (0,255,0), 1)
            # cv2.line(depth_color, (0, int(predpp_y)), (depth_color.shape[1],int(predpp_y)), (0,255,0), 1)
            cv2.namedWindow('State Prediction', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('State Prediction', depth_color)
            filename = 'pingpong'+str(self.name)+'.png'
            cv2.imwrite('/home/idealab/catkin_ws/src/thesis/src/predict/'+filename,depth_color)
            self.name+=1
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
        return v_x,v_y,v_depth,a_x,a_y,a_depth,STATE

    def transformation_camera2base(self, x_camera=-1.10, y_camera=1.5, z_camera=0.31):
        T_camera2base = np.array([  [-1, 0, 0, x_camera],
                                    [0, 0, -1, y_camera],
                                    [0, -1, 0, z_camera],
                                    [0, 0, 0, 1]])
        return T_camera2base

    def transformation_end2base(self, x_end=0, y_end=0, z_end=0):
        T_end2base = np.array([  [0, 0, -1, x_end],
                                 [1, 0, 0, y_end],
                                 [0, -1, 0, z_end],
                                 [0, 0, 0, 1]  ])
        return T_end2base

    # def rot2qua(self,MAT):
    #     # quaternion conversion
    #     w = math.sqrt(1 + MAT[0,0]**2 + MAT[1,1]**2 + MAT[2,2]**2)/2.0
    #     x = (MAT[2,1] - MAT[1,2])/(4.0*w)
    #     y = (MAT[0,2] - MAT[2,0])/(4.0*w)
    #     z = (MAT[1,0] - MAT[0,1])/(4.0*w)
    #     QUA = np.array([[x, 0, 0, 0],
    #             [0, y, 0, 0],
    #             [0, 0, z, 0],
    #             [0, 0, 0, w]])
    #     return QUA

# __main__
# IntelRealsense = IntelRealsense()
# if __name__ == '__main__':
#     # Prepare for excel
#     print("Initiate Object Detection")
#     STATE = 'INITIAL'
#     _,_,_,_, last_capture = IntelRealsense.pingpong_detection(display = False)
#     TRAN =  IntelRealsense.transformation_matrix(0, 0, 0)
# while True:
#     # processing time
#     pp_x, pp_y, pp_depth, pp_k, capture = IntelRealsense.pingpong_detection(display = True)
#     processing = capture - last_capture
#     v_x, v_y, v_depth, a_x, a_y, STATE = IntelRealsense.pingpong_velocity(STATE, pp_x, pp_y, pp_depth, capture, display = False)
#     last_capture = capture
