import sys
import argparse
from PIL import Image
from matplotlib import pyplot as plt
import time
import numpy as np
import cv2
import cv2.aruco as aruco
import threading
from ctypes import cdll
from checkerboard import detect_checkerboard
import ctypes
from numpy.ctypeslib import ndpointer
import math
from ext_calib_optimizer import ping_pong_optimize
from indydcp_client import IndyDCPClient
bind_ip = "192.168.0.6"   
server_ip = "192.168.0.7"
robot_name = "NRMK-Indy7"
indy = IndyDCPClient(bind_ip, server_ip, robot_name) 
indy.connect()


lib = cdll.LoadLibrary('./viewer_opengl.so')
st = lib.Foo_start
t0 = threading.Thread(target=st)
t0.start()
end = lib.Foo_end
dataread =lib.Foo_dataread
dataread_color =lib.Foo_dataread_color
dataread_depth =lib.Foo_dataread_depth
dataread_color_to_depth =lib.Foo_dataread_color_to_depth
dataread.restype = ndpointer(dtype=ctypes.c_uint8, shape=(720,1280,2))
dataread_color.restype = ndpointer(dtype=ctypes.c_uint8, shape=(720,1280,4))
dataread_depth.restype = ndpointer(dtype=ctypes.c_uint16, shape=(512,512))#ctypes.POINTE
dataread_color_to_depth.restype = ndpointer(dtype=ctypes.c_uint8, shape=(512,512,4))

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

checkrboard_size = (9, 6) 
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 2)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 2)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 2)
    return img
def draw_cube(img,corners,imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # 바닥 부분을 초록색으로
    img = cv2.drawContours(img,[imgpts[:4]],-1,(0,255,0),-3)

    # 기둥은 파란색으로
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img,tuple(imgpts[i]),tuple(imgpts[j]),(255,0,0),3)

    # 위의 층은 빨간색으로
    img = cv2.drawContours(img,[imgpts[4:]],-1,(0,0,255),3)

    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
axis = np.float32([[0,0,0],[0,3,0],[3,3,0],[3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])
def save_task():
    global T_we_
    global T_cp_
    global toggle
    toggle =0
    all_T_we= []
    all_T_cp= []
    while 1:
      print("Press Enter, x is end      :",len(all_T_we),len(all_T_cp))
      x=input()
      if x == 'x':
          toggle = 1
          break;
      all_T_we.append(T_we_)
      all_T_cp.append(T_cp_)
      
      time.sleep(1)
    T_wc, T_ep, residual = ping_pong_optimize(np.array(all_T_we), np.array(all_T_cp), 100000, 1e-6)
    print('T_wc:-----------------------\n', T_wc)
    print('------------------------------')
    #T_cw = inv(T_wc)

    
def detect_img():
    global img
    global n
    global gray
    global objpoints
    global imgpoints
    global T_we_
    global T_cp_
    global toggle
    toggle = 0
    with np.load('B.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('arr_0','arr_1','arr_2','arr_3')]
    print(mtx)
  
    print(dist)
    time.sleep(1)
    indy.change_to_direct_teaching()
    while True:
      if toggle ==1:
          break;
      try:
        try:
          color_img = np.array(dataread_color(),dtype=np.uint8)
          
          _img = cv2.cvtColor(color_img.copy(),cv2.COLOR_BGR2RGB)
          img = cv2.cvtColor(_img.copy(),cv2.COLOR_RGB2BGR)
          #img = cv2.resize(_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
          gray = cv2.cvtColor(img.copy(),cv2.COLOR_RGB2GRAY)
          corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
          frame_markers = aruco.drawDetectedMarkers(gray.copy(), corners, ids)
          cv2.imshow('img',frame_markers)
          cv2.waitKey(1) 
          #ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
          '''
          ret = True
          if ret == True:
              corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
              # Find the rotation and translation vectors.
              ret,rvecs, tvecs, inliers =cv2.solvePnPRansac(objp, corners2, mtx, dist)

              # project 3D points to image plane
              imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
              #print(mtx)
              #print(rvecs)
              T_we = np.zeros((4, 4), dtype=float)
              T_we[0:3,0:3]=cv2.Rodrigues(rvecs)[0]
              T_we[0,3]=np.array(-1*tvecs[0]*0.01)
              T_we[1,3]=np.array(-1*tvecs[1]*0.01)
              T_we[2,3]=np.array(-1*tvecs[2]*0.01)

              T_we[3,3]=1
              T_cp = np.empty((4, 4), dtype=float)
              Rz = np.zeros((3,3),dtype=float)
              Ry = np.zeros((3,3),dtype=float)
              Rx = np.zeros((3,3),dtype=float)
              pos = np.array(indy.get_task_pos(),dtype=float)
              Rz[0,0] = math.cos(pos[5]/180*math.pi)
              Rz[0,1] = -math.sin(pos[5]/180*math.pi)
              Rz[1,0] = math.sin(pos[5]/180*math.pi)
              Rz[1,1] = math.cos(pos[5]/180*math.pi)
              Rz[2,2] = 1

              Ry[0,0] = math.cos(pos[4]/180*math.pi)
              Ry[0,2] = math.sin(pos[4]/180*math.pi)
              Ry[2,0] = -math.sin(pos[4]/180*math.pi)
              Ry[2,2] = math.cos(pos[4]/180*math.pi)
              Ry[1,1] = 1
 
              Rx[1,1] = math.cos(pos[3]/180*math.pi)
              Rx[1,2] = -math.sin(pos[3]/180*math.pi)
              Rx[2,1] = math.sin(pos[3]/180*math.pi)
              Rx[2,2] = math.cos(pos[3]/180*math.pi)
              Rx[0,0] = 1
              R = np.matmul(np.matmul(Rz,Ry),Rx)

              T_cp[0:3,0:3] =R
              T_cp[0,3] =pos[0]
              T_cp[1,3] =pos[1]
              T_cp[2,3] =pos[2]
              T_cp[3,3] =1

          
              T_we_ = T_we.copy()
              T_cp_ = T_cp.copy()
              print(T_we_)
              print(T_cp_)
              #print(tvecs)
              '''

              
        except Exception as ex:
          print(ex)
          pass

      except:
        end()
    ##   

FLAGS = None

if __name__ == '__main__':
    t1 = threading.Thread(target=detect_img)
    t1.start()
   
    t2 = threading.Thread(target=save_task)
    t2.start()
