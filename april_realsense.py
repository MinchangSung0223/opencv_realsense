import sys
import argparse
from PIL import Image
from matplotlib import pyplot as plt
import time
import numpy as np
import cv2
import threading
import pyrealsense2 as rs
import cv2.aruco as aruco
from checkerboard import detect_checkerboard
import ctypes
from numpy.ctypeslib import ndpointer
import math
from ext_calib_optimizer import ping_pong_optimize
pipeline = rs.pipeline()
checkrboard_size = (9, 6) 
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
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
    global objpoints
    global corners2
    global imgpoints
    global gray
    toggle =0
    objpoints = []
    while 1:
      print("Press Enter, x is end      :",len(objpoints))
      x=input()
      ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
      if ret == True:
         objpoints.append(objp)
         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
         imgpoints.append(corners2)
      if x == 'x':
          toggle = 1
          break;
      time.sleep(1)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    np.savez('B.npz',mtx)


marker_size_in_mm=50    
def detect_img():
    global img
    global n
    global gray
    global objpoints
    global imgpoints
    global T_we_
    global T_cp_
    global toggle
    global corners2
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    toggle = 0
    objpoints = []
    imgpoints = []
    with np.load('B.npz') as X:
        mtx, dist= [X[i] for i in ('arr_0','arr_1')]
    while True:
      if toggle ==1:
          break;
      try:
        try:

          frames = pipeline.wait_for_frames()
          depth_frame = frames.get_depth_frame()
          color_frame = frames.get_color_frame()
          if not depth_frame or not color_frame:
              continue
          depth_image = np.asanyarray(depth_frame.get_data())
          color_img= np.asanyarray(color_frame.get_data())
          _img = cv2.cvtColor(color_img.copy(),cv2.COLOR_BGR2RGB)
          img = cv2.cvtColor(_img.copy(),cv2.COLOR_RGB2BGR)
          gray = cv2.cvtColor(img.copy(),cv2.COLOR_RGB2GRAY)
          corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
          frame_markers = aruco.drawDetectedMarkers(gray.copy(), corners, ids)
      
          if np.all(ids != None):
                rotation_vector, translation_vector, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], marker_size_in_mm / 1000, mtx, dist)
                R, _ = cv2.Rodrigues(rotation_vector)
                print(R)
                print(translation_vector)
                img = cv2.aruco.drawAxis(img.copy(), mtx, dist,rotation_vector[0], translation_vector[0], 0.1)
                img= cv2.aruco.drawDetectedMarkers(img, corners, ids)
          cv2.imshow('img', img)
          cv2.waitKey(1) 
              
        except Exception as ex:
          print(ex)
          pass

      except:
        pipeline.stop()
    ##   
 

FLAGS = None

if __name__ == '__main__':
    t1 = threading.Thread(target=detect_img)
    t1.start()
   
    #t2 = threading.Thread(target=save_task)
    #t2.start()
