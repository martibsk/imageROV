from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import argparse
import sys
from imutils.video import FPS


def video2image(video_path):
    vs = cv2.VideoCapture(video_path)
    vs.set(3, 1280)
    vs.set(4, 720)
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")
    return vs


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

def printInfo(detections):
    # Check if there is any detections
    if not detections:
        return
    print('[INFO] num detections: {}'.format(len(detections)))
    print('\t Object \t Confidence \t (x,y) \t\t  (w,h)')
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        print('\t{}\t   {}% \t ({},{}) \t ({},{})'.format(detection[0].decode(),
            str(round(detection[1]*100, 2)), int(x), int(y), int(w), int(h)))


    # Print out desired information
    #sys.stdout.write("\r[INFO] {}: {}".format(detection[0].decode(),str(round(detection[1]*100, 2))))
    #sys.stdout.flush()

netMain = None
metaMain = None
altNames = None


def init_yolo(model):
    global metaMain, netMain, altNames
    configPath = os.path.sep.join(["models", model, "yolov3.cfg"])
    weightPath = os.path.sep.join(["models", model, "yolov3.weights"])
    metaPath = os.path.sep.join(["models", model, model + ".data"])
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    return netMain, metaMain




def YOLO(input_frame, netMain, metaMain):

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    frame_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb,
                               (darknet.network_width(netMain),
                                darknet.network_height(netMain)),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

    detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
    image = cvDrawBoxes(detections, frame_resized)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    cv2.imshow('Demo', image)

    return detections

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True,
        help = 'input file')
    ap.add_argument('--output', type=str, default='output_video.avi',
        help = 'output file')
    ap.add_argument('--model', required=True,
        help = 'Which model will be used')
    args = vars(ap.parse_args())

    YOLO()
