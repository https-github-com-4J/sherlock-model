import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

img = "./test.jpg"

confThreshold = 0.5
nmsThreshold = 0.4

inpWidth = 416
inpHeight = 416

classesFile = "files/classes.names";

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = "files/darknet-yolov3.cfg";
modelWeights = "files/lapi.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def cropPred(frame, left, top, right, bottom):
    diff_height = bottom - top
    diff_width = right - left
    crop_img = frame[top:top+diff_height, left:left+diff_width]
    # cv.imshow('Image', crop_img)

    outputFile = "out_" + str(img)
    cv.imwrite(outputFile, crop_img.astype(np.uint8));

def postprocess(frame, outs):
    confThreshold = 0.5
    nmsThreshold = 0.4

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        # print("out.shape : ", out.shape)
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cropPred(frame, left, top, left + width, top + height)

def execute():
    cap = cv.VideoCapture(img)
    hasFrame, frame = cap.read()

    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))

    postprocess(frame, outs)

