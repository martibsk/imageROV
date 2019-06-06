# Import the necessary packages
import numpy as np
import time
import cv2
import os
import imutils





def detectionLoop(image, net, ln):
    # Construct a blob from the input image and then perform a
    # forward pass of the YOLO object detector, giving us out
    # bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1./255, (416, 416),
        swapRB = True, crop = False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    (H, W) = image.shape[:2]

    # Initialize out lists of detected bounding boxes, confidence and
    # class IDs respectively
    boxes = []
    confidences = []
    classIDs = []
    detListTemp = []
    detectionList = list()


    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each of the detections
        for detection in output:
            # Extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter out weak predictions by ensuring the detected
            # probability is greated than the minimum probability
            if confidence > 0.2:
                # Scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO actually
                # returns center (x, y)-coordinates of the bounding box
                # followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x,y)-coordinates to derive the top and
                # left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update our list of bounding box coordinates, confidences
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                detListTemp.append([centerX, centerY, width, height])


    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    # Confidence = 0.5, threshold = 0.3
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
        0.3)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # Extract the bounding box coordinates
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])

            # Draw a bounding box rectangle and label on the image
            cv2.rectangle(image, (x,y), (x+w, y+h), [0,0,255], 2)
            text = "{}: {:.4f}".format('sylinder', confidences[i])
            cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0], 2)


            # add the detection to the output list
            detectionList.append(('sylinder', confidences[i], (detListTemp[i][0],
                detListTemp[i][1], detListTemp[i][2], detListTemp[i][3])))
            '''
            # Draw directional arrows
            if args["direction"].lower() in 'true yes y t yup ok':
                box_x = int(x + w/2)
                box_y = int(y + h/2)
                img_x = int((box_x + W/2) / 2)
                img_y = int((box_y + H/2) / 2)
                cv2.arrowedLine(image, (box_x, box_y), (img_x, img_y), [0,0,255], 2)

                if img_x - box_x > 50:
                    x_dir = 'left'
                elif img_x - box_x < -50:
                    x_dir = 'right'
                else:
                    x_dir = 'still'

                if img_y - box_y > 50:
                    y_dir = 'up'
                elif img_y - box_y < -50:
                    y_dir = 'down'
                else:
                    y_dir = 'still'
                print('Move camera {} and {}'.format(y_dir, x_dir))
            '''
    #print('Det: {}'.format(detection))
    #print('Box: {}'.format(boxes))
    #print('Conf: {}'.format(confidences))
    return detectionList

def detect(image, net, ln):

    detectionList = detectionLoop(image, net, ln)

    # Show the output image
    cv2.imshow("Image", image)

    return detectionList
