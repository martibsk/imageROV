import detector
import cv2
from imutils.video import FPS
import numpy as np
import darknet
import os
import detectorCPU





if __name__ == '__main__':

    model = 'sylinder'
    input_frame = 'sylinder.mp4'
    useGPU = True

    if useGPU:
        netMain, metaMain = detector.init_yolo(model)
        vs = detector.video2image(input_frame)
    else:
        vs = cv2.VideoCapture(input_frame)

        # Derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join(["models", model, "yolov3.weights"])
        configPath = os.path.sep.join(["models", model, "yolov3.cfg"])
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        # Determine only the 'output' layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    fps = FPS().start()

    while True:

        ret, frame_read = vs.read()
        # If frame not grabbed, break out of loop
        if not ret:
            break
        if useGPU:
            detections = detector.YOLO(frame_read, netMain, metaMain)
        else:
            detections = detectorCPU.detect(frame_read, net, ln)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        print(detections)
        #detector.printInfo(detections)
        # Update the FPS counter
        fps.update()



    # Stop the timer and display FPS information
    fps.stop()
    print("\n[INFO] elapsed time: {:2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    vs.release()
    vs.release()
