import detector
import cv2
from imutils.video import FPS
import numpy as np
import darknet






if __name__ == '__main__':
    model = 'finalData'
    netMain, metaMain = detector.init_yolo(model)

    input_frame = 'sylinder.mp4'
    vs = detector.video2image(input_frame)

    fps = FPS().start()

    while True:

        ret, frame_read = vs.read()
        # If frame not grabbed, break out of loop
        if not ret:
            break
        detections = detector.YOLO(frame_read, netMain, metaMain)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        #detector.printInfo(detections)
        # Update the FPS counter
        fps.update()



    # Stop the timer and display FPS information
    fps.stop()
    print("\n[INFO] elapsed time: {:2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    vs.release()
    vs.release()
