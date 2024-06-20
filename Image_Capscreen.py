import time
import cv2

cap = cv2.VideoCapture(0)
i = 0
cap_comfirm = 0

while True:
    if cap.isOpened():
        # time.sleep(5) 
        # print("A")
        # cap_comfirm = 1           
        _ , frame = cap.read()
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'out2/image{i}.jpg',frame)
            i += 1
