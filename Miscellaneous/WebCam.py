import cv2

vs = cv2.VideoCapture(0)
while(vs.isOpened()):
    ok, frame = vs.read()
    if ok:
        cv2.imshow("Video LiveFeed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
vs.release()
cv2.destroyAllWindows()
