import cv2

video_path = "Wallet.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Couldn't open video.")
    exit()

tracker = cv2.TrackerCSRT.create() #Using CSRT Tracker from CV2

ret, frame = cap.read()

bbox = cv2.selectROI("Selection", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Selection")

#object tracking initialization
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()