import cv2

print("OpenCV Version:", cv2.__version__)
for i in range(3):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Camera index {i}: Failed to open.")
    else:
        ok, frame = cap.read()
        print(f"Camera index {i}: Opened successfully. Frame read success: {ok}")
        if ok and frame is not None:
            print(f"  Frame shape: {frame.shape}")
        cap.release()
