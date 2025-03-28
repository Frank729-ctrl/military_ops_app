import cv2
import numpy as np

def start_video_stream(video_source, night_vision=False):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    _, prev_frame = cap.read()  
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't receive frame.")
            break

        frame = cv2.flip(frame, 1)  
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)  

        # Compute difference between frames
        frame_diff = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 7, 255, cv2.THRESH_BINARY)

        #APPlying dilation to make movement clearer
        thresh = cv2.dilate(thresh, None, iterations=5)

        # Find motion contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 5:  # Ignore small movements
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if motion_detected:
            print("⚠️ Motion Detected!")  # Debug output
            cv2.putText(frame, "MOTION DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if night_vision:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.equalizeHist(frame)
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_WINTER)

        cv2.imshow("Drone Feed (Press 'n' for Night Vision, 'q' to Quit)", frame)

        if cv2.getWindowProperty("Drone Feed (Press 'n' for Night Vision, 'q' to Quit)", cv2.WND_PROP_VISIBLE) < 1:
            break  # Detects if window is closed manually


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            night_vision = not night_vision

        prev_frame = gray  # Update previous frame

    cap.release()
    cv2.destroyAllWindows()
