import cv2
import numpy as np

# Load YOLO model
et = cv2.dnn.readNet("src/ai_analysis/yolov3.weights", "src/ai_analysis/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = open("src/ai_analysis/coco.names").read().strip().split("\n")

# Open camera feed (change 0 to your webcam index or drone feed URL)
cap = cv2.VideoCapture(0)

# Read first frame for motion detection
_, prev_frame = cap.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)

night_vision = False  # Default mode

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for better orientation
    frame = cv2.flip(frame, 1)

    # Motion Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    frame_diff = cv2.absdiff(prev_frame, gray)
    _, thresh = cv2.threshold(frame_diff, 7, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust sensitivity
            motion_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if motion_detected:
        cv2.putText(frame, "MOTION DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # YOLO Object Detection
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                # Draw bounding box
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"{classes[class_id]}: {int(confidence * 100)}%"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Apply Night Vision Effect if enabled
    if night_vision:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)  # Enhance contrast
        frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_WINTER)  # Fake night vision look

    cv2.imshow("Military Ops Feed (Press 'N' for Night Vision, 'Q' to Quit)", frame)

    # Update previous frame for motion detection
    prev_frame = gray

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):  # Toggle night vision
        night_vision = not night_vision
    if key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
