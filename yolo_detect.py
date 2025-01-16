from ultralytics import YOLO
import cv2

def detect_objects_from_video(source=0, model_name="yolov8s"):
    model = YOLO(model_name)  # Load YOLOv8 model

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {source}")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot fetch the frame.")
            break

        # Resize frame to reduce processing time (optional)
        frame = cv2.resize(frame, (640, 480))

        # Perform object detection on the current frame
        results = model(frame)

        # Get the annotated frame with detections
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow("Object Detection", annotated_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # For video file, provide the path (e.g., "video.mp4")
     # Use webcam (source=0 represents the default webcam)
    detect_objects_from_video(source="C:\\Users\\its4e\\Downloads\\Obj\\yoloobj\\video\\2932301-uhd_4096_2160_24fps (2).mp4", model_name="yolov8s")  # Use "yolov8s" for faster detection

    # Uncomment to test with a video file path
    # detect_objects_from_video(source="path/to/video.mp4", model_name="yolov8s")
