from collections import defaultdict
import cv2
import numpy as np
import argparse
from ultralytics import YOLO

NAMES = ['bike', 'bus', 'car', 'sign', 'traffic_signal', 'truck']
COLOR = (0, 0, 255)

def track_video(video_path, output_path, weights, tracker):
    model = YOLO(weights)

    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: [])

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, tracker=tracker)
            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)

                # Draw boxes and IDs on the frame
                for box, obj_id, label in zip(boxes, ids, classes):
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), COLOR, 2)
                    cv2.putText(
                        frame,
                        f"# {obj_id}: {NAMES[label]}",
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        COLOR,
                        2
                    )

            cv2.imshow('Video Feed', frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 Traffic Monitoring - Video Tracking")
    parser.add_argument("--input", type=str, default="Road_Traffic.mp4", help="Path to the input video file")
    parser.add_argument("--output", type=str, default="./videos/TrafficCam_video.mp4", help="Path to save the output video file")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to the model weights file")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Tracker configuration file")

    args = parser.parse_args()

    processed_video_path = track_video(args.input, args.output, args.weights, args.tracker)
    print(f"Processed video saved to: {processed_video_path}")
