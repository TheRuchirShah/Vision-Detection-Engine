import cv2
import numpy as np
import argparse
import time
import os

# ===============================
# Vision Detection Engine – Unified Object Detection
# ===============================

# Paths for model files
MODEL_DIR = "model"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov3.weights")
CFG_PATH = os.path.join(MODEL_DIR, "yolov3.cfg")
NAMES_PATH = os.path.join(MODEL_DIR, "coco.names")

# Load class names
with open(NAMES_PATH, "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Initialize colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load Vision Detection Engine (YOLO backend)
print("🔍 Loading Vision Detection Engine...")
net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)

# Get layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ===============================
# DRAW BOUNDING BOX FUNCTION
# ===============================
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ===============================
# PROCESS FRAME FUNCTION
# ===============================
def process_frame(img):
    height, width = img.shape[:2]

    # Create image blob
    blob = cv2.dnn.blobFromImage(
        img, 1/255.0, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)

    # Run Vision Engine
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Detecting objects
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw results
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            draw_prediction(img, class_ids[i], confidences[i], x, y, x + w, y + h)

    return img

# ===============================
# MAIN LOGIC
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Vision Detection Engine – Unified Detector")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--video", help="Path to input video file")
    parser.add_argument("--webcam", help="Use webcam index", nargs='?', const=0, type=int)
    args = parser.parse_args()

    # -------------------------------
    # IMAGE MODE
    # -------------------------------
    if args.image:
        print("🖼 Processing Image...")
        img = cv2.imread(args.image)
        if img is None:
            print("❌ Could not load image.")
            return

        result = process_frame(img)
        os.makedirs("output/images", exist_ok=True)
        output_path = "output/images/result.jpg"
        cv2.imwrite(output_path, result)
        print(f"✅ Saved output: {output_path}")

        cv2.imshow("Vision Detection Engine – Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -------------------------------
    # VIDEO MODE
    # -------------------------------
    elif args.video:
        print("🎥 Processing Video...")
        cap = cv2.VideoCapture(args.video)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs("output/videos", exist_ok=True)
        out = cv2.VideoWriter("output/videos/result.mp4", fourcc, 30.0,
                              (int(cap.get(3)), int(cap.get(4))))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = process_frame(frame)
            out.write(result)
            cv2.imshow("Vision Detection Engine – Video", result)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("✅ Video processing completed.")

    # -------------------------------
    # REAL-TIME WEBCAM MODE
    # -------------------------------
    elif args.webcam is not None:
        print("📹 Starting Real-time Detection...")
        cap = cv2.VideoCapture(args.webcam)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = process_frame(frame)
            cv2.imshow("Vision Detection Engine – Webcam", result)

            if cv2.waitKey(1) == 27:  # ESC key
                break

        cap.release()
        cv2.destroyAllWindows()
        print("🛑 Webcam stream stopped.")

    else:
        print("❌ No input provided. Use --image, --video, or --webcam.")


# Run program
if __name__ == "__main__":
    main()
