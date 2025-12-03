
<div align="center">

# Vision-Detection-Engine
### Real-time Webcam • Image • Video 
Built with **Python + YOLOv3 + OpenCV**

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange?style=for-the-badge)
![YOLO](https://img.shields.io/badge/YOLO-v3-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Maintained](https://img.shields.io/badge/Maintained-Yes-success?style=for-the-badge)

</div>

---

### 📄 Research Background & Motivation  

The Visionary Eye project was introduced as a system to serve as a “visionary eye” for blind or visually impaired persons. The main aspirations include:  

- Mimicking human vision by recognizing and describing objects/scenes from camera inputs (image, video, real-time). :contentReference[oaicite:1]{index=1}  
- Providing auditory or other assistive outputs — making the environment perceivable to users without vision. :contentReference[oaicite:2]{index=2}  
- Offering a real-time, low-latency solution that can be reasonably deployed, potentially on portable devices (smartphone, embedded board, wearable) to support daily navigation and environment awareness. :contentReference[oaicite:3]{index=3}  

This repository bridges research and implementation: it leverages object detection (via YOLO + OpenCV) to realize a core component of the Visionary Eye concept.



### 🚀 What This Project Offers  

- **Image-based Object Detection** — Analyze any still image and detect objects present in it.  
- **Video File Detection** — Process video files frame-by-frame to detect objects throughout duration.  
- **Real-time Webcam Detection** — Live object detection via webcam/camera stream — foundation for real-time assistive feedback.  
- **Auto Saving of Results** — All processed outputs (images/videos) with bounding boxes are saved for review or further processing.  
- **Modular Codebase** — Easy to extend: you can plug in additional modules (text-to-speech, distance sensors, wearable interface) to build towards a full assistive system like in the research.  
- **Open-Source, Extensible** — MIT-licensed, ready for contributions, improvements, upgrades (e.g. newer detection models, tracking, captioning, audio feedback).  



### 📌 Overview

This repository provides a **single, unified implementation** of **YOLOv3-based object detection** using **OpenCV (cv2)**.  
It supports:

- 🖼️ **Image Object Detection**
- 🎞️ **Video File Object Detection**
- 📡 **Real-time Webcam Object Detection**
- 💾 **Auto-saving output with bounding boxes**
- 🎯 **Supports all 80 COCO classes**

All three detection modes are handled inside **one clean Python script (`yolo_unified.py`)**, making it extremely easy to run, maintain, and extend.



### ✨ Features

##### Core Features
- Unified script for Image, Video & Webcam detection  
- YOLOv3 Deep Learning model with OpenCV DNN  
- Efficient, real-time object detection  
- Class names + confidence score overlays  
- Non-Max Suppression (NMS) for accurate detection  
- Adjustable detection thresholds  
- Auto-saves all processed image & video outputs  

##### Project Design Highlights
- Well-structured directory architecture  
- Clean, modular, scalable Python code  
- Beginner-friendly yet production-ready  
- Easy model swapping (YOLOv4, YOLOv5, YOLOv8 upgrade possible)  
- Fully open-source — MIT Licensed  



### 📁 Folder Structure

```
.Vision-Detection-Engine
├── README.md
├── LICENSE
├── requirements.txt
├── vision_detection_engine.py
├── model/
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   └── coco.names
└── src/
    ├── docs/
    │   └── ruchir-shah-awarded-research-paper.pdf
    └── media/
        └── live-vision-detection.webp

```



### 🔧 Installation & Setup

##### 1. Clone the Repository
```
git clone https://github.com/TheRuchirShah/Vision-Detection-Engine.git
cd Vision-Detection-Engine
```



##### 2. Install Dependencies

```
pip install -r requirements.txt
```

Required modules:

* opencv-python
* numpy


##### 3. Download YOLO Model Files

If not included, download them manually:

File	Description
yolov3.cfg	Model configuration
yolov3.weights	Trained YOLOv3 weights
coco.names	COCO dataset class labels

YOLOv3 weights (official):
https://pjreddie.com/media/files/yolov3.weights



Place all files inside:
```
model/
```


### ▶️ Usage Guide
🖼️ Run Image Detection
```
python yolo_unified.py --image input/images/sample.jpg
```

##### Run Video Detection
```
python yolo_unified.py --video input/videos/sample.mp4
```

##### Run Real-time Webcam Detection
```
python yolo_unified.py --webcam
```

##### Output

Processed files are saved here:
```
output/images/
output/videos/
```



### 🎯 How It Works (Technical Deep-Dive)
##### 1️⃣ Preprocessing

* Input image/video frames are converted to a YOLO-compatible blob
* Normalized, resized to 416×416, channels swapped

##### 2️⃣ DNN Forward Pass

Using OpenCV’s cv2.dnn module:
* YOLO returns bounding boxes + class probabilities
* Thresholds applied for confidence filtering

##### 3️⃣ Post-processing

* Non-Max Suppression (NMS) eliminates overlapping boxes
* Best prediction retained
* Class label + confidence drawn on the frame

##### 4️⃣ Real-time Performance

* OpenCV’s DNN backend makes inference extremely fast on CPU.
* GPU acceleration can also be enabled.

### 📊 Supported Classes

All 80 classes from the COCO dataset including:

🛵 person • car • bike • dog • cat • bus • truck • bottle • laptop • phone • chair
…and many more.



### 🤝 Contributing

Contributions are welcome!
Ideas for improvement:

* Add YOLOv4 / YOLOv8 support
* Add object tracking (DeepSORT / SORT)
* Build a GUI (Tkinter / PyQt / Streamlit)
* Add FPS benchmarking

Open a PR or issue anytime.

### 📄 License

This project is open-source under the MIT License.
See the full license file.

### 👤 Author

Ruchir Shah
UI/UX Designer & Developer
<a href="https://ruchir-website.vercel.app/" target="_blank">Website (ruchir-website.vercel.app/)</a>

<div align="center">

⭐ Ruchir Shah ⭐

</div> 
