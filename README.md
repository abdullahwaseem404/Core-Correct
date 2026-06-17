# 🧍 Core-Correct: Real-Time Pose Estimation System

An interactive **real-time pose estimation application** built with Streamlit, OpenCV, MediaPipe, and MoveNet. This system detects human body keypoints, calculates joint angles, and provides live posture feedback.

---

## 🚀 Features

* 📷 Real-time webcam pose detection
* 🤖 Dual model support:
  * MediaPipe Pose
  * MoveNet
* 📐 Joint angle calculation
* 🔄 Temporal smoothing for stable tracking
* ⚡ FPS monitoring
* ⚠️ Posture correction alerts
* 🎛️ Interactive Streamlit controls

---

## 🧠 Models Used
### 🔹 MediaPipe Pose
* Fast and lightweight
* Good for real-time applications
### 🔹 MoveNet (Lightning)
* Deep learning-based pose estimation
* More robust and accurate

---

## ⚙️ Installation

```bash id="w72kq1"
git clone https://github.com/abdullahwaseem404/Core-Correct.git
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit app:
```bash id="z91k2l"
streamlit run app.py
```

---

## 📊 Performance

* Real-time FPS tracking
* Adjustable smoothing for stability
* Optimized for lightweight inference

---
