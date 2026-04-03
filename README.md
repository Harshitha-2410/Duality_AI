# 🏜️ Duality AI — Desert Segmentation Dashboard

A full-stack AI-powered dashboard for **real-time desert scene segmentation**, powered by **DeepLabV3+ with ResNet-101** and a Flask backend.

---

## 📥 Model Download

⚠️ The trained model is not included in this repository due to GitHub file size limits.

👉 Download the model from here:
**https://drive.google.com/file/d/1i0tjy8SpslN4qMB6WLA4-FqAdoYDa3ps/view?usp=drive_link**

After downloading, place the file in:

```
backend/best_model.pth
```

---

## 📁 Folder Structure

```
desert_seg/
├── backend/
│   ├── app.py              ← Flask prediction server
│   ├── requirements.txt
│   └── best_model.pth      ← Download from link above and place here
└── frontend/
    └── index.html          ← Open this in a browser
```

---

## 🚀 Quick Start

### 1. Setup Backend

```bash
cd backend
pip install -r requirements.txt
```

---

### 2. Add the Model

Download the model and place it inside:

```
backend/best_model.pth
```

---

### 3. Run the Server

```bash
python app.py
```

✅ Server will start at:

```
http://localhost:5000
```

You should see:

```
✅ Model loaded successfully
```

---

### 4. Launch Frontend

Open:

```
frontend/index.html
```

in your browser (Chrome/Firefox recommended)

---
## 🔁 Reproducing Results

To reproduce the final model results:

### 1. Use the provided trained model

Download the model from the link above and place it in:

```
backend/best_model.pth
```

### 2. Run the backend server

```
python app.py
```

### 3. Test using sample images

* Upload any desert image via the frontend dashboard
* Or send a POST request to `/predict` API

### 4. Expected Results

* The model will generate a **segmentation mask**
* Metrics returned:

  * Mean IoU (accuracy of segmentation)
  * Per-class IoU
  * Inference time (ms)
  * Confidence score

### 5. Verification

* If model is loaded correctly → console shows:

  ```
  ✅ Model loaded successfully
  ```
* If model is missing → app runs in **DEMO mode**

---

### 📊 Expected Performance

* Mean IoU: ~0.50 – 0.80 (depends on input image)
* Inference time: ~30–100 ms (CPU)
* Higher IoU = better segmentation accuracy

## 🧠 Features

### 🔥 AI Capabilities

* Real-time **semantic segmentation**
* DeepLabV3+ with ResNet-101 backbone
* Multi-class desert terrain classification

### 📊 Live Metrics

* Mean IoU (Intersection over Union)
* Per-class IoU breakdown
* Inference time (real latency)
* Confidence score (softmax-based)
* Pixel-wise class distribution

### 💻 Dashboard Features

* Upload image → get instant segmentation
* Dynamic visualization of results
* Backend-connected live updates

---

## ⚙️ API Endpoints

| Endpoint           | Method | Description                    |
| ------------------ | ------ | ------------------------------ |
| `/health`          | GET    | Check server status            |
| `/model_info`      | GET    | Model metadata                 |
| `/predict`         | POST   | Image → segmentation + metrics |
| `/predict_with_gt` | POST   | Image + GT → IoU evaluation    |

---

## ⚡ Notes

* If `best_model.pth` is missing, the app runs in **DEMO mode**
* Ensure backend runs on `localhost:5000`
* For GPU support:

  ```bash
  nvidia-smi
  ```
* Compatible with CPU and CUDA

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Flask (Python)
* **Model:** DeepLabV3+ (ResNet-101)
* **Frameworks:** PyTorch, NumPy, OpenCV

---

## 🎯 Use Case

This project is designed for:

* Autonomous navigation systems
* Remote sensing analysis
* Disaster/environmental monitoring
* AI hackathons & research demos

---

## 🙌 Acknowledgements

* Duality AI Dataset
* PyTorch Team
* Open-source community

---

## 📌 Important

❗ The model file is intentionally excluded due to GitHub size restrictions (>100MB).
Please download it using the provided link.

---

## 💡 Future Improvements

* Deploy on cloud (AWS / Render)
* Add video stream segmentation
* Improve mAP & IoU with advanced architectures
* Add user authentication & history tracking

---

⭐ If you like this project, consider giving it a star!
