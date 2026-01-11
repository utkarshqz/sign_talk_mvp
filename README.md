# sign_talk_mvp

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Computer%20Vision-orange)
![Azure](https://img.shields.io/badge/Microsoft%20Azure-Cloud-blue)
![Status](https://img.shields.io/badge/Status-MVP-success)

---

## 🚀 About sign_talk

**sign_talk** is a **real-time, camera-based sign language recognition system** designed to reduce communication barriers for **deaf and hard-of-hearing individuals**.

Unlike traditional solutions that rely on gloves, sensors, or user-specific calibration, **sign_talk works instantly on any camera-enabled device**, making it highly accessible, scalable, and deployment-friendly.

This project was built as a **Minimum Viable Product (MVP)** for the  
**Microsoft Imagine Cup – Scale Track**, with a strong focus on **real-world usability and cloud scalability**.

---

## 🧩 Problem Statement

Sign language communication often depends on:
- Human interpreters (costly and not always available)
- Hardware-based systems (gloves, sensors, depth cameras)
- AI models that fail under real-world variations such as:
  - Different hand sizes
  - Orientation changes
  - Personal signing styles

These limitations make existing solutions **inaccessible, expensive, and unreliable** for daily use.

---

## 💡 Our Solution

**sign_talk** approaches sign recognition as a **geometry and similarity problem**, rather than a rigid classification task.

Instead of memorizing gestures, the system:
- Extracts **hand landmarks**
- Normalizes their **relative geometry**
- Creates **orientation-robust embeddings**
- Matches gestures using **similarity search**

This enables **cross-user generalization without retraining**.

### Core Innovations
- Hand landmark extraction using **MediaPipe**
- Geometry-based normalization using distances and ratios
- Orientation-invariant gesture embeddings
- Nearest-neighbor similarity matching
- Cloud-hosted inference using **Microsoft Azure**

---

## ✨ Key Features

- ⚡ **Real-time** sign recognition
- 👥 Works across different users **without retraining**
- 🎥 Uses only a standard RGB camera
- ☁️ **Cloud-based backend** for device independence
- 🧱 Lightweight, modular, and scalable MVP architecture

---

## 🛠 Technology Stack

- **Programming Language:** Python
- **Backend Framework:** FastAPI
- **Frontend Demo:** Streamlit
- **Computer Vision:** MediaPipe, OpenCV
- **Cloud Platform:** Microsoft Azure App Service

---

## 📁 Repository Structure

sign_talk_mvp/
│
├── sign_language_backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── recognizer/
│   └── reference_db/
│
├── sign_language_frontend/
│   ├── app.py
│   └── requirements.txt
│
├── sign_language_mvp/
│
└── README.md

---

## ⚙️ Backend Setup (Local)

cd sign_language_backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
# source venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --reload

Backend API will be available at:
http://127.0.0.1:8000

---

## 🖥 Frontend Demo (Local)

cd sign_language_frontend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
# source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py

---

## ☁️ Azure Deployment

The backend is deployed on **Microsoft Azure App Service (Linux)** using:
- FastAPI
- Gunicorn
- Uvicorn workers

Example deployed endpoint:
https://sign-language-api-utkar-g6csdrhweadcgfft.centralindia-01.azurewebsites.net

---

## 🎥 Project Demonstration

YouTube Demo Videos:
- https://youtu.be/zr85-YRkfeU
- https://youtu.be/mYMCPn2ZX40
- https://youtu.be/qWjBuR88VjQ

---

## 🧭 How It Works

1. User opens the application on any camera-enabled device
2. Performs a sign in front of the camera
3. Hand landmarks are detected and normalized
4. Gesture geometry is compared with stored references
5. The closest matching sign is identified
6. Result is displayed in real time

---

## 🌍 Impact & Use Cases

- Deaf–hearing communication bridge
- Inclusive education tools
- Public service kiosks
- Telemedicine & accessibility platforms
- Smart assistants with gesture input

---

## 🔮 Future Roadmap

- Expanded vocabulary and dynamic gestures
- NLP-based sentence construction
- Native Android & iOS apps
- Multilingual sign language support
- Offline edge deployment
