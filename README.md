# sign_talk_mvp

**sign_talk** is a real-time sign language recognition MVP designed to reduce communication barriers for deaf and hard-of-hearing individuals. The system works using any camera-enabled device and does not require gloves, sensors, or per-user calibration.

This project was built as a Minimum Viable Product (MVP) for the **Microsoft Imagine Cup (Scale Track)**.

---

##  Problem Statement
Sign language communication often requires interpreters or expensive hardware-based solutions. Existing AI systems struggle with variations in hand size, orientation, and signing styles, making them unreliable for real-world use.

---

## Solution Overview
sign_talk uses computer vision and metric learning to recognize signs by comparing normalized hand landmark geometry instead of relying on rigid classification models.

**Core ideas:**
* Hand landmark extraction using **MediaPipe**
* Geometry-based normalization using relative distances and ratios
* Orientation-robust gesture embeddings
* Nearest-neighbor similarity matching for recognition
* Cloud-based inference using **Microsoft Azure**

---

##  Key Features
* **Real-time** sign recognition using a standard camera
* Works across different users **without retraining**
* No special hardware required
* **Cloud-hosted backend** for device independence
* Lightweight and scalable design

---

##  Technology Stack
* **Language:** Python
* **Frameworks:** FastAPI, Streamlit (demo frontend)
* **Libraries:** MediaPipe, OpenCV
* **Cloud:** Azure App Service

---

##  Repository Structure
```text
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

##  Backend Setup (Local)
cd sign_language_backend
python -m venv venv
# Windows:
venv\Scripts\activate   
# Linux/Mac:
# source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload

The API will be available at: http://127.0.0.1:8000

---

## Frontend Demo (Local)
cd sign_language_frontend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

---

## Azure Deployment
The backend is deployed on Microsoft Azure App Service (Linux). FastAPI is served using Gunicorn with Uvicorn workers.

Example deployed endpoint: sign-language-api-utkar-g6csdrhweadcgfft.centralindia-01.azurewebsites.net

---

## **How It Works**
YT links of our project : 
https://youtu.be/zr85-YRkfeU 
https://youtu.be/mYMCPn2ZX40 
https://youtu.be/qWjBuR88VjQ

---

## How Anyone Can Use It
1. Open the application on any camera-enabled device.
2. Perform a sign in front of the camera.
3. Hand landmarks are extracted and normalized automatically.
4.The gesture is matched against known signs using similarity search.
5.The recognized sign is returned and displayed in real time.

---

## Future Roadmap
**Expanded Vocabulary:** Adding support for full phrases and dynamic gestures.
**Sentence Construction:** Implementing NLP to convert sequences of signs into grammatically correct text.
**Mobile App Integration:** Native iOS/Android apps for better accessibility on the go.
