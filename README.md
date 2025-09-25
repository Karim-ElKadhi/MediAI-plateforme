# 🧠 MediAI Platform

📝 Abstract

The early detection of critical health conditions such as brain tumors and strokes plays a vital role in reducing mortality rates and improving patient outcomes. Traditionally, brain tumor classification is performed through invasive procedures such as biopsies, often conducted only during or after surgery. Non-invasive alternatives like magnetic resonance imaging (MRI) remain the standard diagnostic tool, but interpretation is highly dependent on radiologist expertise and can be influenced by subjectivity.
With the improvement of artificial intelligence and machine learning , automated systems can assist clinicians by providing faster, objective, and more accurate predictions. For brain tumor detection, convolutional neural networks (CNNs) have shown remarkable performance in medical image classification. 

MediAI is an **AI-powered healthcare assistant** that provides two core modules:  

- 🏥 **Stroke Risk Assessment** → Predicts the likelihood of a patient experiencing a stroke using clinical and lifestyle data using real time data.  
- 🩺 **Brain Tumor Detection** → Analyzes MRI images with a CNN model to detect brain tumors with high accuracy.  

The platform is built with **Streamlit** for a clean and user-friendly interface.  

---

## ✨ Features

✅ Stroke prediction using a trained **CatBoost model** (`.sav`)  
✅ Brain tumor detection using a **CNN model** (`.h5`)  
✅ Modern dashboard interface with **tabs and cards**  
✅ Real-time medical predictions with simple input forms  
✅ Image upload support for MRI scans (JPG, PNG, JPEG, WEBP)  

---

## 📂 Project Structure

MediAI-platefrome/

│── datasets/

│   └── dataset/

│       └── healthcare-dataset-stroke-data.csv     # Stroke dataset

│

│── models/

│   ├── brain_tumor_model_v2.h5                    # CNN model for tumor detection

│   ├── tr_model_best.sav                          # CatBoost model for stroke prediction

│   ├── brain_tumor.ipynb                          # Notebook (tumor training/experiments)

│   ├── stroke.ipynb                               # Notebook (stroke training/experiments)

│   └── test.png                                   # Sample MRI test image

│                                        

│

│── utils/

│   └── extractor.py                               # Image preprocessing 

│


│── brain_tumor.py                                 # Tumor detection script

│── stroke.py                                      # Stroke prediction 

│── main.py                                        # Streamlit application (MediAI Platform)

│── README.md                                      # Documentation



---

## ⚙️ Installation

1. Clone the repository:


2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run main.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501) 🌐  

---

## 🏥 Modules

### Stroke Risk Assessment
- Input patient details: age, gender, hypertension, heart disease, glucose level, BMI, smoking status.  
- The model predicts **stroke likelihood** and provides feedback:
  - 🟢 No risk  
  - 🔴 High risk  

### Brain Tumor Detection
- Upload an **MRI image** (JPG, PNG, JPEG, WEBP).  
- The CNN model analyzes the scan and outputs:
  - ✅ **No Tumor detected**  
  - ⚠️ **Tumor detected**  

---

## 📊 Screenshots

### Stroke Risk Assessment
![Stroke Module](utils/Screenshot%202025-09-23%20223734.png)

### Brain Tumor Detection
![Tumor Module](utils/Screenshot%202025-09-23%20223759.png)

---

## 📦 Requirements

- Python 3.8+
- Streamlit
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn / Pickle  

(See `requirements.txt` for full details)



---

## 📜 License

This project is licensed under the MIT License.  
