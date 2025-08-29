


# 🩺 Breast Cancer Detection and Medical Chatbot (PyTorch +  Mistral API)

## 📌 Project Overview
This project combines **Deep Learning-based Breast Cancer detection** with a **medical chatbot** powered by the **Mistral API**.  
The system has two major components:
1. **Cancer Detection Model** → Detects breast cancer from medical images (microscopic scans / mammograms).  
2. **Medical Chatbot** → Provides actionable insights such as:
   - ✅ What to do if cancer is detected  
   - 📖 Detailed medical explanation of results  
   - 🏥 Suggested type of doctors to consult  
   - 🧬 Estimated stage of cancer  
   - ⏳ Approximate survival time if untreated  

⚠️ **Disclaimer**: This system is for **educational and research purposes only**. It is **not a replacement for professional medical diagnosis**.

---

## 📂 Project Structure

```
├── dataset\_sorted/
│   ├── train\_dataset/
│   │   ├── benign/
│   │   ├── malignant/
│   ├── test\_dataset/
│       ├── benign/
│       ├── malignant/
│
├── cancer\_train.py        # Training script (PyTorch CNN)
├── cancer\_inference.py    # Inference + chatbot pipeline
├── requirements.txt       # Dependencies
├── README.md              # Documentation (this file)
```


---

## ⚙️ Tech Stack
- **Python 3.9+**
- **PyTorch** → Deep Learning framework  
- **Torchvision** → Data transforms and datasets  
- **Streamlit** → Deployment with an interactive web UI  
- **Mistral API** → LLM-based chatbot for medical inference  
- **PIL (Pillow)** → Image handling  
- **CUDA/cuDNN** → GPU acceleration  

---

## 📊 Dataset Description
The model expects a dataset in the following format:

```
dataset\_sorted/
│
├── train\_dataset/
│   ├── benign/       # Non-cancerous tissue samples
│   ├── malignant/    # Cancerous tissue samples
│
├── test\_dataset/
├── benign/
├── malignant/
```


Each image is **resized to 224x224** and normalized to improve training stability.  

---

## 🧠 Model Architecture
The model (`PlantNetV2`) is a **custom CNN** designed for image classification:  

- **Input Layer** → RGB images (3 channels)  
- **Conv Layers**:  
  - Conv1 → 64 filters  
  - Conv2 → 128 filters  
  - Conv3 → 256 filters  
  - Conv4 → 512 filters  
- **Pooling** → MaxPool (2x2) after each conv  
- **Fully Connected Layers**:  
  - FC1 → 1024 neurons  
  - FC2 → 512 neurons  
  - FC3 → Output (2 classes: benign, malignant)  
- **Dropout** → 0.5 for regularization  
- **Activation** → ReLU  

---

## 🏋️ Training Details
- **Loss Function** → CrossEntropyLoss  
- **Optimizer** → Adam (lr=0.001)  
- **Batch Size** → 32  
- **Epochs** → 30  
- **Hardware** → GPU (CUDA enabled)  

Training command:
```
python model_create.py
````

The trained model is saved as:

```
breast_cancer_cnn.pth
```

---

## 📈 Evaluation Metrics

* **Accuracy** → % of correct classifications
* **Precision** → How many predicted positives are truly cancer
* **Recall (Sensitivity)** → How many real cancer cases are detected
* **F1-Score** → Balance between precision & recall
* **Confusion Matrix** → Visualizes classification quality

---

## 🤖 Chatbot + Medical Inference Integration

After classification, results are passed to the **Mistral API** chatbot for further explanation.

Example:

* If prediction = **malignant** →

  * Bot explains what malignant cancer means
  * Suggests consulting an **Oncologist**
  * Estimates **stage progression**
  * Provides an **approximate untreated survival estimate** (based on medical literature datasets, but NOT clinical accuracy)

---

## 🚀 Deployment Guide

### 🔹 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
torch
torchvision
pillow
streamlit
requests
```

### 🔹 2. Train the Model

```bash
python model_create.py
```

### 🔹 3. Run the Inference + Chatbot

```bash
streamlit run show.py
```

---

## 🔮 Future Improvements

* ✅ Use **Transfer Learning** (ResNet50, DenseNet121) for better accuracy
* ✅ Add **Multi-class classification** (different cancer types + stages)
* ✅ Improve chatbot reasoning with **medical knowledge graphs**
* ✅ Integrate with **FHIR/HL7 standards** for clinical deployment

---

## ⚠️ Disclaimer

This project is built **for research and educational purposes only**.
The predictions and chatbot outputs **must not be used for real-world medical decisions**.
Always consult a **certified oncologist or medical professional** for actual diagnosis and treatment.

---


