# Deepfake Detection Portal

## 📌 Overview
The **Deepfake Detection Portal** is a web application that utilizes deep learning models to analyze images and videos, determining whether they are **real (original)** or **fake (AI-generated)**. This project leverages **ResNet50**, **EfficientNet-B3**, and a **custom Hybrid Model** to perform deepfake detection with high accuracy.

## 🚀 Features
- 🖼️ **Image Analysis**: Upload an image to check if it's real or fake.
- 🎥 **Video Analysis**: Process video files and analyze multiple frames to predict deepfake presence.
- 🔍 **Multiple Model Selection**: Choose between **ResNet50**, **EfficientNet-B3**, and **Hybrid Model** for inference.
- 📊 **Confidence Scores**: Get confidence levels for predictions from different models.
- 🎨 **Dynamic UI**: Displays results in an interactive and visually appealing format.

## 🛠️ Technologies Used
- **Python** 🐍
- **Streamlit** 🎨 (For Web UI)
- **PyTorch** 🔥 (For Deep Learning Models)
- **OpenCV** 📹 (For Video Processing)
- **Pandas** 🏷️ (For Result Tabulation)
- **EfficientNet-PyTorch** 📷 (Pretrained Model)
- **ResNet50** 🖼️ (Pretrained Model)
- **Custom Hybrid CNN Model** ⚙️

## 📂 Project Structure
```
Deepfake-Detection-Portal/
│-- app.py               # Main Streamlit application
│-- resnet_model.pth     # Pre-trained ResNet50 model weights
│-- efficientnet_model.pth # Pre-trained EfficientNet-B3 model weights
│-- hybridNet.pth        # Custom Hybrid Model weights
│-- requirements.txt     # Dependencies for the project
│-- README.md            # Project documentation (this file)
```

## 🔧 Installation
To run the project locally, follow these steps:

### 1️⃣ Clone the Repository
```sh
$ git clone https://github.com/your-repo/deepfake-detection-portal.git
$ cd deepfake-detection-portal
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```sh
$ python -m venv venv
$ source venv/bin/activate   # On macOS/Linux
$ venv\Scripts\activate     # On Windows
```

### 3️⃣ Install Dependencies
```sh
$ pip install -r requirements.txt
```

### 4️⃣ Run the Application
```sh
$ streamlit run app.py
```

The application should open in your web browser at `http://localhost:8501/`.

## 🏆 Models Used
### 1️⃣ **ResNet50**
- Pretrained on **ImageNet**
- Fine-tuned for binary classification (**real vs fake**)

### 2️⃣ **EfficientNet-B3**
- Optimized for high performance with fewer parameters
- Modified fully connected layer for binary classification

### 3️⃣ **Hybrid Model** (Custom CNN)
- Combination of **SE Blocks, Residual Connections, and Depthwise Convolutions**
- Designed for improved deepfake detection accuracy

## 📤 Usage Guide
1. **Upload an Image or Video**: Drag and drop your file or browse to upload.
2. **Select Models**: Choose one or more models for analysis.
3. **Get Predictions**: See real-time results along with confidence scores.
4. **Final Prediction**: The system determines the most confident model's decision.

## 🏗️ Future Improvements
- 🔄 **Real-time Video Processing**
- 📈 **Live Webcam Analysis**
- ⚡ **Faster Inference with GPU Support**
- 📊 **Visualization of Deepfake Artifacts**

## 📌 Developer
👨‍💻 **Aayush Kumar**  
📍 MS Student at UIC  
🌐 [LinkedIn](https://www.linkedin.com/in/aayushakumars/)  
📧 [Email](mailto:aayush@example.com)

## 📜 License
This project is licensed under the **MIT License**.

---
**🚀 Ready to detect deepfakes? Upload your media now!** 🎬
