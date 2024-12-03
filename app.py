import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
 
# -----------------------------
# Function to Load ResNet50 Model
# -----------------------------
@st.cache_resource
def load_resnet_model():
    """
    Loads the pre-trained ResNet50 model with modified final layer.
    """
    model = models.resnet50(pretrained=True)
    
    # Freeze all layers except the last fully connected layer
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: real and fake
    
    # Load the model state dict
    model.load_state_dict(torch.load('resnet_model.pth', map_location=torch.device('cpu')))
    
    model.eval()
    return model

# -----------------------------
# Function to Load EfficientNet-B3 Model
# -----------------------------
@st.cache_resource
def load_efficientnet_model():
    """
    Loads the pre-trained EfficientNet-B3 model with modified final layer.
    """
    model = EfficientNet.from_pretrained('efficientnet-b3')
    
    # Replace the final fully connected layer
    num_features = model._fc.in_features
    model._fc = nn.Linear(num_features, 2)  # 2 classes: real and fake
    
    # Load the model state dict
    model.load_state_dict(torch.load('efficientnet_model.pth', map_location=torch.device('cpu')))
    
    model.eval()
    return model



# -----------------------------
# Function to Load Hybrid Model
# -----------------------------
@st.cache_resource
def load_hybrid_model():
    """
    Loads the trained hybrid model for inference.
    """
    class SEBlock(nn.Module):
        def __init__(self, in_channels, reduction=16):
            super(SEBlock, self).__init__()
            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(in_channels, in_channels // reduction)
            self.fc2 = nn.Linear(in_channels // reduction, in_channels)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            batch, channels, _, _ = x.size()
            y = self.global_avg_pool(x).view(batch, channels)
            y = nn.ReLU()(self.fc1(y))
            y = self.sigmoid(self.fc2(y)).view(batch, channels, 1, 1)
            return x * y

    class HybridBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4):
            super(HybridBlock, self).__init__()
            mid_channels = in_channels * expand_ratio
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False)
            self.bn2 = nn.BatchNorm2d(mid_channels)
            self.se = SEBlock(mid_channels)
            self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            out = nn.ReLU()(self.bn1(self.conv1(x)))
            out = nn.ReLU()(self.bn2(self.depthwise(out)))
            out = self.se(out)
            out = self.bn3(self.conv2(out))
            out += self.shortcut(x)  # Residual connection
            return nn.ReLU()(out)

    class HybridNet(nn.Module):
        def __init__(self, num_classes=2):
            super(HybridNet, self).__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
            self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=1)
            self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
            self.layer3 = self._make_layer(128, 256, num_blocks=3, stride=2)
            self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, num_classes)

        def _make_layer(self, in_channels, out_channels, num_blocks, stride):
            layers = []
            for i in range(num_blocks):
                layers.append(HybridBlock(in_channels if i == 0 else out_channels, out_channels, stride if i == 0 else 1))
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.stem(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.pool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out

    # Instantiate and load the model
    model = HybridNet(num_classes=2)
    model.load_state_dict(torch.load('hybridNet.pth', map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model


# -----------------------------
# Function to Predict Image
# -----------------------------
def predict_image(image, model):
    """
    Predicts whether an image is real or fake using the given model.
    
    Args:
        image (PIL.Image): The input image.
        model (torch.nn.Module): The pre-trained model.
    
    Returns:
        tuple: (predicted_class, confidence_score)
    """
    # Define the image transformations (ensure consistency with training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model's expected input size
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet mean
            std=[0.229, 0.224, 0.225]   # ImageNet std
        )
    ])
    
    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Move input to the same device as the model
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, 1)
    
    return preds.item(), confidence.item()

# -----------------------------
# Function to Predict Video
# -----------------------------
def predict_video(video_path, model, frame_rate=10):
    """
    Predicts whether a video is real or fake by analyzing frames.
    
    Args:
        video_path (str): Path to the video file.
        model (torch.nn.Module): The pre-trained model.
        frame_rate (int): Process every 'frame_rate' frames.
    
    Returns:
        tuple: (final_prediction, average_confidence)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            # Convert the frame to PIL Image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            
            # Predict on the frame
            pred, confidence = predict_image(pil_image, model)
            frames.append((pred, confidence))
        count += 1

    cap.release()
    # cv2.destroyAllWindows()  # Not needed in headless environments

    if not frames:
        return None, None

    # Aggregate results
    preds, confidences = zip(*frames)
    avg_confidence = np.mean(confidences)
    final_pred = max(set(preds), key=preds.count)  # Majority voting

    return final_pred, avg_confidence

# -----------------------------
# Function to Load Models Based on Selection
# -----------------------------
def load_selected_models(selected_models):
    """
    Loads the selected models.
    
    Args:
        selected_models (list): List of selected model names.
    
    Returns:
        dict: Dictionary of loaded models.
    """
    models_dict = {}
    if 'ResNet50' in selected_models:
        models_dict['ResNet50'] = load_resnet_model()
    if 'EfficientNet-B3' in selected_models:
        models_dict['EfficientNet-B3'] = load_efficientnet_model()
    if 'Hybrid Model' in selected_models:
        models_dict['Hybrid Model'] = load_hybrid_model()
    return models_dict

# -----------------------------
# Function to Display Final Prediction with Dynamic Coloring
# -----------------------------
def display_final_prediction(final_result):
    """
    Displays the final prediction with dynamic coloring based on the result.
    
    Args:
        final_result (dict): Dictionary containing 'Prediction' and 'Confidence'.
    """
    prediction = final_result['Prediction']
    confidence = final_result['Confidence']
    
    if prediction == 'Real':
        color = '#28a745'  # Green
    elif prediction == 'Fake':
        color = '#dc3545'  # Red
    else:
        color = '#6c757d'  # Gray for errors or undefined
    
    st.markdown(f"""
    <div class="final-prediction" style="background-color: {color}; color: white;">
        <h3>**Final Prediction: {prediction}**</h3>
        <p>**Confidence:** {confidence}</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Streamlit App Layout
# -----------------------------
def main():
    # Set page configuration
    st.set_page_config(page_title="Deepfake Detection Portal", layout="wide", page_icon="🕵️‍♂️")
    
    # Custom CSS for better visuals
    st.markdown("""
    <style>
    /* Center the title and content */
    .main .block-container {
        max-width: 1200px;
    }
    /* Style for the prediction table */
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        padding: 12px;
        text-align: center;
        border-bottom: 1px solid #ddd;
    }
    th {
        background-color: #4CAF50;
        color: white;
    }
    /* Style for the final prediction */
    .final-prediction {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App Title
    st.title("🕵️‍♂️ Deepfake Detection Portal")
    st.markdown("""
    Welcome to the **Deepfake Detection Portal**! Upload an image or video to determine whether it's **real (original)** or **fake (AI-generated)**. 
    Select the models you wish to use for detection from the sidebar, upload your media, and view the results with confidence scores.
    """)
    
    # Sidebar for Model Selection
    st.sidebar.header("🔧 Model Selection")
    selected_models = st.sidebar.multiselect(
        "Choose models to use for detection:",
        options=["ResNet50", "EfficientNet-B3", "Hybrid Model"],  # "Hybrid Model" can be added when ready
        default=["ResNet50", "EfficientNet-B3"]
    )
    
    # Load selected models
    models_loaded = load_selected_models(selected_models)
    
    # File Uploader
    uploaded_file = st.file_uploader("📤 Upload an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Display file details in the sidebar
        file_details = {"📁 Filename": uploaded_file.name, "📄 File Type": uploaded_file.type}
        st.sidebar.markdown("**📄 Uploaded File Details:**")
        st.sidebar.table(file_details)
        
        # Initialize columns for media and results
        media_col, results_col = st.columns([2, 1])
        
        with media_col:
            if uploaded_file.type.startswith('image'):
                # Process and display image
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='📸 Uploaded Image', use_column_width=True)
            elif uploaded_file.type.startswith('video'):
                # Read video bytes
                video_bytes = uploaded_file.read()
                
                # Check if video bytes are non-empty
                if len(video_bytes) == 0:
                    st.error("⚠️ Uploaded video is empty or corrupted.")
                else:
                    # Display video with responsive width
                    st.video(video_bytes)
        
        with results_col:
            st.header("🔍 Prediction Results")
            with st.spinner('Analyzing...'):
                results = []
    
                if uploaded_file.type.startswith('image'):
                    for model_name, model in models_loaded.items():
                        try:
                            pred, confidence = predict_image(image, model)
                            label = 'Real' if pred == 0 else 'Fake'
                            results.append({
                                'Model': model_name,
                                'Prediction': label,
                                'Confidence': f"{confidence * 100:.2f}%"
                            })
                        except Exception as e:
                            results.append({
                                'Model': model_name,
                                'Prediction': 'Error',
                                'Confidence': 'N/A'
                            })
                            st.error(f"⚠️ Error with {model_name}: {e}")
                elif uploaded_file.type.startswith('video'):
                    # Save the video to a temporary file for processing
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(video_bytes)
                        temp_video_path = tfile.name
    
                    for model_name, model in models_loaded.items():
                        try:
                            pred, confidence = predict_video(temp_video_path, model, frame_rate=10)
                            if pred is not None:
                                label = 'Real' if pred == 0 else 'Fake'
                                results.append({
                                    'Model': model_name,
                                    'Prediction': label,
                                    'Confidence': f"{confidence * 100:.2f}%"
                                })
                            else:
                                results.append({
                                    'Model': model_name,
                                    'Prediction': 'Error',
                                    'Confidence': 'N/A'
                                })
                        except Exception as e:
                            results.append({
                                'Model': model_name,
                                'Prediction': 'Error',
                                'Confidence': 'N/A'
                            })
                            st.error(f"⚠️ Error with {model_name}: {e}")
    
                    # Clean up temporary file
                    try:
                        os.unlink(temp_video_path)
                    except PermissionError:
                        st.warning("⚠️ Could not delete temporary video file. Please check file permissions.")
    
                # Display individual model results
                if results:
                    df_results = pd.DataFrame(results)
                    st.table(df_results)
    
                    # Determine the final prediction based on highest confidence
                    valid_results = [res for res in results if res['Confidence'] != 'N/A']
                    if valid_results:
                        # Convert confidence to float for comparison
                        for res in valid_results:
                            res['Confidence_Value'] = float(res['Confidence'].strip('%'))
                        
                        # Find the result with highest confidence
                        final_result = max(valid_results, key=lambda x: x['Confidence_Value'])
                        
                        # Display final prediction with dynamic coloring
                        display_final_prediction(final_result)
                else:
                    st.write("No predictions available.")
    
        # Footer with Hyperlinked LinkedIn Profile
        st.markdown("""
        ---
        <div style="text-align: center;">
            Developed by <a href="https://www.linkedin.com/in/aayushakumars/" target="_blank" style="color: #163166; text-decoration: none;">Aayush Kumar</a>, MS student at UIC  | © 2024
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


# 01_04__exit_phone_room__0XUW13RW.mp4
# 01_02__talking_angry_couch__YVGY8LOK.mp4 ---> fake


# 01__talking_angry_couch.mp4 --> real 


# 01_02__hugging_happy__YVGY8LOK.mp4 -> r,r, feak

