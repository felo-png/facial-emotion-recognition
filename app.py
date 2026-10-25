import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
import os
import urllib.request

# ضبط إعدادات الصفحة
st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🎭")

st.title("🎭 Facial Emotion Recognition")
st.write("قم برفع صورة لوجه للتعرف على الانفعال المعبر عنه.")

# تحميل النموذج من رابط مباشر ثابت
@st.cache_resource
def load_model():
    model_path = 'best_emotion_resnet18.pth'
    
    # لو الملف قديم أو مساحته صغيرة (تالف)، هنحذفه وننزله صح
    if os.path.exists(model_path) and os.path.getsize(model_path) < 10000000:
        os.remove(model_path)
        
    if not os.path.exists(model_path):
        # رابط تنزيل مباشر ومستقر جداً
        url = 'https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/model.safetensors' # سنستخدم التنزيل المباشر المضمون
        
        # باستخدام urllib للتحميل المباشر
        req = urllib.request.Request(
            'https://github.com/m-aly/facial-emotion-recognition/releases/download/v1.0/best_emotion_resnet18.pth', 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    device = torch.device('cpu')
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model
