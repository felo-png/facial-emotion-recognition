import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision import models

# ضبط إعدادات الصفحة
st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🎭")

st.title("🎭 Facial Emotion Recognition")
st.write("قم برفع صورة لوجه للتعرف على الانفعال المعبر عنه.")

# تحميل النموذج مع تحديد الجهاز كـ CPU
@st.cache_resource
def load_model():
    # استخدام resnet18 بدون weights جاہزة
    model = models.resnet18(weights=None)
    
    # تعديل الطبقة الأخيرة لتناسب عدد الكلاسات (7)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    # تحميل الأوزان المسجلة
    device = torch.device('cpu')
    state_dict = torch.load('best_emotion_resnet18.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    with st.spinner("جاري تحميل النموذج..."):
        model = load_model()
    st.success("تم تحميل النموذج بنجاح!")
except Exception as e:
     st.error(f"حدث خطأ أثناء تحميل النموذج: {e}")
# أسماء المشاعر
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# رفع الصورة
uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_column_width=True)
    
    # التحويلات المطلوبة للصورة
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    with st.spinner('جاري التحليل...'):
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        emotion = class_names[preds[0]]
        confidence = probabilities[preds[0]].item() * 100
        
        st.subheader(f"الانفعال المتوقع: **{emotion}**")
        st.write(f"نسبة الثقة: **{confidence:.2f}%**")
