import os
import io
import datetime
import numpy as np
import tensorflow as tf
from PIL import Image

# Templating ve Redirect için gerekli kütüphaneler
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates 
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse  # <-- YENİ EKLENDİ
from sqlalchemy import create_engine, Column, Integer, String, DateTime, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- AYARLAR ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./predictions.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    prediction_result = Column(String)
    confidence = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

app = FastAPI()

# Şablon klasörünü tanıtıyoruz
templates = Jinja2Templates(directory="templates")

# Dosya yolları
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Modeli Yükle
print("Model yükleniyor...")
try:
    model = tf.keras.models.load_model('topmodel.keras')
    print("✅ Model başarıyla yüklendi!")
except Exception as e:
    print(f"❌ Hata: Model yüklenemedi! {e}")
    model = None

def process_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Modelin içinde preprocessing olduğu için ekstra işlem yapmıyoruz
    return img_array

# --- SAYFALAR VE ENDPOINTLER ---

# 1. ANA SAYFA (GET) - index.html dosyasını döndürür
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 2. TAHMİN (POST) - Sonucu result.html içinde gösterir
@app.post("/predict/")
async def predict_image(request: Request, file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model yüklenemedi"}

    contents = await file.read()
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(contents)

    processed_img = process_image(contents)
    predictions = model.predict(processed_img)
    print(f"Prediction: {predictions}")

    score = predictions[0][0]
    
    if score > 0.5:
        result_text = "Kanserli Hücre (Malignant)"
        confidence = score * 100
    else:
        result_text = "Kansersiz Hücre (Benign)"
        confidence = (1 - score) * 100

    # DB Kayıt
    db = SessionLocal()
    db_record = PredictionRecord(
        filename=file_location,
        prediction_result=result_text,
        confidence=f"{confidence:.2f}"
    )
    db.add(db_record)
    db.commit()
    db.close()

    # result.html şablonunu verilerle doldurup gönderiyoruz
    return templates.TemplateResponse("result.html", {
        "request": request,
        "filename": file_location,
        "result": result_text,
        "confidence": f"{confidence:.2f}"
    })

# --- YENİ EKLENEN KISIM: HATA ÖNLEYİCİ ---
# Eğer biri yanlışlıkla /predict adresine "girmeye" (GET) çalışırsa,
# onu ana sayfaya yönlendiriyoruz.
@app.get("/predict/")
async def redirect_predict_to_home():
    return RedirectResponse(url="/")

# 3. GEÇMİŞ (GET) - history.html dosyasını döndürür
@app.get("/history")
def show_history(request: Request):
    db = SessionLocal()
    records = db.query(PredictionRecord).order_by(desc(PredictionRecord.created_at)).all()
    db.close()
    
    return templates.TemplateResponse("history.html", {
        "request": request, 
        "records": records
    })