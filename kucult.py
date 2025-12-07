import tensorflow as tf

# Mevcut büyük modeli yükle
print("Büyük model yükleniyor...")
model = tf.keras.models.load_model('topmodel.keras')

# Optimizer olmadan tekrar kaydet (Dosya boyutu çok düşecek)
print("Küçültülmüş model kaydediliyor...")
model.save('topmodel_light.keras', include_optimizer=False)

print("Tamamlandı! 'topmodel_light.keras' dosyasını GitHub'a yükleyebilirsin.")