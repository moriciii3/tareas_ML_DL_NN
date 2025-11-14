# webcam_detector.py
import torch
from ultralytics import YOLO

# Podemos cambiar el modelo a uno mejor entrenado
MODEL_PATH = "runs/detect/train3/weights/best.pt" 

# Determinar el dispositivo de inferencia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Cargando modelo desde: {MODEL_PATH}")
print(f"Dispositivo de inferencia: {DEVICE}")

# --- CARGAR MODELO ---
try:
    # Cargamos el modelo ajustado
    model = YOLO(MODEL_PATH)

    # --- INFERENCIA EN WEBCAM ---
    print("\nIniciando detección en tiempo real (Webcam 0)...")
    print("Presiona la tecla 'q' para salir de la ventana de detección.")
    
    # El método predict() se encarga de abrir la cámara, procesar los frames 
    # y dibujar las detecciones si 'show=True'.
    model.predict(
        source="0",       # Usar la primera webcam (0)
        show=True,        # Mostrar los resultados en una ventana
        conf=0.25,        # Umbral mínimo de confianza para mostrar una caja
        device=DEVICE     # Usar GPU si está disponible
    )

except FileNotFoundError:
    print(f"\nERROR: No se encontró el modelo en la ruta especificada: {MODEL_PATH}")
    print("Por favor, verifica la ruta a tu archivo 'best.pt'.")
except Exception as e:
    print(f"\nOcurrió un error al ejecutar la detección: {e}")
    if "camera" in str(e).lower():
        print("Asegúrate de que la cámara web está disponible y no está siendo usada por otro programa.")