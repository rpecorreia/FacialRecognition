import cv2
import pickle
import os
from datetime import datetime

# Carrega classificador Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carrega modelo LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("modelo.yml")

# Carrega o mapeamento ID → nome
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Cria pasta para guardar fotos reconhecidas (se não existir)
os.makedirs("fotos_reconhecidas", exist_ok=True)

# Inicia a câmara
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Guarda os nomes já fotografados recentemente (para evitar duplicados seguidos)
fotos_registadas = {}

while True:
    ret, frame = cam.read()
    if not ret:
        print("Erro ao aceder à câmara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(face)

        if conf < 90:
            name = labels[id_]
            color = (0, 255, 0)

            # Verifica se já foi guardado há pouco tempo
            agora = datetime.now()
            ultima_foto = fotos_registadas.get(name)
            tempo_passado = (agora - ultima_foto).total_seconds() if ultima_foto else 999

            if tempo_passado > 10:  # evita tirar várias fotos em segundos seguidos
                timestamp = agora.strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"fotos_reconhecidas/{name}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[INFO] Foto guardada: {filename}")
                fotos_registadas[name] = agora
        else:
            name = "Desconhecido"
            color = (0, 0, 255)

        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Reconhecimento Facial (ESC ou 'q' para sair)", frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

cam.release()
cv2.destroyAllWindows()
