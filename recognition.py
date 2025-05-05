import cv2
import pickle
import os
from datetime import datetime
import csv

# Carrega classificador Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carrega modelo LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("modelo.yml")

# Carrega o mapeamento ID → nome
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Cria pasta para guardar fotos reconhecidas
os.makedirs("fotos_reconhecidas", exist_ok=True)

# Prepara o ficheiro de log CSV (cria se não existir)
log_file = "log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Nome", "Data", "Hora", "Confiança"])

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Erro: não foi possível aceder à câmara.")
    exit()


# Define resolução
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Guarda quem já foi registado recentemente
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

            agora = datetime.now()
            data = agora.strftime("%Y-%m-%d")
            hora = agora.strftime("%H:%M:%S")
            nome_foto = f"{name}_{data}_{hora.replace(':', '-')}.jpg"

            ultima_foto = fotos_registadas.get(name)
            tempo_passado = (agora - ultima_foto).total_seconds() if ultima_foto else 999

            if tempo_passado > 10:
                # Guarda imagem
                caminho_foto = os.path.join("fotos_reconhecidas", nome_foto)
                cv2.imwrite(caminho_foto, frame)
                fotos_registadas[name] = agora

                # Escreve no CSV
                with open(log_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([name, data, hora, round(conf, 2)])

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
