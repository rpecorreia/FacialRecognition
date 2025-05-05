import cv2
import pickle

# Carrega classificador Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carrega modelo treinado
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("modelo.yml")

# Carrega dicionário de IDs → nomes
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Inicia a câmara
cam = cv2.VideoCapture(1)

# Reduz resolução da câmara para acelerar
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Erro ao aceder à câmara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteção com scaleFactor mais alto = mais rápido
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(face)

        if conf < 80:
            name = labels[id_]
            color = (0, 255, 0)
        else:
            name = "Desconhecido"
            color = (0, 0, 255)

        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Reconhecimento Facial (prima 'q' ou ESC para sair)", frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # ESC
        break

cam.release()
cv2.destroyAllWindows()
