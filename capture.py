import cv2
import os

# Carrega o classificador Haar para deteção de rostos
# Cria o classificador Haar com base num ficheiro .xml que já vem com o OpenCV
#Este classificador já sabe detetar rostos frontais com base em padrões visuais.
#cv2.data.haarcascades devolve o caminho até à pasta dos ficheiros .xml.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


#Inicia a câmara. O 0 indica que estamos a usar a câmara principal do computador.
#Esta função devolve um objeto que permite capturar vídeo frame a frame.
cam = cv2.VideoCapture(1)

#Pede ao utilizador para introduzir um nome ou ID.
#Este valor será usado para nomear a pasta onde as imagens vão ser guardadas, garantindo que cada utilizador tem a sua própria pasta de imagens.
user_id = input("ID do utilizador: ")

#Define o caminho da pasta onde as imagens do rosto vão ser guardadas.
#os.makedirs(..., exist_ok=True) cria a pasta (e as subpastas) se ainda não existirem, sem erro.
output_dir = f"faces/{user_id}"
os.makedirs(output_dir, exist_ok=True)

#Inicializa o contador de imagens capturadas.
count = 0

#Queremos guardar, por exemplo, 30 imagens diferentes do rosto.
#O número de imagens pode ser ajustado (mais imagens = melhor treino).
while count < 30:
    ret, frame = cam.read() #ret é um bool que indica se o frame foi capturado com sucesso; frame é a imagem capturada (colorida, por padrão).
    #Converte a imagem para tons de cinzento (grayscale). 
    # O classificador Haar funciona melhor em imagens sem cor, porque procura apenas contrastes de luz.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    #Aplica o classificador Haar para detetar rostos na imagem em tons de cinzento.
    # detectMultiScale devolve uma lista de coordenadas dos rostos encontrados.
    # scaleFactor=1.3: redimensiona a imagem a cada passo da deteção (ajusta a sensibilidade).
    # minNeighbors=5: quanto maior este valor, menos falsos positivos.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) 
    #Percorre todos os rostos detetados.
    #Cada rosto vem como um retângulo com:
    # x e y = canto superior esquerdo
    # w = largura
    # h = altura
    for (x, y, w, h) in faces:
        #Recorta apenas a zona do rosto detetado da imagem original em tons de cinzento.
        #Isto cria uma imagem mais pequena que contém só o rosto.
        face = gray[y:y+h, x:x+w]
        #Guarda a imagem do rosto num ficheiro .jpg, com o nome 0.jpg, 1.jpg, ..., até 29.jpg.
        cv2.imwrite(f"{output_dir}/{count}.jpg", face)
        count += 1 #Incrementa contador
        #Desenha um retângulo azul sobre o rosto detetado na imagem original (só para visualização).
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #Mostra a imagem com o retângulo em tempo real, numa janela chamada "A capturar rosto".
    cv2.imshow('A capturar rosto (prima "q" para sair)', frame) 

    #Permite que o utilizador saia do ciclo prematuramente, carregando na tecla "q".
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Liberta a câmara para não ficar bloqueada.
#Fecha todas as janelas abertas pelo OpenCV.
cam.release()
cv2.destroyAllWindows()
















