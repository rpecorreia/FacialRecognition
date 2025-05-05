import cv2                # OpenCV para visão computacional
from cv2 import face      # Módulo específico para reconhecimento facial (LBPH)
import numpy as np        # Para manipular arrays (necessário para treino)
import os                 # Para percorrer pastas e ficheiros
import pickle             # Para guardar o dicionário ID → nome

# Cria o reconhecedor facial usando o algoritmo LBPH (Local Binary Patterns Histograms)
# O algoritmo LBPH é simples, rápido e funciona bem com imagens pequenas.
# Transforma a imagem num padrão de texturas e depois compara esses padrões.
recognizer = face.LBPHFaceRecognizer_create()

faces = []       # Lista de imagens dos rostos (em cinzento)
labels = []      # Lista dos IDs (números inteiros)
label_map = {}   # Mapeamento de ID → nome (ex: {0: "rita", 1: "joao"})

base_path = "faces/"  # Caminho onde estão guardadas as pastas com os rostos
label_id = 0          # Começa no ID 0

# Percorre todas as subpastas dentro da pasta "faces/"
for user in os.listdir(base_path):
    label_map[label_id] = user  # Guarda o nome associado ao ID atual (ex: 0 → rita)
    user_path = os.path.join(base_path, user)  # Caminho da pasta do utilizador

    # Percorre todos os ficheiros de imagem dentro da pasta desse utilizador
    for img_name in os.listdir(user_path):
        img_path = os.path.join(user_path, img_name)  # Caminho completo do ficheiro de imagem

        # Lê a imagem em tons de cinzento
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Guarda a imagem e o ID associado
        faces.append(image)
        labels.append(label_id)

    label_id += 1  # Passa para o ID seguinte para o próximo utilizador
    
# Treina o modelo com as imagens e os IDs associados
#faces: lista de imagens
#labels: lista de números (IDs) que correspondem ao nome do utilizador
#O modelo aprende a detetar padrões faciais associados a cada ID
recognizer.train(faces, np.array(labels))

# Guarda o modelo treinado num ficheiro
#Este ficheiro será usado depois pelo reconhecimento.py para identificar os rostos.
recognizer.save("modelo.yml")

# Guarda o mapeamento de IDs para nomes (para converter o número → nome real)
# Guarda o dicionário {0: "rita", 1: "joao"} num ficheiro chamado labels.pkl.
with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("Modelo treinado e guardado com sucesso.")
