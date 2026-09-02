"""
Teste de Alinhamento e Retificação de Ângulo de Placas (Deskew)
"""

import cv2
import numpy as np
import easyocr
from src.detector import DetectorPlacas
from src.utils import extrair_melhor_placa_de_texto

reader = easyocr.Reader(['pt', 'en'], gpu=False)

def endireitar_placa(img_placa):
    """Detecta o ângulo de rotação da placa e corrige a inclinação."""
    cinza = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY)
    
    # Binariza para achar bordas principais
    bordas = cv2.Canny(cinza, 50, 200, apertureSize=3)
    
    # Linhas de Hough para detectar o ângulo dominante das bordas da placa
    linhas = cv2.HoughLinesP(bordas, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)
    
    angulos = []
    if linhas is not None:
        for linha in linhas:
            x1, y1, x2, y2 = linha[0]
            if x2 != x1:
                angulo = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # Filtra apenas ângulos próximos da horizontal (-40 a +40 graus)
                if -40 < angulo < 40:
                    angulos.append(angulo)
                    
    if angulos:
        angulo_medio = float(np.median(angulos))
    else:
        angulo_medio = 0.0
        
    # Rotaciona a imagem para compensar a inclinação
    h, w = img_placa.shape[:2]
    centro = (w // 2, h // 2)
    matriz_rot = cv2.getRotationMatrix2D(centro, angulo_medio, 1.0)
    placa_reta = cv2.warpAffine(img_placa, matriz_rot, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return placa_reta, angulo_medio

img = cv2.imread("data/test_images/ladomaisinfos.png")
p = DetectorPlacas().detectar(img)[0]["crop"]
placa_reta, ang = endireitar_placa(p)
print(f"Ângulo corrigido: {ang:.2f}°")

# Agora testa OCR na placa endireitada
from src.ocr import isolar_regiao_caracteres
miolo_reto = isolar_regiao_caracteres(placa_reta)

# Aumenta resolução
h, w = miolo_reto.shape[:2]
escala = 140.0 / h
redim = cv2.resize(miolo_reto, (int(w * escala), 140), interpolation=cv2.INTER_LANCZOS4)

# Filtro CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cinza = clahe.apply(cv2.cvtColor(redim, cv2.COLOR_BGR2GRAY))

res = reader.readtext(cinza, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", mag_ratio=1.5, detail=1)
print("Resultado no miolo alinhado:", res)
if res:
    txt = "".join([r[1] for r in res])
    print("Texto extraído:", extrair_melhor_placa_de_texto(txt))
