"""
Teste de Segmentacao e Classificacao Geometrica de Caracteres
"""

import cv2
import numpy as np
import easyocr
from src.ocr import isolar_regiao_caracteres, endireitar_placa

def analisar_placa_segmentada(caminho_img):
    img = cv2.imread(caminho_img)
    from src.detector import DetectorPlacas
    p = DetectorPlacas().detectar(img)[0]["crop"]
    
    placa_reta = endireitar_placa(p)
    miolo = isolar_regiao_caracteres(placa_reta)
    
    h, w = miolo.shape[:2]
    escala = 140.0 / h
    redim = cv2.resize(miolo, (int(w * escala), 140), interpolation=cv2.INTER_LANCZOS4)
    cinza = cv2.cvtColor(redim, cv2.COLOR_BGR2GRAY)
    
    # Binarizacao Otsu invertida (letras brancas em fundo preto para achar contornos)
    _, bin_inv = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontra contornos dos caracteres
    contornos, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in contornos:
        x, y, bw, bh = cv2.boundingRect(c)
        # Filtra contornos com tamanho compativel com caractere de placa
        if bh > 40 and bw > 10 and bw < 100:
            boxes.append((x, y, bw, bh))
            
    # Ordena da esquerda para a direita
    boxes = sorted(boxes, key=lambda b: b[0])
    print(f"\n==========================================")
    print(f"Imagem: {caminho_img}")
    print(f"Total de caracteres segmentados: {len(boxes)}")
    
    reader = easyocr.Reader(['pt', 'en'], gpu=False)
    
    # Leitura individual de cada caractere segmentado com margem
    caracteres_lidos = []
    for i, (x, y, bw, bh) in enumerate(boxes):
        # Margem de seguranca ao redor do caractere
        pad = 8
        cy1 = max(0, y - pad)
        cy2 = min(140, y + bh + pad)
        cx1 = max(0, x - pad)
        cx2 = min(redim.shape[1], x + bw + pad)
        
        crop_char = redim[cy1:cy2, cx1:cx2]
        res_char = reader.readtext(crop_char, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", detail=1)
        txt_char = res_char[0][1] if res_char else "?"
        
        aspect_ratio = bw / float(bh)
        print(f"  Char #{i+1} [x={x}, w={bw}, h={bh}, ratio={aspect_ratio:.2f}]: '{txt_char}'")
        caracteres_lidos.append(txt_char)
        
    print(f"Resultado final montado: {''.join(caracteres_lidos)}")

analisar_placa_segmentada("data/test_images/mercedesladomercosul.png")
analisar_placa_segmentada("data/test_images/ladomaisinfos.png")
