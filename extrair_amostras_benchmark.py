"""
Script para recortar os veículos individuais das imagens de benchmark RodoSol e UFPR
e gerar um catálogo de testes prontos em data/exemplos/.
"""

import os
import cv2
import numpy as np

DIR_EXEMPLOS = os.path.join("data", "exemplos")
os.makedirs(DIR_EXEMPLOS, exist_ok=True)

# 1. Processa rodosol_samples.png (grade com 8 veículos em diferentes ângulos e iluminação)
caminho_rodosol = os.path.join(DIR_EXEMPLOS, "rodosol_samples.png")
if os.path.exists(caminho_rodosol):
    img = cv2.imread(caminho_rodosol)
    h, w = img.shape[:2]
    # Grade de 2 linhas x 4 colunas (ou similar)
    linhas = 2
    colunas = 4
    delta_h = h // linhas
    delta_w = w // colunas
    
    idx = 1
    for r in range(linhas):
        for c in range(colunas):
            y1 = r * delta_h
            y2 = (r + 1) * delta_h
            x1 = c * delta_w
            x2 = (c + 1) * delta_w
            carro = img[y1:y2, x1:x2]
            
            nome_salvo = os.path.join(DIR_EXEMPLOS, f"veiculo_rodosol_{idx:02d}.png")
            cv2.imwrite(nome_salvo, carro)
            print(f"[OK] Gerado: {nome_salvo}")
            idx += 1

# 2. Processa ufpr_samples.png (veículos em trânsito real)
caminho_ufpr = os.path.join(DIR_EXEMPLOS, "ufpr_samples.png")
if os.path.exists(caminho_ufpr):
    img_ufpr = cv2.imread(caminho_ufpr)
    h, w = img_ufpr.shape[:2]
    linhas = 3
    colunas = 3
    delta_h = h // linhas
    delta_w = w // colunas
    
    idx = 1
    for r in range(linhas):
        for c in range(colunas):
            y1 = r * delta_h
            y2 = (r + 1) * delta_h
            x1 = c * delta_w
            x2 = (c + 1) * delta_w
            carro = img_ufpr[y1:y2, x1:x2]
            
            nome_salvo = os.path.join(DIR_EXEMPLOS, f"veiculo_transito_ufpr_{idx:02d}.png")
            cv2.imwrite(nome_salvo, carro)
            print(f"[OK] Gerado: {nome_salvo}")
            idx += 1

print("\nCatalogo de imagens de teste criado com sucesso!")
