"""
Script para recortar perfeitamente as placas individuais de rodosol_cropped_plates.png
e limpar amostras de baixa qualidade.
"""

import os
import glob
import cv2
import numpy as np

DIR_EXEMPLOS = os.path.join("data", "exemplos")

# 1. Remove os arquivos antigos mal recortados
arquivos_antigos = glob.glob(os.path.join(DIR_EXEMPLOS, "veiculo_*.png"))
for f in arquivos_antigos:
    try:
        os.remove(f)
        print(f"Removido arquivo antigo: {os.path.basename(f)}")
    except Exception:
        pass

# 2. Recorta as placas individuais de rodosol_cropped_plates.png
caminho_mosaico = os.path.join(DIR_EXEMPLOS, "rodosol_cropped_plates.png")
if os.path.exists(caminho_mosaico):
    img = cv2.imread(caminho_mosaico)
    h, w = img.shape[:2]
    
    # rodosol_cropped_plates é composto por 2 linhas de placas organizadas horizontalmente
    linhas = 2
    colunas = 6
    dh = h // linhas
    dw = w // colunas
    
    idx = 1
    for r in range(linhas):
        for c in range(colunas):
            y1 = int(r * dh + dh * 0.05)
            y2 = int((r + 1) * dh - dh * 0.05)
            x1 = int(c * dw + dw * 0.03)
            x2 = int((c + 1) * dw - dw * 0.03)
            
            placa = img[y1:y2, x1:x2]
            if placa.size > 0:
                # Aumenta resolução com interpolação Lanczos para alta nitidez
                placa_hd = cv2.resize(placa, (400, 130), interpolation=cv2.INTER_LANCZOS4)
                nome_salvo = os.path.join(DIR_EXEMPLOS, f"placa_alta_definicao_{idx:02d}.png")
                cv2.imwrite(nome_salvo, placa_hd)
                print(f"[OK] Gerada placa limpa em alta definicao: {os.path.basename(nome_salvo)}")
                idx += 1

print("\nProcessamento concluido!")
