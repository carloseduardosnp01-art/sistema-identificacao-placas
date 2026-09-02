"""
Script para baixar imagens reais de placas veiculares do Open Images Dataset V7 público
"""

import os
import urllib.request
import json

DIRETORIO_EXEMPLOS = os.path.join("data", "exemplos")
os.makedirs(DIRETORIO_EXEMPLOS, exist_ok=True)

# IDs reais de imagens da classe 'Vehicle registration plate' no Open Images Dataset V7
# Disponíveis publicamente no S3 da AWS aberto sem autenticação
OPEN_IMAGES_IDS = [
    "0000a6c0b39e4f50",
    "0000a8927051b755",
    "0000a94e803e05a7",
    "00010ec1f2518e38",
    "0001e405a305542a",
    "00021b369f826270",
    "00027f311c6d3df3",
    "0002f2324dc8122c",
    "00030c6a713838b0",
    "00037a505b38f8cf"
]

print("Baixando 10 imagens de validacao do Open Images Dataset V7...")

headers = {'User-Agent': 'Mozilla/5.0'}

sucessos = 0
for img_id in OPEN_IMAGES_IDS:
    # URL pública do bucket S3 do Google Open Images
    url = f"https://open-images-dataset.s3.amazonaws.com/train/{img_id}.jpg"
    destino = os.path.join(DIRETORIO_EXEMPLOS, f"openimages_{img_id}.jpg")
    
    if os.path.exists(destino):
        print(f"[OK] Ja existe: openimages_{img_id}.jpg")
        sucessos += 1
        continue
        
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp, open(destino, "wb") as f:
            f.write(resp.read())
        print(f"[Sucesso] Baixada: openimages_{img_id}.jpg")
        sucessos += 1
    except Exception as e:
        # Tenta no split de teste se não estiver no train
        try:
            url_test = f"https://open-images-dataset.s3.amazonaws.com/test/{img_id}.jpg"
            req2 = urllib.request.Request(url_test, headers=headers)
            with urllib.request.urlopen(req2, timeout=10) as resp2, open(destino, "wb") as f2:
                f2.write(resp2.read())
            print(f"[Sucesso] Baixada (test): openimages_{img_id}.jpg")
            sucessos += 1
        except Exception as e2:
            print(f"[Erro] Falha em {img_id}: {e2}")

print(f"\nTotal baixado com sucesso: {sucessos} imagens no diretorio '{DIRETORIO_EXEMPLOS}'")
