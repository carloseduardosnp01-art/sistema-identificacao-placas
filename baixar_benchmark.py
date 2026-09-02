"""
Script para baixar o conjunto de validação do UFPR-ALPR e RodoSol-ALPR (Benchmark Nacional de Placas Brasileiras)
"""

import os
import urllib.request
import cv2

DIRETORIO_EXEMPLOS = os.path.join("data", "exemplos")
os.makedirs(DIRETORIO_EXEMPLOS, exist_ok=True)

URLS_BENCHMARK = [
    {
        "nome": "rodosol_samples.png",
        "url": "https://raw.githubusercontent.com/raysonlaroca/rodosol-alpr-dataset/master/media/samples.png"
    },
    {
        "nome": "rodosol_cropped_plates.png",
        "url": "https://raw.githubusercontent.com/raysonlaroca/rodosol-alpr-dataset/master/media/samples-cropped.png"
    },
    {
        "nome": "ufpr_samples.png",
        "url": "https://raw.githubusercontent.com/raysonlaroca/ufpr-alpr-dataset/master/media/samples.png"
    },
    {
        "nome": "ufpr_cropped_plates.png",
        "url": "https://raw.githubusercontent.com/raysonlaroca/ufpr-alpr-dataset/master/media/samples-cropped.png"
    }
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for item in URLS_BENCHMARK:
    destino = os.path.join(DIRETORIO_EXEMPLOS, item["nome"])
    try:
        req = urllib.request.Request(item["url"], headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp, open(destino, "wb") as f:
            f.write(resp.read())
        print(f"[Sucesso] Baixado: {item['nome']}")
    except Exception as e:
        print(f"[Erro] {item['nome']}: {e}")

print("Download dos benchmarks concluido.")
