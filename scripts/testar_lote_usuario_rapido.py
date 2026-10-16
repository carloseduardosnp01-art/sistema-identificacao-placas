import os
import sys
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.detector import DetectorPlacas
from src.pipeline_misto import executar_pipeline_misto

detector = DetectorPlacas()
pasta = 'data/drive_fotos/Fotos placas'

fotos = [
    '17_RAK9H13.jpeg',
    '20_FDX0I15.jpeg',
    '21_RAQ8D85.jpeg',
    '23_QBH4I14.jpeg',
    '26_QCX4G24.jpeg',
    '28_SPD2I07.jpeg',
    '30_RNK0J18.jpeg',
    '32_NUC6D56_SPD2I07.jpeg',
    '33_OAU2G95_FXK7J13.jpeg',
    '35_QBS6B50_NCT5E31.jpeg',
    '37_FWE0A48.jpeg',
    '38_NCT5E31_NPL3E65.jpeg'
]

print("=== TESTANDO COM O DETECTOR CORRIGIDO (RECORTE EXATO) ===", flush=True)

for nome in fotos:
    caminho = os.path.join(pasta, nome)
    if not os.path.exists(caminho):
        print(f"Arquivo {nome} nao encontrado!", flush=True)
        continue
    img = cv2.imread(caminho)
    placas = detector.detectar(img)
    print(f"\n[{nome}] ({len(placas)} placa(s) detectada(s)):", flush=True)
    for i, p in enumerate(placas):
        crop = p["crop"]
        res = executar_pipeline_misto(crop)
        print(f"  -> Placa {i+1}: {res.get('texto_corrigido')} | Metodo: {res.get('metodo')} | Conf: {res.get('confianca', 0):.2f}", flush=True)
