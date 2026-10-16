import os
import sys
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.detector import DetectorPlacas
from src.pipeline_misto import endireitar_placa, obter_yolo_caracteres, obter_easyocr_reader
from src.desambiguacao import binarizar_caractere

det = DetectorPlacas()
yolo = obter_yolo_caracteres()
reader = obter_easyocr_reader()
pasta = "data/drive_fotos/Fotos placas"

alvos = [
    ('32_NUC6D56_SPD2I07.jpeg', 'NUC6D56'),
    ('36_MPK2E82.jpeg', 'MPK2E82'),
    ('38_NCT5E31_NPL3E65.jpeg', 'NCT5E31'),
    ('40_SPZ6A62.jpeg', 'SPZ6A62')
]

for arq, placa_alvo in alvos:
    img = cv2.imread(os.path.join(pasta, arq))
    placas = det.detectar(img)
    print(f"=== {arq} ({len(placas)} placas detectadas) ===")
    for i, p in enumerate(placas):
        reta, _ = endireitar_placa(p['crop'])
        res_y = yolo.reconhecer(reta)
        print(f"  Placa {i}: YOLO={res_y.get('texto_bruto')} | Corrig={res_y.get('texto_corrigido')}")
