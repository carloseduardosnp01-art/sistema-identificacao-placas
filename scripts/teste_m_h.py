import os
import sys
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.detector import DetectorPlacas
from src.pipeline_misto import endireitar_placa, obter_yolo_caracteres
from src.desambiguacao import binarizar_caractere

det = DetectorPlacas()
yolo = obter_yolo_caracteres()
pasta = "data/drive_fotos/Fotos placas"

# Teste M vs H na foto 36
img36 = cv2.imread(os.path.join(pasta, '36_MPK2E82.jpeg'))
placa36 = det.detectar(img36)[0]['crop']
reta36, _ = endireitar_placa(placa36)
res36 = yolo.reconhecer(reta36)
caixas36 = res36.get('caixas_caracteres', [])
if caixas36:
    bx1, by1, bx2, by2 = caixas36[0]['bbox']
    crop_0 = reta36[by1:by2, bx1:bx2]
    th = binarizar_caractere(crop_0)
    centro_inf = float(np.mean(th[30:50, 16:24] == 255)) if th is not None else 0.0
    print("Foto 36 caractere 0 centro_inf:", centro_inf)
