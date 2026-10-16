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

for a in ['02_MWX0J06.jpeg', '03_EIB6H02.jpeg', '04_RNP5H44.jpeg', '06_QCZ9H33.jpeg', '10_FBV7G29.jpeg']:
    img = cv2.imread(os.path.join(pasta, a))
    placas = det.detectar(img)
    if not placas:
        continue
    crop = placas[0]['crop']
    reta, _ = endireitar_placa(crop)
    res = yolo.reconhecer(reta)
    h, w = reta.shape[:2]
    print(f"=== {a} ===")
    for i, c in enumerate(res.get('caixas_caracteres', [])):
        bx1, by1, bx2, by2 = c['bbox']
        char_crop = reta[max(0,by1):min(h,by2), max(0,bx1):min(w,bx2)]
        th = binarizar_caractere(char_crop)
        ch, cw = char_crop.shape[:2]
        ratio = cw / max(1, ch)
        
        # Testes de morfologia
        base_preenchida = float(np.mean(th[48:58, 8:32] == 255)) if th is not None else 0.0
        topo_preenchido = float(np.mean(th[3:14, 6:34] == 255)) if th is not None else 0.0
        meio_preenchido = float(np.mean(th[25:35, 10:30] == 255)) if th is not None else 0.0
        
        print(f"  caixa {i}: char={c['char']} conf={c['conf']:.2f} ratio={ratio:.2f} | base={base_preenchida:.2f} topo={topo_preenchido:.2f} meio={meio_preenchido:.2f}")
