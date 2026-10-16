import os
import sys
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.detector import DetectorPlacas
from src.pipeline_misto import (
    executar_pipeline_misto,
    obter_yolo_caracteres,
    obter_easyocr_reader,
    endireitar_placa,
    executar_ocr_multicontraste
)

detector = DetectorPlacas()
yolo_caracteres = obter_yolo_caracteres()
pasta = 'data/drive_fotos/Fotos placas'
arquivos = sorted([f for f in os.listdir(pasta) if f.endswith('.jpeg') or f.endswith('.jpg')])

fotos_alvo = ['03_', '05_', '08_', '16_', '17_', '20_', '21_', '23_']

for f in arquivos:
    if not any(f.startswith(pref) for pref in fotos_alvo):
        continue
    print(f"Processando {f}...", flush=True)
    caminho = os.path.join(pasta, f)
    img = cv2.imread(caminho)
    preds = detector.detectar(img)
    if not preds:
        print(f"[{f}] Nenhuma placa detectada pelo YOLO 1", flush=True)
        continue
    for i, p in enumerate(preds):
        crop_detector = p["crop"]
        bx1, by1, bx2, by2 = p["bbox"]
        crop_exato = img[by1:by2, bx1:bx2] if (bx2 > bx1 and by2 > by1) else crop_detector
        
        print("Executando com crop_detector...", flush=True)
        res_crop_det = executar_pipeline_misto(crop_detector)
        print("Executando com crop_exato...", flush=True)
        res_crop_exato = executar_pipeline_misto(crop_exato)
        print("Executando com YOLO 2 puro...", flush=True)
        res_yolo_puro = yolo_caracteres.reconhecer(crop_detector)
        
        print(f"[{f}] (Placa {i+1})", flush=True)
        print(f"   -> YOLO 2 PURO:       {res_yolo_puro.get('texto_corrigido')}", flush=True)
        print(f"   -> SERVER (crop_det): {res_crop_det.get('texto_corrigido')} ({res_crop_det.get('metodo')})", flush=True)
        print(f"   -> BENCH  (crop_ex):  {res_crop_exato.get('texto_corrigido')} ({res_crop_exato.get('metodo')})", flush=True)
