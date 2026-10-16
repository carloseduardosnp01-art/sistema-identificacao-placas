"""
Script preliminar para avaliar o comportamento do fast-plate-ocr
com diferentes modelos e tratamentos de imagem nas placas reais brasileiras.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.detector import DetectorPlacas
from src.utils import extrair_melhor_placa_de_texto, tentar_corrigir_placa, classificar_padrao_placa
from fast_plate_ocr import LicensePlateRecognizer

def testar():
    pasta_fotos = os.path.join(BASE_DIR, "data", "drive_fotos", "Fotos placas")
    arquivos = sorted([f for f in os.listdir(pasta_fotos) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:10]

    detector = DetectorPlacas()

    modelos = [
        "cct-s-v2-global-model",
        "cct-xs-v2-global-model"
    ]

    recognizers = {}
    for m in modelos:
        print(f"Carregando modelo {m}...")
        recognizers[m] = LicensePlateRecognizer(m)

    for arq in arquivos:
        gabarito = arq.split('_')[0]
        caminho = os.path.join(pasta_fotos, arq)
        img = cv2.imread(caminho)
        if img is None:
            continue
        
        placas = detector.detectar(img, confianca_minima=0.20)
        print(f"\n--- Arquivo: {arq} (Gabarito aprox: {gabarito}) ---")
        if not placas:
            print("  Nenhuma placa detectada pelo YOLO detector.")
            continue

        for i, p in enumerate(placas):
            crop = p["crop"]
            h, w = crop.shape[:2]
            # Fast plate OCR espera RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            for m_nome, rec in recognizers.items():
                pred = rec.run_one(crop_rgb, return_confidence=True)
                raw_text = pred.plate
                # Tentar pós-processamento CONTRAN
                corr, padrao = tentar_corrigir_placa(raw_text)
                confs = [round(float(c), 2) for c in (pred.char_probs if pred.char_probs is not None else [])]
                media_conf = float(np.mean(confs)) if len(confs) > 0 else 0.0
                print(f"  [{m_nome}] Raw: '{raw_text}' | Região: {pred.region} | Pós-CONTRAN: '{corr}' ({padrao}) | Conf Média: {media_conf:.2f}")

if __name__ == "__main__":
    testar()
