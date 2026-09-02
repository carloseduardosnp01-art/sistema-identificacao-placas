"""
Script de Diagnóstico e Calibração do OCR para as 6 imagens de teste do usuário.
"""

import os
import glob
import cv2
import numpy as np
from src.detector import DetectorPlacas
from src.ocr import extrair_texto_placa, isolar_regiao_caracteres, obter_leitor_ocr
from src.utils import formatar_placa_exibicao, classificar_padrao_placa, extrair_melhor_placa_de_texto

detector = DetectorPlacas()
reader = obter_leitor_ocr()

caminho_imagens = glob.glob("data/test_images/*.png")

print(f"Encontradas {len(caminho_imagens)} imagens de teste.")
print("=" * 60)

for img_path in caminho_imagens:
    nome_arquivo = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro ao carregar {nome_arquivo}")
        continue

    placas = detector.detectar(img, confianca_minima=0.20)
    print(f"\n📸 Imagem: {nome_arquivo}")
    print(f"Placas detectadas pelo YOLO: {len(placas)}")

    for i, p in enumerate(placas):
        crop = p["crop"]
        conf_yolo = p["confianca"]
        resultado_ocr = extrair_texto_placa(crop)

        # Teste direto de OCR no crop inteiro e no crop recortado
        miolo = isolar_regiao_caracteres(crop)
        raw_ocr = reader.readtext(miolo, detail=0)

        print(f"  [Placa #{i+1}] (Conf. YOLO: {conf_yolo*100:.1f}%)")
        print(f"    - Raw OCR no miolo: {raw_ocr}")
        print(f"    - Texto Bruto do Pipeline: '{resultado_ocr['texto_bruto']}'")
        print(f"    - Placa Identificada: '{resultado_ocr['texto_corrigido']}' ({resultado_ocr['padrao']})")
        print(f"    - Confiança OCR: {resultado_ocr['confianca']*100:.1f}%")

print("\n" + "=" * 60)
