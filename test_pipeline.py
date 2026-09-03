"""
Script de Diagnóstico e Calibração do OCR para as 6 imagens de teste do usuário.
"""

import os
import glob
import cv2
from src.detector import DetectorPlacas
from src.ocr import extrair_texto_placa, obter_leitor_ocr

def testar_pipeline_completo():
    detector = DetectorPlacas()
    reader = obter_leitor_ocr()
    
    pasta_testes = os.path.join("data", "test_images")
    imagens = glob.glob(os.path.join(pasta_testes, "*.*"))
    
    print(f"Encontradas {len(imagens)} imagens de teste.")
    print("=" * 60)
    
    for caminho_img in imagens:
        nome_arquivo = os.path.basename(caminho_img)
        img = cv2.imread(caminho_img)
        if img is None:
            continue
            
        print(f"\n📸 Imagem: {nome_arquivo}")
        placas = detector.detectar(img, confianca_minima=0.25)
        print(f"Placas detectadas pelo YOLO: {len(placas)}")
        
        for i, p in enumerate(placas):
            crop = p["crop"]
            conf_yolo = p["confianca"]
            
            # Leitura com o pipeline otimizado usando a placa completa
            res = extrair_texto_placa(crop)
            
            print(f"  [Placa #{i+1}] (Conf. YOLO: {conf_yolo * 100:.1f}%)")
            print(f"    - Texto Bruto do Pipeline: '{res['texto_bruto']}'")
            print(f"    - Placa Identificada: '{res['texto_corrigido']}' ({res['padrao']})")
            print(f"    - Confiança OCR: {res['confianca'] * 100:.1f}%")

print("\n" + "=" * 60)
