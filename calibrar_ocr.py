"""
Script de calibração fina para PLW8A46 e ENZ0G18
Testa diferentes parâmetros do EasyOCR (mag_ratio, contrast_ths, sharpen)
"""

import cv2
import numpy as np
import easyocr
from src.ocr import isolar_regiao_caracteres
from src.utils import extrair_melhor_placa_de_texto

reader = easyocr.Reader(['pt', 'en'], gpu=False)

def testar_configuracoes(img_path, nome_esperado):
    img = cv2.imread(img_path)
    from src.detector import DetectorPlacas
    detector = DetectorPlacas()
    placas = detector.detectar(img)
    if not placas:
        print(f"Nenhuma placa detectada em {img_path}")
        return

    crop = placas[0]["crop"]
    miolo = isolar_regiao_caracteres(crop)
    
    h, w = miolo.shape[:2]
    
    print(f"\n==========================================")
    print(f"Testando {img_path} -> Esperado: {nome_esperado}")
    print(f"Dimensões do miolo: {w}x{h}")
    
    # Variações de pré-processamento
    testes_imagem = []
    
    # 1. Redimensionamento 2x com Lanczos + Escala de Cinza + CLAHE suave
    escala = 140.0 / max(1, h)
    w_novo = int(w * escala)
    redim = cv2.resize(miolo, (w_novo, 140), interpolation=cv2.INTER_LANCZOS4)
    cinza = cv2.cvtColor(redim, cv2.COLOR_BGR2GRAY)
    
    clahe_suave = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(cinza)
    testes_imagem.append(("CLAHE suave", clahe_suave))
    
    clahe_forte = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(cinza)
    testes_imagem.append(("CLAHE forte", clahe_forte))
    
    # Sharpen leve
    kernel_sharp = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
    sharp = cv2.filter2D(clahe_suave, -1, kernel_sharp)
    testes_imagem.append(("Sharpen suave", sharp))
    
    # Original colorido redimensionado
    testes_imagem.append(("RGB 2x", redim))

    # Testando combinações de parâmetros do EasyOCR
    for nome_img, im in testes_imagem:
        for mag in [1.0, 1.3, 1.5, 1.8]:
            for th_contrast in [0.1, 0.2]:
                res = reader.readtext(
                    im,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    mag_ratio=mag,
                    contrast_ths=th_contrast,
                    detail=1
                )
                if res:
                    txt = "".join([r[1] for r in res])
                    placa, padrao = extrair_melhor_placa_de_texto(txt)
                    conf = np.mean([r[2] for r in res])
                    status = "✅ ACERTOU!" if placa == nome_esperado else "❌"
                    if status == "✅ ACERTOU!" or placa in [nome_esperado, "PLH8A46", "ENZ0G28"]:
                        print(f"[{status}] Img: {nome_img} | mag={mag} | c_ths={th_contrast} -> Raw: '{txt}' => Placa: '{placa}' (Conf: {conf*100:.1f}%)")

testar_configuracoes("data/test_images/ladomaisinfos.png", "PLW8A46")
testar_configuracoes("data/test_images/mercedesladomercosul.png", "ENZ0G18")
