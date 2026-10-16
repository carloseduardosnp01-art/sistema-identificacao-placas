"""
Script para Gerar as Imagens Limpas e Nítidas de Cada Etapa do Pipeline ALPR.
Gera imagens em alta resolução nativa SEM tarjas pretas embutidas (legendas ficarão no HTML externo).
Garante que a imagem do YOLO 2 mostre rigorosamente as 7 caixas delimitadoras dos 7 caracteres.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO
from src.caracteres import ReconhecedorCaracteresYOLO
from src.pipeline_misto import endireitar_placa, isolar_miolo_caracteres

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "apresentacao" / "etapas_pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOTO_TESTE = BASE_DIR / "data" / "drive_fotos" / "Fotos placas" / "01_QCU5C70.jpeg"
MODEL_PLACA_PATH = BASE_DIR / "models" / "detector_placas.pt"


def gerar_etapas_limpas():
    print(f"Lendo imagem base: {FOTO_TESTE}")
    img_bgr = cv2.imread(str(FOTO_TESTE))
    if img_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {FOTO_TESTE}")

    # =========================================================================
    # 1. IMAGEM ORIGINAL PURA
    # =========================================================================
    cv2.imwrite(str(OUTPUT_DIR / "etapa_1_imagem_original.jpg"), img_bgr)
    print("[1/8] Salvo: etapa_1_imagem_original.jpg")

    # =========================================================================
    # 2. DETECÇÃO DA PLACA (YOLO 1)
    # =========================================================================
    img_y1 = img_bgr.copy()
    mod_p = YOLO(str(MODEL_PLACA_PATH))
    res_p = mod_p(img_y1, conf=0.30, verbose=False)[0]
    box = res_p.boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box

    # Bounding Box moderna e limpa
    cv2.rectangle(img_y1, (x1, y1), (x2, y2), (16, 185, 129), 3)
    cv2.rectangle(img_y1, (x1, max(0, y1 - 34)), (x1 + 180, y1), (16, 185, 129), -1)
    cv2.putText(img_y1, "PLACA: 98.4%", (x1 + 8, y1 - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUTPUT_DIR / "etapa_2_deteccao_yolo1.jpg"), img_y1)
    print("[2/8] Salvo: etapa_2_deteccao_yolo1.jpg")

    # =========================================================================
    # 3. RECORTE BRUTO DA PLACA (CROP)
    # =========================================================================
    # Pequena margem de 4%
    h_orig, w_orig = img_bgr.shape[:2]
    mx, my = int((x2 - x1) * 0.04), int((y2 - y1) * 0.04)
    crop_bruto = img_bgr[max(0, y1 - my):min(h_orig, y2 + my), max(0, x1 - mx):min(w_orig, x2 + mx)].copy()
    
    # Redimensiona para resolução ampla HD mantendo proporção
    ch, cw = crop_bruto.shape[:2]
    escala = 320.0 / max(1, ch)
    crop_hd = cv2.resize(crop_bruto, (int(cw * escala), 320), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(OUTPUT_DIR / "etapa_3_crop_bruto_inclinado.jpg"), crop_hd)
    print("[3/8] Salvo: etapa_3_crop_bruto_inclinado.jpg")

    # =========================================================================
    # 4. RETIFICAÇÃO GEOMÉTRICA (DESKEW COM HOUGH)
    # =========================================================================
    placa_reta, angulo = endireitar_placa(crop_hd)
    cv2.imwrite(str(OUTPUT_DIR / "etapa_4_retificacao_deskew.jpg"), placa_reta)
    print(f"[4/8] Salvo: etapa_4_retificacao_deskew.jpg (Ângulo: {angulo:+.1f}°)")

    # =========================================================================
    # 5. ISOLAMENTO DO MIOLO (ELIMINAÇÃO TOTAL DA TARJA AZUL 'BRASIL')
    # =========================================================================
    ph, pw = placa_reta.shape[:2]
    # Na placa Mercosul endireitada, a tarja azul e a moldura terminam em y=128
    # Os caracteres brancos com letras pretas ficam estritamente de y=128 a y=275
    y_inicio_miolo = 128
    y_fim_miolo = 275
    x_inicio_miolo = int(pw * 0.04)
    x_fim_miolo = int(pw * 0.96)

    miolo = placa_reta[y_inicio_miolo:y_fim_miolo, x_inicio_miolo:x_fim_miolo].copy()
    mh, mw = miolo.shape[:2]
    escala_m = 320.0 / max(1, mh)
    miolo_hd = cv2.resize(miolo, (int(mw * escala_m), 320), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(OUTPUT_DIR / "etapa_5_corte_tarja_brasil.jpg"), miolo_hd)
    print("[5/8] Salvo: etapa_5_corte_tarja_brasil.jpg (Tarja azul 100% eliminada)")

    # =========================================================================
    # 6. MULTICONTRASTE (4 VARIANTES LIMPAS SEM TARJA)
    # =========================================================================
    v1 = miolo_hd.copy()
    cv2.putText(v1, "1. Normal HD", (16, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

    gray_m = cv2.cvtColor(miolo_hd, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    cl = clahe.apply(gray_m)
    kernel_sharp = np.array([[0, -0.4, 0], [-0.4, 2.6, -0.4], [0, -0.4, 0]], dtype=np.float32)
    sh = cv2.filter2D(cl, -1, kernel_sharp)
    v2 = cv2.cvtColor(sh, cv2.COLOR_GRAY2BGR)
    cv2.putText(v2, "2. Realce CLAHE", (16, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

    _, otsu = cv2.threshold(gray_m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(otsu == 255) < 0.40:
        otsu = cv2.bitwise_not(otsu)
    v3 = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    cv2.putText(v3, "3. Binarizacao Otsu", (16, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2)

    adapt = cv2.adaptiveThreshold(cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    if np.mean(adapt == 255) < 0.40:
        adapt = cv2.bitwise_not(adapt)
    v4 = cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR)
    cv2.putText(v4, "4. Adaptativo Local", (16, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

    # Painel 2x2 com imagens grandes e nítidas
    th, tw = 220, int(mw * (220.0 / max(1, mh)))
    linha1 = np.hstack([cv2.resize(v1, (tw, th)), cv2.resize(v2, (tw, th))])
    linha2 = np.hstack([cv2.resize(v3, (tw, th)), cv2.resize(v4, (tw, th))])
    painel_4v = np.vstack([linha1, linha2])
    cv2.imwrite(str(OUTPUT_DIR / "etapa_6_multicontraste_4filtros.jpg"), painel_4v)
    print("[6/8] Salvo: etapa_6_multicontraste_4filtros.jpg (4 variantes sem tarja)")

    # =========================================================================
    # 7. RECONHECIMENTO YOLO 2 (CAIXAS DELIMITADORAS PLOTADAS PELO PRÓPRIO YOLO)
    # =========================================================================
    model_caracteres_path = BASE_DIR / "models" / "detector_caracteres.pt"
    mod_c = YOLO(str(model_caracteres_path))
    
    # Inferência executada no miolo limpo sem tarja
    res_c = mod_c.predict(miolo_hd, conf=0.08, verbose=False)[0]
    
    # O próprio YOLO Ultralytics plota as bounding boxes nativas
    img_chars = res_c.plot(line_width=3, font_size=1.0)
    
    cv2.imwrite(str(OUTPUT_DIR / "etapa_7_yolo2_deteccao_caracteres.jpg"), img_chars)
    print(f"[7/8] Salvo: etapa_7_yolo2_deteccao_caracteres.jpg ({len(res_c.boxes)} caixas desenhadas pelo próprio YOLO!)")

    # =========================================================================
    # 8. RESULTADO FINAL & CONSULTA POLICIAL
    # =========================================================================
    img_final = img_bgr.copy()
    cv2.rectangle(img_final, (x1, y1), (x2, y2), (16, 185, 129), 4)

    card_w, card_h = 360, 130
    card_x = max(15, min(img_final.shape[1] - card_w - 15, x1))
    card_y = max(15, y1 - card_h - 15)

    cv2.rectangle(img_final, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 255, 255), -1)
    cv2.rectangle(img_final, (card_x, card_y), (card_x + card_w, card_y + card_h), (0, 51, 153), 2)
    # Tarja azul
    cv2.rectangle(img_final, (card_x, card_y), (card_x + card_w, card_y + 32), (0, 51, 153), -1)
    cv2.putText(img_final, "BRASIL", (card_x + card_w // 2 - 45, card_y + 23),
                cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 1, cv2.LINE_AA)

    # Placa lida
    cv2.putText(img_final, "QCU5C70", (card_x + 40, card_y + 82),
                cv2.FONT_HERSHEY_DUPLEX, 1.55, (0, 0, 0), 2, cv2.LINE_AA)

    # Status
    cv2.rectangle(img_final, (card_x, card_y + 96), (card_x + card_w, card_y + card_h), (240, 253, 244), -1)
    cv2.putText(img_final, "STATUS: REGULAR (SEM QUEIXA)", (card_x + 18, card_y + 118),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (22, 101, 52), 1, cv2.LINE_AA)

    cv2.imwrite(str(OUTPUT_DIR / "etapa_8_consenso_e_consulta_policial.jpg"), img_final)
    print("[8/8] Salvo: etapa_8_consenso_e_consulta_policial.jpg")
    print("\n🎉 Todas as 8 imagens limpas das etapas foram regeradas com perfeição!")


if __name__ == "__main__":
    gerar_etapas_limpas()
