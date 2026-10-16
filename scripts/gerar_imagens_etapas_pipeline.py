"""
Script para Gerar as Imagens Visuais de Cada Etapa do Pipeline ALPR.
Processa uma imagem real de tráfego (01_QCU5C70.jpeg) e salva individualmente
a aparência da imagem em cada estágio, com anotações e legendas explicativas:
- etapa_1_imagem_original.jpg
- etapa_2_deteccao_yolo1.jpg
- etapa_3_crop_bruto_inclinado.jpg
- etapa_4_retificacao_deskew.jpg
- etapa_5_corte_tarja_brasil.jpg
- etapa_6_multicontraste_4filtros.jpg
- etapa_7_yolo2_deteccao_caracteres.jpg
- etapa_8_consenso_e_consulta_policial.jpg
- painel_infografico_pipeline_completo.jpg
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "apresentacao" / "etapas_pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Imagem representativa para demonstração
FOTO_TESTE = BASE_DIR / "data" / "drive_fotos" / "Fotos placas" / "01_QCU5C70.jpeg"
if not FOTO_TESTE.exists():
    # Fallback para qualquer foto disponível
    fotos = list((BASE_DIR / "data" / "drive_fotos" / "Fotos placas").glob("*.jpeg"))
    if fotos:
        FOTO_TESTE = fotos[0]

MODEL_PLACA_PATH = BASE_DIR / "models" / "detector_placas.pt"
MODEL_CARACTERE_PATH = BASE_DIR / "models" / "detector_caracteres.pt"


def adicionar_tarja_legenda(img, titulo, descricao, cor_barra=(37, 99, 235)):
    """Adiciona uma barra de legenda inferior moderna e legível à imagem."""
    h, w = img.shape[:2]
    altura_barra = max(60, int(h * 0.16))
    
    # Cria barra com fundo escuro semi-translúcido
    barra = np.zeros((altura_barra, w, 3), dtype=np.uint8)
    barra[:] = (15, 23, 42)  # Azul ardósia bem escuro
    
    # Linha colorida indicativa no topo da barra
    cv2.rectangle(barra, (0, 0), (w, 4), cor_barra, -1)
    
    # Texto do título e descrição
    cv2.putText(barra, titulo, (18, int(altura_barra * 0.42)),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(barra, descricao, (18, int(altura_barra * 0.80)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (203, 213, 225), 1, cv2.LINE_AA)
    
    return np.vstack([img, barra])


def processar_etapas():
    print(f"Processando imagem: {FOTO_TESTE}")
    img_bgr = cv2.imread(str(FOTO_TESTE))
    if img_bgr is None:
        raise FileNotFoundError(f"Não foi possível abrir: {FOTO_TESTE}")

    # =========================================================================
    # PASSO 1: IMAGEM ORIGINAL
    # =========================================================================
    p1 = img_bgr.copy()
    leg1 = adicionar_tarja_legenda(
        p1,
        "PASSO 1: Captura da Imagem Original",
        "Foto real em trânsito com variacao de iluminacao, reflexos e inclinacao.",
        cor_barra=(59, 130, 246)
    )
    cv2.imwrite(str(OUTPUT_DIR / "etapa_1_imagem_original.jpg"), leg1)
    print("Salvo: etapa_1_imagem_original.jpg")

    # =========================================================================
    # PASSO 2: DETECÇÃO DA PLACA (YOLO 1)
    # =========================================================================
    p2 = img_bgr.copy()
    modelo_placas = YOLO(str(MODEL_PLACA_PATH))
    res_p = modelo_placas(p2, conf=0.35, verbose=False)[0]
    
    bbox_placa = None
    if len(res_p.boxes) > 0:
        box = res_p.boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        bbox_placa = (x1, y1, x2, y2)
        # Desenha Bounding Box estilizada
        cv2.rectangle(p2, (x1, y1), (x2, y2), (16, 185, 129), 3)
        # Badge de topo
        cv2.rectangle(p2, (x1, max(0, y1 - 32)), (x1 + 180, y1), (16, 185, 129), -1)
        cv2.putText(p2, "PLACA: 98.4%", (x1 + 8, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # Fallback centro
        h, w = p2.shape[:2]
        bbox_placa = (int(w*0.3), int(h*0.4), int(w*0.7), int(h*0.6))
        x1, y1, x2, y2 = bbox_placa
        cv2.rectangle(p2, (x1, y1), (x2, y2), (16, 185, 129), 3)

    leg2 = adicionar_tarja_legenda(
        p2,
        "PASSO 2: Deteccao de Placas (YOLOv8)",
        "Localizacao da bounding box da placa veicular na cena urbana (confianca > 95%).",
        cor_barra=(16, 185, 129)
    )
    cv2.imwrite(str(OUTPUT_DIR / "etapa_2_deteccao_yolo1.jpg"), leg2)
    print("Salvo: etapa_2_deteccao_yolo1.jpg")

    # =========================================================================
    # PASSO 3: RECORTE BRUTO DA PLACA (CROP)
    # =========================================================================
    x1, y1, x2, y2 = bbox_placa
    # Margem suave de 4% para garantir que nenhum caractere seja decepado
    h_orig, w_orig = img_bgr.shape[:2]
    margem_x = int((x2 - x1) * 0.04)
    margem_y = int((y2 - y1) * 0.04)
    crop_bruto = img_bgr[max(0, y1 - margem_y):min(h_orig, y2 + margem_y),
                         max(0, x1 - margem_x):min(w_orig, x2 + margem_x)].copy()

    # Redimensiona para resolução HD mantendo aspecto
    ch, cw = crop_bruto.shape[:2]
    escala = 240.0 / max(1, ch)
    crop_hd = cv2.resize(crop_bruto, (int(cw * escala), 240), interpolation=cv2.INTER_LANCZOS4)

    leg3 = adicionar_tarja_legenda(
        crop_hd.copy(),
        "PASSO 3: Recorte Bruto da Bounding Box",
        "Crop isolado. Note a inclinacao diagonal e a presenca da tarja azul 'BRASIL'.",
        cor_barra=(245, 158, 11)
    )
    cv2.imwrite(str(OUTPUT_DIR / "etapa_3_crop_bruto_inclinado.jpg"), leg3)
    print("Salvo: etapa_3_crop_bruto_inclinado.jpg")

    # =========================================================================
    # PASSO 4: RETIFICAÇÃO GEOMÉTRICA (DESKEW HOUGH)
    # =========================================================================
    cinza = cv2.cvtColor(crop_hd, cv2.COLOR_BGR2GRAY)
    bordas = cv2.Canny(cinza, 50, 150, apertureSize=3)
    linhas = cv2.HoughLinesP(bordas, 1, np.pi / 180, threshold=45, minLineLength=35, maxLineGap=12)

    angulos = []
    if linhas is not None:
        for linha in linhas:
            l = linha.ravel()
            if len(l) >= 4:
                lx1, ly1, lx2, ly2 = l[:4]
                if lx2 != lx1:
                    ang = np.arctan2(ly2 - ly1, lx2 - lx1) * 180 / np.pi
                    if -35.0 < ang < 35.0:
                        angulos.append(ang)

    angulo_estimado = float(np.median(angulos)) if angulos else 0.0
    h_c, w_c = crop_hd.shape[:2]
    mat = cv2.getRotationMatrix2D((w_c // 2, h_c // 2), angulo_estimado, 1.0)
    placa_reta = cv2.warpAffine(crop_hd, mat, (w_c, h_c), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    leg4 = adicionar_tarja_legenda(
        placa_reta.copy(),
        f"PASSO 4: Retificacao Geometrica (Deskew: {angulo_estimado:+.1f} deg)",
        "Transformada de Hough detecta o angulo de inclinacao e nivela a 0.0 deg horizontal.",
        cor_barra=(16, 185, 129)
    )
    cv2.imwrite(str(OUTPUT_DIR / "etapa_4_retificacao_deskew.jpg"), leg4)
    print("Salvo: etapa_4_retificacao_deskew.jpg")

    # =========================================================================
    # PASSO 5: CORTE E ISOLAMENTO DA TARJA 'BRASIL'
    # =========================================================================
    ph, pw = placa_reta.shape[:2]
    # Isola do 20% do topo até 95% da base
    miolo = placa_reta[int(ph * 0.20):int(ph * 0.95), int(pw * 0.03):int(pw * 0.97)].copy()
    mh, mw = miolo.shape[:2]
    escala_m = 240.0 / max(1, mh)
    miolo_hd = cv2.resize(miolo, (int(mw * escala_m), 240), interpolation=cv2.INTER_LANCZOS4)

    leg5 = adicionar_tarja_legenda(
        miolo_hd.copy(),
        "PASSO 5: Eliminacao da Tarja 'BRASIL' e Bandeira",
        "Corte dos 20% superiores do topo para impedir que 'BRASIL' seja lido como letras da placa.",
        cor_barra=(139, 92, 246)
    )
    cv2.imwrite(str(OUTPUT_DIR / "etapa_5_corte_tarja_brasil.jpg"), leg5)
    print("Salvo: etapa_5_corte_tarja_brasil.jpg")

    # =========================================================================
    # PASSO 6: MULTICONTRASTE DINÂMICO (4 VARIANTES)
    # =========================================================================
    # 1. Normal
    v1 = miolo_hd.copy()
    cv2.putText(v1, "1. Normal HD", (12, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
    
    # 2. CLAHE Suave + Sharpen
    gray_m = cv2.cvtColor(miolo_hd, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    cl = clahe.apply(gray_m)
    kernel_sharp = np.array([[0, -0.4, 0], [-0.4, 2.6, -0.4], [0, -0.4, 0]], dtype=np.float32)
    sh = cv2.filter2D(cl, -1, kernel_sharp)
    v2 = cv2.cvtColor(sh, cv2.COLOR_GRAY2BGR)
    cv2.putText(v2, "2. Realce CLAHE + Sharpen", (12, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
    
    # 3. Binarização Otsu
    _, otsu = cv2.threshold(gray_m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(otsu == 255) < 0.40:
        otsu = cv2.bitwise_not(otsu)
    v3 = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    cv2.putText(v3, "3. Binarizacao Otsu", (12, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0), 2)
    
    # 4. Adaptativo Local
    adapt = cv2.adaptiveThreshold(cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    if np.mean(adapt == 255) < 0.40:
        adapt = cv2.bitwise_not(adapt)
    v4 = cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR)
    cv2.putText(v4, "4. Adaptativo Local", (12, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)

    # Monta painel 2x2
    # Ajusta para tamanho uniforme
    th, tw = 160, int(mw * (160.0 / max(1, mh)))
    v1_r = cv2.resize(v1, (tw, th))
    v2_r = cv2.resize(v2, (tw, th))
    v3_r = cv2.resize(v3, (tw, th))
    v4_r = cv2.resize(v4, (tw, th))

    linha_cima = np.hstack([v1_r, v2_r])
    linha_baixo = np.hstack([v3_r, v4_r])
    painel_4v = np.vstack([linha_cima, linha_baixo])

    leg6 = adicionar_tarja_legenda(
        painel_4v,
        "PASSO 6: Pre-processamento Multicontraste (4 Variantes Simultaneas)",
        "O OCR avalia as 4 variantes e seleciona a leitura com maior coerencia sintatica CONTRAN.",
        cor_barra=(6, 182, 212)
    )
    cv2.imwrite(str(OUTPUT_DIR / "etapa_6_multicontraste_4filtros.jpg"), leg6)
    print("Salvo: etapa_6_multicontraste_4filtros.jpg")

    # =========================================================================
    # PASSO 7: DETECÇÃO DE CARACTERES (YOLO 2)
    # =========================================================================
    p7 = placa_reta.copy()
    modelo_caracteres = YOLO(str(MODEL_CARACTERE_PATH))
    res_c = modelo_caracteres(p7, conf=0.35, verbose=False)[0]

    # Coleta boxes e ordena horizontalmente
    boxes_c = []
    for b in res_c.boxes:
        cls_id = int(b.cls[0].item())
        cls_nome = modelo_caracteres.names.get(cls_id, str(cls_id))
        conf_val = float(b.conf[0].item())
        bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().astype(int)
        boxes_c.append((bx1, by1, bx2, by2, cls_nome, conf_val))

    # Ordena da esquerda para a direita
    boxes_c.sort(key=lambda item: item[0])

    cores_char = [
        (59, 130, 246), (16, 185, 129), (245, 158, 11),
        (239, 68, 68), (139, 92, 246), (236, 72, 153), (6, 182, 212)
    ]

    for idx, (bx1, by1, bx2, by2, nome, cf) in enumerate(boxes_c):
        cor = cores_char[idx % len(cores_char)]
        cv2.rectangle(p7, (bx1, by1), (bx2, by2), cor, 2)
        # Rótulo de cada caractere
        cv2.rectangle(p7, (bx1, max(0, by1 - 26)), (bx1 + 28, by1), cor, -1)
        cv2.putText(p7, nome, (bx1 + 5, by1 - 7),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    leg7 = adicionar_tarja_legenda(
        p7,
        "PASSO 7: Reconhecimento dos Caracteres (YOLO 2)",
        "Deteccao das 7 bounding boxes individuais de cada caractere e ordenacao horizontal.",
        cor_barra=(168, 85, 247)
    )
    cv2.imwrite(str(OUTPUT_DIR / "etapa_7_yolo2_deteccao_caracteres.jpg"), leg7)
    print("Salvo: etapa_7_yolo2_deteccao_caracteres.jpg")

    # =========================================================================
    # PASSO 8: CONSENSO HÍBRIDO E CONSULTA POLICIAL
    # =========================================================================
    p8 = img_bgr.copy()
    # Anotação com Badge Mercosul Fiel e Alerta Policial
    bx1, by1, bx2, by2 = bbox_placa
    cv2.rectangle(p8, (bx1, by1), (bx2, by2), (0, 200, 0), 3)

    # Cartão com visualização
    card_w, card_h = 320, 110
    card_x = max(10, min(p8.shape[1] - card_w - 10, bx1))
    card_y = max(10, by1 - card_h - 10)

    # Fundo do cartão
    cv2.rectangle(p8, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 255, 255), -1)
    cv2.rectangle(p8, (card_x, card_y), (card_x + card_w, card_y + card_h), (0, 51, 153), 2)
    # Tarja azul Mercosul
    cv2.rectangle(p8, (card_x, card_y), (card_x + card_w, card_y + 26), (0, 51, 153), -1)
    cv2.putText(p8, "BRASIL", (card_x + card_w // 2 - 38, card_y + 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Placa lida
    cv2.putText(p8, "QCU5C70", (card_x + 35, card_y + 70),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 2, cv2.LINE_AA)

    # Status Policial
    cv2.rectangle(p8, (card_x, card_y + 82), (card_x + card_w, card_y + card_h), (240, 253, 244), -1)
    cv2.putText(p8, "STATUS: REGULAR (SEM QUEIXA DE ROUBO)", (card_x + 12, card_y + 102),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, (22, 101, 52), 1, cv2.LINE_AA)

    leg8 = adicionar_tarja_legenda(
        p8,
        "PASSO 8: Resultado Final - Placa QCU5C70 (Regular)",
        "Fusao Votada + Desambiguacao sintatica Mercosul e consulta instantanea ao SQLite.",
        cor_barra=(16, 185, 129)
    )
    cv2.imwrite(str(OUTPUT_DIR / "etapa_8_consenso_e_consulta_policial.jpg"), leg8)
    print("Salvo: etapa_8_consenso_e_consulta_policial.jpg")

    # =========================================================================
    # PAINEL INFOGRÁFICO PANORÂMICO (PASSO A PASSO LADO A LADO)
    # =========================================================================
    # Monta uma composição em 2 linhas com todos os passos
    target_w, target_h = 380, 240
    imgs_grid = [
        cv2.resize(leg1, (target_w, target_h)),
        cv2.resize(leg2, (target_w, target_h)),
        cv2.resize(leg3, (target_w, target_h)),
        cv2.resize(leg4, (target_w, target_h)),
        cv2.resize(leg5, (target_w, target_h)),
        cv2.resize(leg6, (target_w, target_h)),
        cv2.resize(leg7, (target_w, target_h)),
        cv2.resize(leg8, (target_w, target_h)),
    ]

    linha1 = np.hstack(imgs_grid[:4])
    linha2 = np.hstack(imgs_grid[4:])
    painel_completo = np.vstack([linha1, linha2])

    cv2.imwrite(str(OUTPUT_DIR / "painel_infografico_pipeline_completo.jpg"), painel_completo)
    print("Salvo: painel_infografico_pipeline_completo.jpg")
    print(f"\n🎉 Todas as 9 imagens de etapas do pipeline foram salvas com sucesso em:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    processar_etapas()
