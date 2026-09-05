"""
Módulo de OCR (Optical Character Recognition) de Alta Precisão para Placas Brasileiras.
Inclui desambiguação geométrica FE-Schrift (1 vs 7, 4 vs 1, I vs T, 9 vs 2), tratamento morfológico e resolução adaptativa.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from .utils import extrair_melhor_placa_de_texto, tentar_corrigir_placa, limpar_texto

_reader_easyocr = None


def obter_leitor_easyocr():
    """Inicializa e retorna o leitor EasyOCR em memória."""
    global _reader_easyocr
    if _reader_easyocr is None:
        import easyocr
        _reader_easyocr = easyocr.Reader(['pt', 'en'], gpu=False)
    return _reader_easyocr


def pre_processar_placa(img_placa: np.ndarray) -> np.ndarray:
    """Retorna a placa original em alta definição e formato RGB correto."""
    if img_placa is None or img_placa.size == 0:
        return img_placa

    if len(img_placa.shape) == 3 and img_placa.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_placa, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_placa.copy()

    h, w = img_rgb.shape[:2]
    escala = 200.0 / max(1, h)
    nova_w = int(w * escala)
    return cv2.resize(img_rgb, (nova_w, 200), interpolation=cv2.INTER_LANCZOS4)


def obter_variacoes_placa(img_placa: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Gera variações controladas de pré-processamento para o pipeline Multi-Pass."""
    if img_placa is None or img_placa.size == 0:
        return []

    hd_rgb = pre_processar_placa(img_placa)
    variacoes = [("rgb_hd", hd_rgb)]

    # 1. Faixa focada nos caracteres (remove topo azul Mercosul e bordas externas)
    h_hd, w_hd = hd_rgb.shape[:2]
    faixa_caracteres = hd_rgb[int(h_hd * 0.25):int(h_hd * 0.95), :]
    if faixa_caracteres.shape[0] > 20:
        variacoes.append(("faixa_caracteres", faixa_caracteres))

    # 2. Alinhamento angular por transformada de Hough
    gray = cv2.cvtColor(hd_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=10)
    if lines is not None:
        angles = []
        for l in lines:
            coords = l.reshape(-1)
            if len(coords) >= 4:
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -25 < angle < 25 and abs(angle) > 1.5:
                    angles.append(angle)
        if len(angles) >= 2:
            median_angle = float(np.median(angles))
            if abs(median_angle) > 1.5:
                center = (w_hd // 2, h_hd // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                hd_rot = cv2.warpAffine(hd_rgb, M, (w_hd, h_hd), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                variacoes.append(("alinhada", hd_rot))

    # 3. Fechamento morfológico para preenchimento de micro-falhas
    kernel_morf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cinza_morf = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_morf)
    variacoes.append(("morfologia", cinza_morf))

    # 4. CLAHE para realce de contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variacoes.append(("clahe", clahe))

    return variacoes


def desambiguar_caractere_fe_schrift(char_pred: str, pos: int, crop_char: np.ndarray) -> str:
    """
    Desambiguação estrutural e morfológica da fonte FE-Schrift brasileira:
    - 1 vs 7: '7' possui barra horizontal contínua no topo; '1' é uma haste vertical contínua.
    - 4 vs 1 / L: '4' tem barra transversal larga no terço médio; '1' é estreito.
    - I vs L vs T: 'I' é uma haste vertical pura; 'L' tem base horizontal; 'T' tem topo largo.
    - 9 vs 2: '9' tem anel superior fechado e concentração de massa no topo; '2' tem base horizontal.
    """
    if not char_pred or crop_char is None or crop_char.size == 0:
        return char_pred

    c = char_pred.upper()
    h, w = crop_char.shape[:2]
    if h < 10 or w < 3:
        return c

    try:
        gray = cv2.cvtColor(crop_char, cv2.COLOR_RGB2GRAY) if len(crop_char.shape) == 3 else crop_char
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        nz = cv2.findNonZero(th)
        if nz is None:
            return c

        x_ink, y_ink, w_ink, h_ink = cv2.boundingRect(nz)
        if h_ink < 8 or w_ink < 3:
            return c

        char_th = th[y_ink:y_ink + h_ink, x_ink:x_ink + w_ink]
        h_c, w_c = char_th.shape[:2]
        ratio_wh = w_c / float(max(1, h_c))

        metade_y = h_c // 2
        topo = char_th[:metade_y, :]
        base = char_th[metade_y:, :]
        peso_topo = float(np.sum(topo > 0))
        peso_base = float(max(1, np.sum(base > 0)))
        ratio_topo_base = peso_topo / peso_base

        # 1. Desambiguação 9 vs 2
        if c in ['2', '9'] or (pos in [3, 5, 6] and c in ['2', '9']):
            if ratio_topo_base > 1.55:
                return '9'
            elif ratio_topo_base < 1.30:
                return '2'

        # 2. Desambiguação 1 vs 7
        if c in ['1', '7'] or (pos in [3, 5, 6] and c in ['1', '7']):
            if ratio_topo_base > 1.60:
                return '7'
            elif ratio_topo_base < 1.40:
                return '1'

        # 3. Desambiguação 4 vs 1 / L
        if pos in [3, 5, 6] and c in ['1', '4', 'L']:
            if ratio_wh > 0.35:
                return '4'
            else:
                return '1'

        # 4. Desambiguação I vs L vs T
        if pos in [0, 1, 2, 4] and c in ['I', 'L', 'T', '1', '7']:
            base_faixa = char_th[int(h_c * 0.8):, :]
            topo_faixa = char_th[:int(h_c * 0.2), :]
            largura_base = np.sum(np.any(base_faixa > 0, axis=0))
            largura_topo = np.sum(np.any(topo_faixa > 0, axis=0))

            if largura_topo > w_c * 0.75 and ratio_wh > 0.45:
                return 'T'
            elif largura_base > w_c * 0.60 and ratio_wh > 0.40:
                return 'L'
            elif ratio_wh < 0.38:
                return 'I'

    except Exception:
        pass

    return c


def processar_bloco_ocr(txt: str, bx1: int, by1: int, bx2: int, by2: int, img_hd: np.ndarray) -> str:
    """Descompacta e desambigua individualmente cada caractere dentro de uma bounding box."""
    txt_limpo = limpar_texto(txt)
    if not txt_limpo:
        return ""

    if len(txt_limpo) == 1:
        crop_char = img_hd[by1:by2, bx1:bx2]
        return desambiguar_caractere_fe_schrift(txt_limpo, 0, crop_char)

    chars = []
    w_bloco = bx2 - bx1
    w_char = max(1, w_bloco // len(txt_limpo))

    for idx_c, char_c in enumerate(txt_limpo):
        cx1 = bx1 + idx_c * w_char
        cx2 = min(bx2, cx1 + w_char)
        crop_char = img_hd[by1:by2, cx1:cx2]
        c_corrigido = desambiguar_caractere_fe_schrift(char_c, idx_c, crop_char)
        chars.append(c_corrigido)

    return "".join(chars)


def extrair_texto_placa(img_placa: np.ndarray) -> Dict[str, Any]:
    """
    Pipeline Multi-Pass de Alta Precisão com EasyOCR e Desambiguação Geométrica FE-Schrift.
    Executa múltiplos passes (RGB HD, Alinhamento de Rotação, Morfologia e CLAHE)
    e seleciona a melhor leitura conforme o formato CONTRAN.
    """
    if img_placa is None or img_placa.size == 0:
        return {
            "texto_bruto": "",
            "texto_corrigido": "",
            "padrao": None,
            "confianca": 0.0,
            "img_processada": None
        }

    variacoes = obter_variacoes_placa(img_placa)
    if not variacoes:
        return {
            "texto_bruto": "",
            "texto_corrigido": "",
            "padrao": None,
            "confianca": 0.0,
            "img_processada": None
        }

    img_hd = variacoes[0][1]
    h_hd, w_hd = img_hd.shape[:2]
    reader = obter_leitor_easyocr()

    melhor_resultado = {
        "texto_bruto": "",
        "texto_corrigido": "",
        "padrao": None,
        "confianca": 0.0,
        "img_processada": img_hd
    }
    melhor_score = -1.0

    for nome_var, img_var in variacoes:
        try:
            resultados = reader.readtext(
                img_var,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                detail=1,
                paragraph=False,
                contrast_ths=0.05,
                adjust_contrast=0.5
            )
        except Exception:
            continue

        if not resultados:
            continue

        # Ordena caixas da esquerda para a direita pela coordenada X
        resultados_ordenados = sorted(resultados, key=lambda r: min(pt[0] for pt in r[0]))
        textos_proc = []
        confs = []

        h_v, w_v = img_var.shape[:2]
        for bbox, txt, prob in resultados_ordenados:
            txt_l = limpar_texto(txt)
            if not txt_l or prob <= 0.05:
                continue

            pts = np.array(bbox, dtype=np.int32)
            bx1, by1 = max(0, int(np.min(pts[:, 0]))), max(0, int(np.min(pts[:, 1])))
            bx2, by2 = min(w_v, int(np.max(pts[:, 0]))), min(h_v, int(np.max(pts[:, 1])))

            crop_base = img_var if len(img_var.shape) == 3 else img_hd
            txt_desambiguado = processar_bloco_ocr(txt, bx1, by1, bx2, by2, crop_base)
            if txt_desambiguado:
                textos_proc.append(txt_desambiguado)
                confs.append(prob)

        if not textos_proc:
            continue

        texto_concatenado = "".join(textos_proc)
        conf_media = float(np.mean(confs)) if confs else 0.0

        texto_corrigido, padrao = extrair_melhor_placa_de_texto(texto_concatenado)

        # Avaliação de pontuação
        score = conf_media + (5.0 if padrao is not None else 0.0)
        if len(texto_corrigido) == 7:
            score += 2.0
        if padrao in ["Mercosul", "Antigo"]:
            score += 3.0

        if score > melhor_score:
            melhor_score = score
            melhor_resultado = {
                "texto_bruto": texto_concatenado,
                "texto_corrigido": texto_corrigido[:7],
                "padrao": padrao,
                "confianca": conf_media,
                "img_processada": img_hd
            }

    return melhor_resultado
