"""
Módulo de OCR (Optical Character Recognition) de Alta Precisão para Placas Brasileiras.
Processa placas Mercosul e Antigas em RGB nativo, com suporte a múltiplas escalas e correção FE-Schrift.
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
    """
    Retorna a placa original em alta definição e no formato RGB correto.
    Garante que os canais de cor e resolução estejam ideais para leitura.
    """
    if img_placa is None or img_placa.size == 0:
        return img_placa

    # Garante formato RGB
    if len(img_placa.shape) == 3 and img_placa.shape[2] == 3:
        # Se veio em BGR do OpenCV, converte para RGB
        img_rgb = cv2.cvtColor(img_placa, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_placa.copy()

    h, w = img_rgb.shape[:2]
    escala = 200.0 / max(1, h)
    nova_w = int(w * escala)
    return cv2.resize(img_rgb, (nova_w, 200), interpolation=cv2.INTER_LANCZOS4)


def extrair_texto_placa(img_placa: np.ndarray) -> Dict[str, Any]:
    """
    Pipeline de Alta Precisão de OCR:
    1. Trata a imagem em RGB nativo em alta definição
    2. Aplica EasyOCR multi-escala
    3. Converte caracteres com gramática estrita brasileira (Mercosul e Antigo)
    """
    if img_placa is None or img_placa.size == 0:
        return {
            "texto_bruto": "",
            "texto_corrigido": "",
            "padrao": None,
            "confianca": 0.0,
            "img_processada": None
        }

    img_hd = pre_processar_placa(img_placa)
    reader = obter_leitor_easyocr()

    # Variações: 1. RGB Natural HD | 2. Tons de cinza de alto contraste suave
    cinza = cv2.cvtColor(img_hd, cv2.COLOR_RGB2GRAY) if len(img_hd.shape) == 3 else img_hd
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cinza)

    variacoes = [
        ("rgb_hd", img_hd),
        ("clahe_suave", clahe)
    ]

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

        # Ordena caixas da esquerda para a direita
        resultados_ordenados = sorted(resultados, key=lambda r: min(pt[0] for pt in r[0]))
        textos = []
        confs = []

        for bbox, txt, prob in resultados_ordenados:
            txt_limpo = limpar_texto(txt)
            if txt_limpo and prob > 0.05:
                textos.append(txt_limpo)
                confs.append(prob)

        if not textos:
            continue

        texto_concatenado = "".join(textos)
        conf_media = float(np.mean(confs)) if confs else 0.0

        texto_corrigido, padrao = extrair_melhor_placa_de_texto(texto_concatenado)

        score = conf_media + (3.0 if padrao is not None else 0.0)
        if len(texto_corrigido) == 7:
            score += 1.0

        if score > melhor_score:
            melhor_score = score
            melhor_resultado = {
                "texto_bruto": texto_concatenado,
                "texto_corrigido": texto_corrigido[:7],
                "padrao": padrao,
                "confianca": conf_media,
                "img_processada": img_hd
            }

            if padrao is not None and conf_media > 0.60:
                break

    return melhor_resultado
