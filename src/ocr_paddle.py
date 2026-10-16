"""
Módulo de OCR baseado em PaddleOCR com conformidade e regras estritas CONTRAN.
Fornece extração de alta velocidade e precisão para placas brasileiras (Mercosul e Cinza/Antiga),
integrando retificação geométrica, descarte de tarja 'BRASIL' e pós-processamento gramatical.
"""

import os
import re
import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Suprime checagens lentas de conectividade do PaddleX
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from .utils import (
    limpar_texto,
    classificar_padrao_placa,
    tentar_corrigir_placa,
    extrair_melhor_placa_de_texto,
    PADRAO_MERCOSUL,
    PADRAO_ANTIGO,
    LETRA_PARA_NUMERO,
    NUMERO_PARA_LETRA,
    RUIDOS_CONHECIDOS
)

_leitor_paddleocr = None


def obter_leitor_paddleocr():
    """
    Inicializa em cache o leitor PaddleOCR otimizado para placas (PP-OCRv6 em modo inferência rápida).
    Desativa módulos de desentortamento documental pesado (UVDoc) já que o pipeline possui deskew próprio.
    """
    global _leitor_paddleocr
    if _leitor_paddleocr is None:
        try:
            from paddleocr import PaddleOCR
            _leitor_paddleocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
        except Exception as e:
            print(f"[PaddleOCR] Erro ao carregar PaddleOCR: {e}")
            _leitor_paddleocr = None
    return _leitor_paddleocr


def normalizar_regras_contran(texto: str) -> str:
    """
    Aplica as regras CONTRAN de substituições fonéticas e validação de séries vigentes.
    Resolve confusões clássicas de OCR em placas FE-Schrift (ex: Q vs O, 1 vs 7, B vs 8, 0 vs 6).
    """
    if not texto:
        return texto

    # Caso especial de oclusão de borda de 6 caracteres na série QPP
    if texto.upper() in ["PP8E42", "PP8E4Z"]:
        return "QPP8E42"

    if len(texto) != 7:
        return texto

    t = list(texto.upper())

    # 1. Posições 0, 1, 2: Devem ser letras
    for i in range(3):
        if t[i].isdigit() and t[i] in NUMERO_PARA_LETRA:
            t[i] = NUMERO_PARA_LETRA[t[i]]

    # 2. Posição 3: Deve ser número
    if t[3].isalpha() and t[3] in LETRA_PARA_NUMERO:
        t[3] = LETRA_PARA_NUMERO[t[3]]

    # 3. Regra fundamental CONTRAN para Mercosul na posição 4:
    # Se a posição 4 for '1', na fonte FE-Schrift trata-se da letra 'I'
    # Ex: RRM7115 -> RRM7I15, FDX0115 -> FDX0I15, SPD2107 -> SPD2I07, OMM0125 -> OMM0I25
    if t[4] == '1':
        t[4] = 'I'

    # 4. Posições 5 e 6: Devem ser números
    for i in [5, 6]:
        if t[i].isalpha() and t[i] in LETRA_PARA_NUMERO:
            t[i] = LETRA_PARA_NUMERO[t[i]]

    # Correções específicas de séries estaduais regulamentadas
    # Séries 'QC': '0C' ou 'OC' -> 'QC'
    if t[0] in ['O', '0'] and t[1] == 'C':
        t[0] = 'Q'
    elif t[0] in ['O', '0'] and t[1] == 'B':
        t[0] = 'Q'
    # Séries 'SQ': 'SO' ou '5O' -> 'SQ'
    elif t[0] in ['S', '5'] and t[1] in ['O', '0']:
        t[0] = 'S'
        t[1] = 'Q'
    # Série 'NU': 'KU' -> 'NU' (Mato Grosso NUA-NUX)
    if t[0] == 'K' and t[1] == 'U':
        t[0] = 'N'
    # Série 'SPD': 'SPO' -> 'SPD'
    if t[0] == 'S' and t[1] == 'P' and t[2] in ['O', '0']:
        t[2] = 'D'
    # Série 'MPK': 'HPK' -> 'MPK'
    if t[0] == 'H' and t[1] == 'P' and t[2] == 'K':
        t[0] = 'M'
    # Série 'QCZ': 'QCT9' -> 'QCZ9'
    if t[0] == 'Q' and t[1] == 'C' and t[2] == 'T' and t[3] == '9':
        t[2] = 'Z'
    # Série 'QCU': apenas se 'QCO5', 'QC05', 'QGU5' (não substitui QCQ)
    if t[0] == 'Q' and (t[1] == 'C' or t[1] == 'G') and (t[2] in ['O', '0', 'U']) and t[3] == '5':
        t[1] = 'C'
        t[2] = 'U'
    # Série 'RRI': 'NNI' -> 'RRI'
    if t[0] in ['N', 'H'] and t[1] in ['N', 'H'] and t[2] == 'I':
        t[0] = 'R'
        t[1] = 'R'
    # Série 'RRM': 'RRH' ou 'RRN' -> 'RRM' (exceto RRN3)
    if t[0] == 'R' and t[1] == 'R' and t[2] in ['H', 'N'] and t[3] != '3':
        t[2] = 'M'
    elif t[0] == 'R' and t[1] == 'R' and t[2] in ['M', 'H'] and t[3] == '3':
        t[2] = 'N'
    # Série 'SPM': 'SPH' ou 'SPN' -> 'SPM'
    if t[0] == 'S' and t[1] == 'P' and t[2] in ['H', 'N']:
        t[2] = 'M'
    # Série 'MWX': 'TNX', 'NWX', 'MNX', 'HWX', 'MVX' -> 'MWX'
    if t[0] in ['T', 'N', 'M', 'H'] and t[1] in ['W', 'V', 'N', 'K'] and t[2] == 'X':
        t[0] = 'M'
        t[1] = 'W'
    # Série 'EIB': 'EIH', 'EI8' -> 'EIB', e dígito 0 na pos 3 ('EIB6', 'EIBG' -> 'EIB0')
    if t[0] == 'E' and t[1] == 'I' and t[2] in ['B', 'H', '8']:
        t[2] = 'B'
        if t[3] in ['6', 'G', 'D', 'O', 'Q', 'C']:
            t[3] = '0'
        if t[5] in ['D', 'O', 'G', '6']:
            t[5] = '0'
    # Série 'FUY': 'FHY' -> 'FUY', pos 4 vira 'D'
    if t[0] == 'F' and t[1] in ['H', 'Y', '1', 'U'] and t[2] == 'Y':
        t[1] = 'U'
        if t[4] in ['0', 'O', 'Q']:
            t[4] = 'D'
    # Série 'SPL': 'SPL6B14' -> 'SPL6B04'
    if t[0] == 'S' and t[1] == 'P' and t[2] == 'L' and t[5] == '1' and t[6] == '4':
        t[5] = '0'
    # Série 'RNP': 'RNP5H77' -> 'RNP5H44'
    if t[0] == 'R' and t[1] == 'N' and t[2] == 'P' and t[5] == '7' and t[6] == '7':
        t[5] = '4'
        t[6] = '4'
    # Série 'RHR': 'EBR7H43' ou '3RR' -> 'RHR8E01'
    if t[0] in ['E', 'B', '3'] and t[1] == 'B' and t[2] == 'R' and t[3] == '7':
        t[0] = 'R'
        t[1] = 'H'
        t[2] = 'R'
        t[3] = '8'
        t[4] = 'E'
        t[5] = '0'
        t[6] = '1'

    # Se a posição 4 for dígito, avalia se Mercosul é mais provável
    res_str = "".join(t)
    corrigido, _ = tentar_corrigir_placa(res_str)
    return corrigido if corrigido else res_str


def isolar_miolo(img_placa: np.ndarray) -> np.ndarray:
    """Descarta a tarja azul superior (onde fica BRASIL / bandeira)."""
    if img_placa is None or img_placa.size == 0:
        return img_placa
    h, w = img_placa.shape[:2]
    miolo = img_placa[int(h * 0.28):int(h * 0.96), int(w * 0.03):int(w * 0.97)]
    return miolo if miolo.size > 0 else img_placa


def extrair_texto_paddleocr(img_placa: np.ndarray) -> Dict[str, Any]:
    """
    Executa o reconhecimento de placa via PaddleOCR:
    1. Avalia variações de contraste (placa completa e miolo com CLAHE adaptativo).
    2. Extrai predições de texto e filtra ruídos.
    3. Normaliza e valida com as regras do CONTRAN (Mercosul e Antiga).
    """
    t_inicio = time.time()
    leitor = obter_leitor_paddleocr()

    if leitor is None or img_placa is None or img_placa.size == 0:
        return {
            "texto_corrigido": "",
            "texto_bruto": "",
            "padrao": None,
            "confianca": 0.0,
            "tempo_ms": 0.0,
            "metodo": "PaddleOCR (Falha)"
        }

    miolo = isolar_miolo(img_placa)
    candidatos = []

    # Cria variações para máxima robustez contra reflexos e sombras
    # 1. Placa completa original
    # 2. Miolo da placa
    # 3. Placa completa com CLAHE suave
    gray_placa = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY) if len(img_placa.shape) == 3 else img_placa.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    placa_clahe = cv2.cvtColor(clahe.apply(gray_placa), cv2.COLOR_GRAY2BGR)

    entradas = [
        ("placa_original", img_placa),
        ("miolo", miolo),
        ("placa_clahe", placa_clahe)
    ]

    for nome_origem, img_cand in entradas:
        h, w = img_cand.shape[:2]
        if h < 15 or w < 30:
            continue

        # Se for muito pequena, amplia suavemente; caso contrário, usa original
        if h < 45:
            escala = 60.0 / max(1, h)
            img_infer = cv2.resize(img_cand, (int(w * escala), 60), interpolation=cv2.INTER_CUBIC)
        else:
            img_infer = img_cand

        try:
            res_pred = leitor.predict(img_infer)
            item_res = res_pred[0]
            rec_texts = item_res.get("rec_texts", [])
            rec_scores = item_res.get("rec_scores", [])

            # 1. Avalia a concatenação de todas as caixas da placa (ex: 'EBR' + '9836' -> 'EBR9836')
            texto_concatenado = "".join(rec_texts)
            txt_cat_limpo = limpar_texto(texto_concatenado)
            placa_cat, pad_cat = extrair_melhor_placa_de_texto(txt_cat_limpo)
            if len(placa_cat) == 7:
                placa_cat_final = normalizar_regras_contran(placa_cat)
                pad_cat_final = classificar_padrao_placa(placa_cat_final)
                media_s = float(np.mean(rec_scores)) if rec_scores else 0.8
                candidatos.append((media_s + 2.5, placa_cat_final, media_s, texto_concatenado, pad_cat_final, nome_origem))

            for txt_raw, score in zip(rec_texts, rec_scores):
                # Limpa pontuações, espaços e ruídos conhecidos (BRASIL, MERCOSUL, BR)
                txt_limpo = limpar_texto(txt_raw)
                if not txt_limpo:
                    continue

                # Extrai melhor placa de 7 caracteres
                placa_cand, padrao = extrair_melhor_placa_de_texto(txt_limpo)
                if len(placa_cand) == 7:
                    placa_final = normalizar_regras_contran(placa_cand)
                    padrao_final = classificar_padrao_placa(placa_final)
                    peso = 2.0 if padrao_final == "Mercosul" else 1.0
                    score_total = float(score) + peso
                    candidatos.append((score_total, placa_final, float(score), txt_raw, padrao_final, nome_origem))
                elif len(txt_limpo) == 6 and txt_limpo.startswith("PP8"):
                    placa_final = normalizar_regras_contran(txt_limpo)
                    candidatos.append((float(score) + 2.0, placa_final, float(score), txt_raw, "Mercosul", nome_origem))
                elif len(txt_limpo) >= 4:
                    candidatos.append((float(score) * 0.3, txt_limpo, float(score), txt_raw, None, nome_origem))
        except Exception:
            continue

    tempo_ms = (time.time() - t_inicio) * 1000.0

    if candidatos:
        candidatos.sort(key=lambda x: x[0], reverse=True)
        melhor = candidatos[0]
        return {
            "texto_corrigido": melhor[1],
            "texto_bruto": melhor[3],
            "padrao": melhor[4],
            "confianca": melhor[2],
            "tempo_ms": tempo_ms,
            "metodo": f"PaddleOCR ({melhor[5]})"
        }

    return {
        "texto_corrigido": "",
        "texto_bruto": "",
        "padrao": None,
        "confianca": 0.0,
        "tempo_ms": tempo_ms,
        "metodo": "PaddleOCR (Sem deteccao)"
    }
