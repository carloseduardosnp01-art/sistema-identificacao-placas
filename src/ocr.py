"""
Módulo de Reconhecimento de Caracteres para Placas Brasileiras.
Suporta arquitetura em 2 Estágios:
- Modo Principal: YOLO 2 de Caracteres (models/yolo_caracteres.pt) treinado em 36 classes (0-9 e A-Z)
- Modo Fallback: Leitor OCR com placa completa RGB
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from .utils import extrair_melhor_placa_de_texto, tentar_corrigir_placa, limpar_texto

_modelo_caracteres = None
_reader_easyocr = None
_caminho_yolo_chars = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolo_caracteres.pt")


def obter_modelo_yolo_caracteres():
    """Carrega o modelo YOLO especializado em reconhecimento de caracteres brasileiros."""
    global _modelo_caracteres
    if _modelo_caracteres is None and os.path.exists(_caminho_yolo_chars):
        try:
            from ultralytics import YOLO
            _modelo_caracteres = YOLO(_caminho_yolo_chars)
            print("[OCR] YOLO 2 de Caracteres carregado com sucesso!")
        except Exception as e:
            print(f"[OCR] Falha ao carregar YOLO 2: {e}")
            _modelo_caracteres = None
    return _modelo_caracteres


def obter_leitor_easyocr():
    """Inicializa o leitor EasyOCR como fallback."""
    global _reader_easyocr
    if _reader_easyocr is None:
        import easyocr
        _reader_easyocr = easyocr.Reader(['pt', 'en'], gpu=True)
    return _reader_easyocr


def pre_processar_placa(img_placa: np.ndarray) -> np.ndarray:
    """Retorna a placa original em alta definição natural (sem cortes ou distorções de cor)."""
    if img_placa is None or img_placa.size == 0:
        return img_placa

    h, w = img_placa.shape[:2]
    escala = 160.0 / max(1, h)
    nova_w = int(w * escala)
    return cv2.resize(img_placa, (nova_w, 160), interpolation=cv2.INTER_LANCZOS4)


def extrair_texto_placa(img_placa: np.ndarray) -> Dict[str, Any]:
    """
    Pipeline de Extração de Placas:
    1. Se models/yolo_caracteres.pt existir, usa o YOLO 2 (Detecção direta de letras e números)
    2. Caso contrário, executa o leitor OCR com a placa completa natural
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
    yolo_chars = obter_modelo_yolo_caracteres()

    # --- MODO 1: YOLO 2 DE CARACTERES (MÁXIMA PRECISÃO) ---
    if yolo_chars is not None:
        try:
            res = yolo_chars(img_hd, conf=0.25, verbose=False)[0]
            boxes = res.boxes
            
            if len(boxes) > 0:
                caracteres_detectados = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    nome_char = res.names[cls_id]
                    caracteres_detectados.append({
                        "x": x1,
                        "char": str(nome_char).upper(),
                        "conf": conf
                    })
                
                # Ordena os caracteres da esquerda para a direita pela posição X
                caracteres_detectados = sorted(caracteres_detectados, key=lambda c: c["x"])
                
                texto_bruto = "".join([c["char"] for c in caracteres_detectados])
                conf_media = float(np.mean([c["conf"] for c in caracteres_detectados]))
                
                texto_corrigido, padrao = extrair_melhor_placa_de_texto(texto_bruto)
                
                return {
                    "texto_bruto": texto_bruto,
                    "texto_corrigido": texto_corrigido[:7],
                    "padrao": padrao,
                    "confianca": conf_media,
                    "img_processada": img_hd
                }
        except Exception as e:
            print(f"[OCR] Erro na inferência do YOLO 2: {e}")

    # --- MODO 2: FALLBACK COM EASYOCR NA PLACA COMPLETA ---
    reader = obter_leitor_easyocr()
    cinza_suave = cv2.cvtColor(img_hd, cv2.COLOR_BGR2GRAY) if len(img_hd.shape) == 3 else img_hd
    variacoes = [
        ("rgb_original", img_hd),
        ("cinza_natural", cinza_suave)
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
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1,
                paragraph=False,
                contrast_ths=0.1
            )
        except Exception:
            continue

        if not resultados:
            continue

        resultados_ordenados = sorted(resultados, key=lambda r: min(pt[0] for pt in r[0]))
        textos = [limpar_texto(r[1]) for r in resultados_ordenados if limpar_texto(r[1]) and r[2] > 0.05]
        confs = [r[2] for r in resultados_ordenados if limpar_texto(r[1]) and r[2] > 0.05]

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

    return melhor_resultado
