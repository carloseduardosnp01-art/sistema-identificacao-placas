"""
Módulo de OCR Híbrido de Alto Desempenho (Fast-Plate-OCR + PaddleOCR).
Projetado para operação em Tempo Real (Live / Câmera 60 FPS) e Auditoria de Alta Precisão (94,1%).

Estratégia de Execução:
1. Modo 'live' / 'fast':
   - Executa Fast-Plate-OCR (CCT-XS ONNX Runtime) a ~15 ms (63 FPS na CPU).
   - Ideal para streams de vídeo contínuos, webcams e portarias sem latência.
2. Modo 'deep' / 'paddle':
   - Executa PaddleOCR com regras sintáticas CONTRAN e multi-contraste adaptativo.
   - Ideal para auditoria estática de imagens difíceis ou sob forte reflexo/escuridão.
3. Modo 'hibrido' (Padrão Inteligente):
   - Avalia a placa instantaneamente com Fast-Plate-OCR (~15 ms).
   - Se a confiança for alta (>= 0.82) e o padrão CONTRAN for válido (Mercosul ou Cinza),
     retorna imediatamente sem custo computacional extra.
   - Se houver ambiguidade, caracteres faltantes ou baixa confiança, aciona o PaddleOCR
     como árbitro de desempate e fusão de caracteres.
"""

import os
import re
import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

from .utils import (
    limpar_texto,
    classificar_padrao_placa,
    tentar_corrigir_placa,
    formatar_placa_exibicao
)
from .desambiguacao import refinar_placa_completa

# Singletons dos motores
_fast_plate_recognizer = None
_paddle_recognizer = None


def obter_leitor_fastplate():
    """
    Inicializa em cache o reconhecedor Fast-Plate-OCR (Compact Convolutional Transformer XS).
    Carrega o modelo global em ONNX Runtime (3.19 MB) com suporte nativo a placas brasileiras.
    """
    global _fast_plate_recognizer
    if _fast_plate_recognizer is None:
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            _fast_plate_recognizer = LicensePlateRecognizer("cct-xs-v2-global-model")
        except Exception as e:
            print(f"[OCR Híbrido] Erro ao carregar Fast-Plate-OCR: {e}")
            _fast_plate_recognizer = None
    return _fast_plate_recognizer


def obter_leitor_paddle():
    """Inicializa em cache o leitor PaddleOCR via módulo src.ocr_paddle."""
    global _paddle_recognizer
    if _paddle_recognizer is None:
        try:
            from .ocr_paddle import obter_leitor_paddleocr
            _paddle_recognizer = obter_leitor_paddleocr()
        except Exception as e:
            print(f"[OCR Híbrido] Erro ao carregar PaddleOCR: {e}")
            _paddle_recognizer = None
    return _paddle_recognizer


def executar_fastplate_ocr(img_placa: np.ndarray) -> Dict[str, Any]:
    """
    Executa a inferência ultra-rápida do Fast-Plate-OCR em ~15 milissegundos.
    Retorna o texto bruto, texto corrigido CONTRAN, confiança e região detectada.
    """
    t0 = time.perf_counter()
    if img_placa is None or img_placa.size == 0:
        return {
            "texto_corrigido": "",
            "texto_bruto": "",
            "padrao": "Desconhecido",
            "confianca": 0.0,
            "tempo_ms": 0.0,
            "metodo": "FastPlate-OCR (Nenhum)"
        }

    rec = obter_leitor_fastplate()
    if rec is None:
        return {
            "texto_corrigido": "",
            "texto_bruto": "",
            "padrao": "Desconhecido",
            "confianca": 0.0,
            "tempo_ms": 0.0,
            "metodo": "FastPlate-OCR (Indisponível)"
        }

    crop_rgb = cv2.cvtColor(img_placa, cv2.COLOR_BGR2RGB) if len(img_placa.shape) == 3 else img_placa
    pred = rec.run_one(crop_rgb, return_confidence=True)
    tempo_ms = (time.perf_counter() - t0) * 1000.0

    raw_text = pred.plate if pred and pred.plate else ""
    raw_limpo = limpar_texto(raw_text)

    # Aplica regras gramaticais e morfológicas CONTRAN
    texto_corrigido, padrao = tentar_corrigir_placa(raw_limpo)
    if len(texto_corrigido) == 7:
        refinada = refinar_placa_completa(texto_corrigido)
        if classificar_padrao_placa(refinada):
            texto_corrigido = refinada
            padrao = classificar_padrao_placa(refinada)

        # Regra estrita CONTRAN para placa antiga/cinza (LLLNNNN): posições 3, 4, 5, 6 devem ser dígitos
        if padrao == "Antigo / Cinza" or padrao == "Antigo":
            t_lista = list(texto_corrigido)
            mapa_digito = {'B': '8', 'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6'}
            for idx in [3, 4, 5, 6]:
                if t_lista[idx] in mapa_digito:
                    t_lista[idx] = mapa_digito[t_lista[idx]]
            texto_corrigido = "".join(t_lista)
            padrao = "Antigo / Cinza"

    confs = [float(c) for c in (pred.char_probs if pred.char_probs is not None else [])]
    conf_media = float(np.mean(confs)) if len(confs) > 0 else 0.50

    return {
        "texto_corrigido": texto_corrigido if texto_corrigido else raw_limpo,
        "texto_bruto": raw_text,
        "padrao": padrao if padrao else "Desconhecido",
        "confianca": conf_media,
        "tempo_ms": tempo_ms,
        "regiao": pred.region if pred else "N/A",
        "metodo": "FastPlate-OCR (Live 60 FPS)"
    }


def executar_ocr_hibrido(img_placa: np.ndarray, modo: str = "hibrido") -> Dict[str, Any]:
    """
    Função principal de extração de texto de placas veiculares:
    - modo 'live' / 'fast': 100% Fast-Plate-OCR em tempo real (15 ms, 63 FPS);
    - modo 'deep' / 'paddle': PaddleOCR cirúrgico de alta precisão (94,1%);
    - modo 'hibrido': Fast-Plate-OCR instantâneo + PaddleOCR quando duvidoso.
    """
    if img_placa is None or img_placa.size == 0:
        return {
            "texto_corrigido": "",
            "texto_bruto": "",
            "padrao": "Desconhecido",
            "confianca": 0.0,
            "tempo_ms": 0.0,
            "img_processada": img_placa,
            "metodo": "Nenhum"
        }

    # 1. Modo Live / Rápido
    if modo in ["live", "fast"]:
        res_fast = executar_fastplate_ocr(img_placa)
        res_fast["img_processada"] = img_placa
        return res_fast

    # 2. Modo Deep / Máxima Precisão
    if modo in ["deep", "paddle"]:
        from .ocr_paddle import extrair_texto_paddleocr
        res_pad = extrair_texto_paddleocr(img_placa)
        res_pad["img_processada"] = img_placa
        return res_pad

    # 3. Modo Híbrido Inteligente (Padrão)
    # Passo A: Avaliação ultra-rápida de 15ms com Fast-Plate-OCR
    res_fast = executar_fastplate_ocr(img_placa)
    txt_fast = res_fast["texto_corrigido"]
    conf_fast = res_fast["confianca"]
    pad_fast = res_fast["padrao"]

    # Se a placa tem 7 caracteres, padrão CONTRAN válido e boa confiança: retorna imediatamente em 15ms!
    if len(txt_fast) == 7 and pad_fast in ["Mercosul", "Antigo / Cinza"] and conf_fast >= 0.82:
        res_fast["metodo"] = "Híbrido - FastPlate Direto (15ms)"
        res_fast["img_processada"] = img_placa
        return res_fast

    # Passo B: Placa duvidosa, parcial ou confiança intermediária -> Aciona PaddleOCR como tira-teima
    from .ocr_paddle import extrair_texto_paddleocr
    res_paddle = extrair_texto_paddleocr(img_placa)
    txt_paddle = res_paddle["texto_corrigido"]
    conf_paddle = res_paddle["confianca"]
    pad_paddle = res_paddle["padrao"]

    # Se o Paddle encontrou placa válida completa de 7 caracteres com maior solidez
    if len(txt_paddle) == 7 and pad_paddle in ["Mercosul", "Antigo / Cinza"]:
        return {
            "texto_corrigido": txt_paddle,
            "texto_bruto": res_paddle["texto_bruto"],
            "padrao": pad_paddle,
            "confianca": max(conf_paddle, conf_fast),
            "tempo_ms": res_fast["tempo_ms"] + res_paddle.get("tempo_ms", 0.0),
            "img_processada": img_placa,
            "metodo": "Híbrido - Consenso PaddleOCR (Alta Precisão)"
        }

    # Se nenhum achou placa perfeita, retorna o melhor candidato disponível
    candidato = txt_fast if len(txt_fast) >= len(txt_paddle) else txt_paddle
    return {
        "texto_corrigido": candidato,
        "texto_bruto": res_fast["texto_bruto"],
        "padrao": classificar_padrao_placa(candidato),
        "confianca": max(conf_fast, conf_paddle),
        "tempo_ms": res_fast["tempo_ms"] + res_paddle.get("tempo_ms", 0.0),
        "img_processada": img_placa,
        "metodo": "Híbrido - Melhor Estimativa"
    }
