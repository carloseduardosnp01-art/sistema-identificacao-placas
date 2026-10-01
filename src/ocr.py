"""
Módulo de Reconhecimento de Placas e Caracteres (100% YOLO).
Pipeline de 2 Estágios de Deep Learning:
- Estágio 1: YOLO Detector de Placas (Localização e recorte)
- Estágio 2: YOLO Detector de Caracteres (Tratamento P&B de Alto Contraste e Inferência de Letras/Números)
Sem dependência de bibliotecas de OCR externas (EasyOCR, Tesseract).
"""

import numpy as np
from typing import Dict, Any, Optional
from .caracteres import ReconhecedorCaracteresYOLO

_reconhecedor_yolo = None


def obter_reconhecedor_yolo() -> ReconhecedorCaracteresYOLO:
    """Inicializa e retorna a instância singleton do reconhecedor de caracteres YOLO 2."""
    global _reconhecedor_yolo
    if _reconhecedor_yolo is None:
        _reconhecedor_yolo = ReconhecedorCaracteresYOLO()
    return _reconhecedor_yolo


def extrair_texto_placa(img_placa: np.ndarray, confianca_minima: float = 0.15) -> Dict[str, Any]:
    """
    Executa a identificação dos caracteres da placa (Estágio 2 100% YOLO):
    1. Recebe o recorte da placa do Estágio 1
    2. Aplica o tratamento de Preto e Branco com Alto Contraste (Otsu)
    3. Executa a inferência direta no modelo YOLO 2 de caracteres
    4. Retorna texto identificado, padrão CONTRAN, confiança e imagem tratada com caixas
    """
    if img_placa is None or img_placa.size == 0:
        return {
            "texto_bruto": "",
            "texto_corrigido": "",
            "padrao": None,
            "confianca": 0.0,
            "caixas_caracteres": [],
            "img_processada": None
        }

    yolo_char = obter_reconhecedor_yolo()
    if yolo_char and yolo_char.disponivel:
        return yolo_char.reconhecer(img_placa, confianca_minima=confianca_minima)

    return {
        "texto_bruto": "",
        "texto_corrigido": "",
        "padrao": None,
        "confianca": 0.0,
        "caixas_caracteres": [],
        "img_processada": None
    }
