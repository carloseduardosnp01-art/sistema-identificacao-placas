"""
Módulo de OCR e Reconhecimento Híbrido de Placas (YOLO + EasyOCR Adaptativo).
Combina o detector YOLO de caracteres com o EasyOCR alimentado por pré-processamento
de alto contraste (CLAHE e Sharpen), garantindo máxima taxa de leitura.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from .caracteres import ReconhecedorCaracteresYOLO, gerar_variacoes_alto_contraste
from .utils import extrair_melhor_placa_de_texto, limpar_texto, tentar_corrigir_placa, classificar_padrao_placa
from .desambiguacao import refinar_placa_completa

_reconhecedor_yolo = None
_leitor_easyocr = None
_leitor_paddleocr = None


def obter_reconhecedor_yolo() -> ReconhecedorCaracteresYOLO:
    """Inicializa e retorna o singleton do reconhecedor YOLO de caracteres."""
    global _reconhecedor_yolo
    if _reconhecedor_yolo is None:
        _reconhecedor_yolo = ReconhecedorCaracteresYOLO()
    return _reconhecedor_yolo


def obter_leitor_paddleocr():
    """Inicializa em cache o leitor PaddleOCR otimizado para placas (PP-OCRv6)."""
    global _leitor_paddleocr
    if _leitor_paddleocr is None:
        from .ocr_paddle import obter_leitor_paddleocr as _obter_paddle
        _leitor_paddleocr = _obter_paddle()
    return _leitor_paddleocr


def obter_leitor_easyocr():
    """Inicializa em cache o leitor EasyOCR em português e inglês."""
    global _leitor_easyocr
    if _leitor_easyocr is None:
        try:
            import easyocr
            print("[OCR Híbrido] Carregando motor EasyOCR...")
            _leitor_easyocr = easyocr.Reader(['en', 'pt'], gpu=False)
        except Exception as e:
            print(f"[OCR Híbrido] Aviso: EasyOCR não disponível: {e}")
            _leitor_easyocr = None
    return _leitor_easyocr


def extrair_texto_placa(img_placa: np.ndarray, motor: str = "paddle", confianca_minima: float = 0.12) -> Dict[str, Any]:
    """
    Pipeline de Extração Inteligente:
    - Se motor == 'paddle': utiliza PaddleOCR com regras CONTRAN otimizadas.
    - Se motor == 'hibrido' ou 'misto': delega para o pipeline misto.
    """
    if motor.lower() in ["paddle", "paddleocr"]:
        from .ocr_paddle import extrair_texto_paddleocr
        return extrair_texto_paddleocr(img_placa)
    from .pipeline_misto import executar_pipeline_misto
    return executar_pipeline_misto(img_placa)

