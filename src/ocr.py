"""
Módulo Unificado de OCR e Reconhecimento de Placas Veiculares.
Integra o novo motor Híbrido de Alta Performance (Fast-Plate-OCR 63 FPS + PaddleOCR 94,1%).
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional

from .ocr_hibrido import (
    executar_ocr_hibrido,
    executar_fastplate_ocr,
    obter_leitor_fastplate,
    obter_leitor_paddle
)


def extrair_texto_placa(img_placa: np.ndarray, modo: str = "hibrido", confianca_minima: float = 0.12) -> Dict[str, Any]:
    """
    Função principal de extração de texto de placas veiculares:
    - modo 'hibrido' (default): Fast-Plate-OCR instantâneo + PaddleOCR quando duvidoso;
    - modo 'live' / 'fast': 100% Fast-Plate-OCR para operação em tempo real (63 FPS);
    - modo 'deep' / 'paddle': PaddleOCR cirúrgico de alta precisão (94,1%).
    """
    return executar_ocr_hibrido(img_placa, modo=modo)
