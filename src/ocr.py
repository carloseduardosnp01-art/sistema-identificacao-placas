"""
Módulo de OCR (Optical Character Recognition) e Pré-processamento de Imagens.
Responsável por aprimorar a região da placa e extrair os caracteres alfanuméricos.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from .utils import tentar_corrigir_placa, limpar_texto

# Instância preguiçosa (lazy loading) do EasyOCR para economizar memória e acelerar inicialização
_reader = None


def obter_leitor_ocr():
    """Inicializa e retorna o leitor EasyOCR (carregado apenas na primeira chamada)."""
    global _reader
    if _reader is None:
        import easyocr
        # Carrega o modelo de OCR em português/inglês (funciona com CPU ou GPU automaticamente)
        _reader = easyocr.Reader(['pt', 'en'], gpu=True)
    return _reader


def pre_processar_placa(img_placa: np.ndarray) -> np.ndarray:
    """
    Aplica técnicas de visão computacional com OpenCV para melhorar a legibilidade dos caracteres:
    1. Redimensionamento
    2. Conversão para escala de cinza
    3. Ajuste de contraste adaptativo (CLAHE)
    4. Remoção de ruídos (Filtro Bilateral)
    5. Binarização adaptativa / Otsu
    """
    if img_placa is None or img_placa.size == 0:
        return img_placa

    # 1. Aumenta a resolução caso a placa esteja muito pequena
    altura, largura = img_placa.shape[:2]
    if largura < 200:
        fator = 200 / largura
        img_placa = cv2.resize(img_placa, (int(largura * fator), int(altura * fator)), interpolation=cv2.INTER_CUBIC)

    # 2. Converte para escala de cinza
    if len(img_placa.shape) == 3:
        cinza = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY)
    else:
        cinza = img_placa.copy()

    # 3. Melhora de contraste com CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contraste = clahe.apply(cinza)

    # 4. Filtro bilateral preserva bordas dos caracteres enquanto remove ruído
    suave = cv2.bilateralFilter(contraste, 11, 17, 17)

    # 5. Binarização usando Otsu
    _, binarizada = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binarizada


def extrair_texto_placa(img_placa: np.ndarray) -> Dict[str, Any]:
    """
    Executa o processo completo de OCR sobre o recorte de uma placa.
    Retorna um dicionário com:
    - 'texto_bruto': texto lido diretamente pelo OCR
    - 'texto_corrigido': texto após heurísticas de placas brasileiras
    - 'padrao': 'Mercosul', 'Antigo' ou None
    - 'confianca': nível de confiança do OCR (0 a 1)
    - 'img_processada': imagem da placa após pré-processamento
    """
    if img_placa is None or img_placa.size == 0:
        return {
            "texto_bruto": "",
            "texto_corrigido": "",
            "padrao": None,
            "confianca": 0.0,
            "img_processada": None
        }

    # Pré-processamento
    img_proc = pre_processar_placa(img_placa)

    reader = obter_leitor_ocr()

    # Tentativa 1: Executar OCR na imagem pré-processada (binarizada)
    resultados = reader.readtext(
        img_proc,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        detail=1
    )

    # Tentativa 2: Se não encontrar nada na binarizada, tenta na imagem original em cinza
    if not resultados:
        cinza_orig = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY) if len(img_placa.shape) == 3 else img_placa
        resultados = reader.readtext(
            cinza_orig,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1
        )

    if not resultados:
        return {
            "texto_bruto": "",
            "texto_corrigido": "",
            "padrao": None,
            "confianca": 0.0,
            "img_processada": img_proc
        }

    # Concatena os textos detectados ordenando pela posição horizontal
    # resultado: (bbox, text, prob)
    resultados_ordenados = sorted(resultados, key=lambda r: r[0][0][0])
    texto_completo = "".join([r[1] for r in resultados_ordenados])
    confiancas = [r[2] for r in resultados_ordenados]
    confianca_media = float(np.mean(confiancas)) if confiancas else 0.0

    texto_limpo = limpar_texto(texto_completo)
    texto_corrigido, padrao = tentar_corrigir_placa(texto_limpo)

    return {
        "texto_bruto": texto_limpo,
        "texto_corrigido": texto_corrigido,
        "padrao": padrao,
        "confianca": confianca_media,
        "img_processada": img_proc
    }
