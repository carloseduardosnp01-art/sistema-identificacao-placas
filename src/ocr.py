"""
Módulo de OCR (Optical Character Recognition) para Placas Brasileiras (Mercosul e Antiga).
Processa a placa completa original em alta resolução sem cortes destrutivos ou filtros agressivos.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from .utils import extrair_melhor_placa_de_texto, tentar_corrigir_placa, limpar_texto

# Instância preguiçosa (lazy loading) do EasyOCR
_reader = None


def obter_leitor_ocr():
    """Inicializa e retorna o leitor EasyOCR com suporte a GPU/CPU."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['pt', 'en'], gpu=True)
    return _reader


def pre_processar_placa(img_placa: np.ndarray) -> np.ndarray:
    """
    Retorna a placa em alta definição natural para visualização e leitura.
    Mantém 100% da imagem original sem cortes ou binarizações destrutivas.
    """
    if img_placa is None or img_placa.size == 0:
        return img_placa

    h, w = img_placa.shape[:2]
    # Redimensiona para resolução ideal (~160px de altura) com alta nitidez Lanczos
    escala = 160.0 / max(1, h)
    nova_w = int(w * escala)
    return cv2.resize(img_placa, (nova_w, 160), interpolation=cv2.INTER_LANCZOS4)


def extrair_texto_placa(img_placa: np.ndarray) -> Dict[str, Any]:
    """
    Pipeline de Extração de Placas:
    1. Usa a placa completa em cores reais (RGB original em alta definição)
    2. Utiliza o detector de texto profundo para localizar a sequência
    3. Extrai e valida rigorosamente o padrão brasileiro de 7 caracteres
    """
    if img_placa is None or img_placa.size == 0:
        return {
            "texto_bruto": "",
            "texto_corrigido": "",
            "padrao": None,
            "confianca": 0.0,
            "img_processada": None
        }

    # Prepara a imagem em alta resolução com aspecto natural
    img_hd = pre_processar_placa(img_placa)
    reader = obter_leitor_ocr()

    # Variações naturais: 1. Imagem colorida original em HD | 2. Tons de cinza suaves
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
                contrast_ths=0.1,
                adjust_contrast=0.5
            )
        except Exception:
            continue

        if not resultados:
            continue

        # Ordena caixas detectadas da esquerda para a direita
        resultados_ordenados = sorted(resultados, key=lambda r: min(pt[0] for pt in r[0]))

        textos_filtrados = []
        confiancas = []
        for bbox, txt, prob in resultados_ordenados:
            txt_limpo = limpar_texto(txt)
            if txt_limpo and prob > 0.05:
                textos_filtrados.append(txt_limpo)
                confiancas.append(prob)

        if not textos_filtrados:
            continue

        texto_concatenado = "".join(textos_filtrados)
        conf_media = float(np.mean(confiancas)) if confiancas else 0.0

        # Aplica extração rigorosa de 7 caracteres
        texto_corrigido, padrao = extrair_melhor_placa_de_texto(texto_concatenado)

        # Cálculo de pontuação: padrão reconhecido tem bônus
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

            if padrao is not None and conf_media > 0.65:
                break

    if len(melhor_resultado["texto_corrigido"]) > 7:
        melhor_resultado["texto_corrigido"] = melhor_resultado["texto_corrigido"][:7]

    return melhor_resultado
