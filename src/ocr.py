"""
Módulo de OCR (Optical Character Recognition) Avançado para Placas Brasileiras (Mercosul e Antiga).
Inclui correção de inclinação (deskewing), realce morfológico FE-Schrift e parâmetros calibrados.
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


def endireitar_placa(img_placa: np.ndarray) -> np.ndarray:
    """
    Detecta e corrige a inclinação angular da placa (fotos tiradas de lado).
    Garante que os 7 caracteres fiquem perfeitamente alinhados na horizontal.
    """
    if img_placa is None or img_placa.size == 0:
        return img_placa

    try:
        cinza = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY) if len(img_placa.shape) == 3 else img_placa
        suave = cv2.GaussianBlur(cinza, (5, 5), 0)
        _, thresh = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos:
            c_max = max(contornos, key=cv2.contourArea)
            area_total = img_placa.shape[0] * img_placa.shape[1]
            if cv2.contourArea(c_max) > (area_total * 0.15):
                rect = cv2.minAreaRect(c_max)
                (cx, cy), (w, h), angulo = rect
                if w < h:
                    angulo = angulo - 90
                # Se houver inclinação real (-35° a +35°)
                if -35 < angulo < 35 and abs(angulo) > 1.5:
                    h_img, w_img = img_placa.shape[:2]
                    mat = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), angulo, 1.0)
                    return cv2.warpAffine(
                        img_placa, mat, (w_img, h_img),
                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                    )
    except Exception:
        pass

    return img_placa


def isolar_regiao_caracteres(img_placa: np.ndarray) -> np.ndarray:
    """
    Remove com precisão a faixa superior azul com 'BRASIL' (~28% do topo)
    e as bordas com logotipo 'BR' e moldura (~5% das laterais).
    """
    if img_placa is None or img_placa.size == 0:
        return img_placa

    h, w = img_placa.shape[:2]
    y1 = int(h * 0.28)
    y2 = int(h * 0.94)
    x1 = int(w * 0.05)
    x2 = int(w * 0.95)

    crop_util = img_placa[y1:y2, x1:x2]
    if crop_util.size > 0 and crop_util.shape[0] > 10 and crop_util.shape[1] > 20:
        return crop_util
    return img_placa


def preparar_variacoes_imagem(img_crop: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Gera variações otimizadas para o leitor OCR:
    1. CLAHE forte + Sharpen suave (ideal para caracteres FE-Schrift como N, 1, W)
    2. CLAHE suave com filtro bilateral
    3. Binarização morfológica
    """
    h, w = img_crop.shape[:2]

    # Redimensiona para resolução ideal (~140px de altura para alta definição)
    escala = 140.0 / max(1, h)
    nova_largura = int(w * escala)
    img_redim = cv2.resize(img_crop, (nova_largura, 140), interpolation=cv2.INTER_LANCZOS4)

    # Conversão para tons de cinza
    if len(img_redim.shape) == 3:
        cinza = cv2.cvtColor(img_redim, cv2.COLOR_BGR2GRAY)
    else:
        cinza = img_redim.copy()

    # Variação 1: CLAHE forte + Sharpen
    clahe_forte = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(cinza)
    kernel_sharp = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
    cinza_sharp = cv2.filter2D(clahe_forte, -1, kernel_sharp)

    # Variação 2: CLAHE equilibrado com filtro bilateral
    clahe_suave = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6)).apply(cinza)
    cinza_bilateral = cv2.bilateralFilter(clahe_suave, 7, 50, 50)

    # Variação 3: Binarização Otsu limpa
    _, binarizada = cv2.threshold(cinza_bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binarizada) < 127:
        binarizada = cv2.bitwise_not(binarizada)
    kernel_morf = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binarizada_limpa = cv2.morphologyEx(binarizada, cv2.MORPH_CLOSE, kernel_morf)

    return [
        ("clahe_sharp", cinza_sharp),
        ("clahe_bilateral", cinza_bilateral),
        ("binarizada", binarizada_limpa)
    ]


def pre_processar_placa(img_placa: np.ndarray) -> np.ndarray:
    """Retorna a imagem pré-processada da região de caracteres para exibição."""
    placa_alinhada = endireitar_placa(img_placa)
    miolo = isolar_regiao_caracteres(placa_alinhada)
    variacoes = preparar_variacoes_imagem(miolo)
    return variacoes[0][1]


def extrair_texto_placa(img_placa: np.ndarray) -> Dict[str, Any]:
    """
    Pipeline completo de extração de placas:
    1. Alinha a inclinação da placa
    2. Isola a zona de caracteres (removendo faixa BRASIL e molduras)
    3. Aplica Multi-Pass OCR com parâmetros calibrados (mag_ratio=1.5, contrast_ths=0.1)
    4. Extrai rigorosamente a sequência de 7 caracteres
    """
    if img_placa is None or img_placa.size == 0:
        return {
            "texto_bruto": "",
            "texto_corrigido": "",
            "padrao": None,
            "confianca": 0.0,
            "img_processada": None
        }

    # 1. Endireita a placa e corta o miolo dos 7 caracteres
    placa_alinhada = endireitar_placa(img_placa)
    img_miolo = isolar_regiao_caracteres(placa_alinhada)
    variacoes = preparar_variacoes_imagem(img_miolo)
    reader = obter_leitor_ocr()

    melhor_resultado = {
        "texto_bruto": "",
        "texto_corrigido": "",
        "padrao": None,
        "confianca": 0.0,
        "img_processada": variacoes[0][1]
    }

    melhor_score = -1.0

    # 2. Executa Multi-Pass com parâmetros calibrados
    for nome_var, img_var in variacoes:
        try:
            resultados = reader.readtext(
                img_var,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                mag_ratio=1.5,       # Calibrado para alta resolução de fontes FE-Schrift
                contrast_ths=0.1,    # Melhora distinção entre 1/2 e W/H
                text_threshold=0.4,
                detail=1,
                paragraph=False
            )
        except Exception:
            continue

        if not resultados:
            continue

        # Ordena caixas da esquerda para a direita
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

        # Aplica extração e validação rigorosa de 7 caracteres
        texto_corrigido, padrao = extrair_melhor_placa_de_texto(texto_concatenado)

        # Pontuação: padrão válido tem bônus alto
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
                "img_processada": img_var
            }

            if padrao is not None and conf_media > 0.70:
                break

    # Garantia final de 7 caracteres
    if len(melhor_resultado["texto_corrigido"]) > 7:
        melhor_resultado["texto_corrigido"] = melhor_resultado["texto_corrigido"][:7]

    return melhor_resultado
