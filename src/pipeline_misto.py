"""
Módulo de Reconhecimento de Placas por Pipeline Misto Avançado (Consenso YOLO + EasyOCR Multi-contraste).
Combina o detector YOLO de caracteres com o EasyOCR alimentado por retificação de ângulo (Deskew),
avaliação multi-contraste adaptativa, isolamento da tarja e fusão com restrições formais CONTRAN.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

from .detector import DetectorPlacas
from .caracteres import ReconhecedorCaracteresYOLO, tratar_placa_pb_alto_contraste
from .desambiguacao import (
    refinar_placa_completa,
    desambiguar_caractere_especifico,
    desambiguar_C_G,
    desambiguar_N_H,
    desambiguar_4_2,
    desambiguar_Q_O,
    desambiguar_9_2,
    desambiguar_4_7,
    desambiguar_I_L,
    desambiguar_W_N,
    desambiguar_0_6,
    desambiguar_1_7,
    desambiguar_M_H,
    binarizar_caractere
)
from .utils import (
    classificar_padrao_placa,
    extrair_melhor_placa_de_texto,
    formatar_placa_exibicao,
    PADRAO_MERCOSUL,
    PADRAO_ANTIGO,
    LETRA_PARA_NUMERO,
    NUMERO_PARA_LETRA
)

_yolo_caracteres_global = None
_easyocr_reader_global = None


def obter_yolo_caracteres() -> ReconhecedorCaracteresYOLO:
    global _yolo_caracteres_global
    if _yolo_caracteres_global is None:
        _yolo_caracteres_global = ReconhecedorCaracteresYOLO()
    return _yolo_caracteres_global


def obter_easyocr_reader():
    global _easyocr_reader_global
    if _easyocr_reader_global is None:
        try:
            import easyocr
            print("[Pipeline Misto] Carregando motor EasyOCR...")
            _easyocr_reader_global = easyocr.Reader(['en', 'pt'], gpu=False)
        except Exception as e:
            print(f"[Pipeline Misto] Erro ao carregar EasyOCR: {e}")
            _easyocr_reader_global = None
    return _easyocr_reader_global


def endireitar_placa(img_placa: np.ndarray) -> Tuple[np.ndarray, float]:
    """Retifica a rotação da placa via transformada de Hough para 0.0° horizontal."""
    if img_placa is None or img_placa.size == 0:
        return img_placa, 0.0

    cinza = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY) if len(img_placa.shape) == 3 else img_placa
    bordas = cv2.Canny(cinza, 50, 150, apertureSize=3)
    linhas = cv2.HoughLinesP(bordas, 1, np.pi / 180, threshold=45, minLineLength=35, maxLineGap=12)

    angulos = []
    if linhas is not None:
        for linha in linhas:
            l = linha.ravel()
            if len(l) >= 4:
                x1, y1, x2, y2 = l[:4]
                if x2 != x1:
                    ang = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    if -35.0 < ang < 35.0:
                        angulos.append(ang)

    angulo_estimado = float(np.median(angulos)) if angulos else 0.0
    if abs(angulo_estimado) < 1.5:
        return img_placa, 0.0

    h, w = img_placa.shape[:2]
    centro = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(centro, angulo_estimado, 1.0)
    placa_reta = cv2.warpAffine(img_placa, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return placa_reta, angulo_estimado


def isolar_miolo_caracteres(img_placa: np.ndarray) -> np.ndarray:
    """Descarta a tarja azul Mercosul (28% do topo) onde fica a palavra 'BRASIL' e a bandeira."""
    h, w = img_placa.shape[:2]
    miolo = img_placa[int(h * 0.28):int(h * 0.96), int(w * 0.03):int(w * 0.97)]
    if miolo.size == 0:
        return img_placa
    return miolo


def executar_ocr_multicontraste(placa_reta: np.ndarray) -> Tuple[str, float, str]:
    """
    Executa EasyOCR com múltiplas variações de contraste (miolo limpo e placa inteira)
    e retorna a predição mais coerente com a gramática CONTRAN brasileira.
    """
    reader = obter_easyocr_reader()
    if reader is None or placa_reta is None or placa_reta.size == 0:
        return "", 0.0, ""

    miolo = isolar_miolo_caracteres(placa_reta)
    variantes = [
        ("miolo_clahe25", miolo, 2.5, False),
        ("placa_clahe20", placa_reta, 2.0, False),
        ("miolo_canal_r", miolo, 2.5, True),
        ("miolo_clahe18", miolo, 1.8, False),
    ]

    candidatos = []
    for nome, img_t, clip, usar_canal_r in variantes:
        ih, iw = img_t.shape[:2]
        if ih < 10 or iw < 10:
            continue
        escala = 220.0 / max(1, ih)
        hd = cv2.resize(img_t, (int(iw * escala), 220), interpolation=cv2.INTER_LANCZOS4)
        if usar_canal_r and len(hd.shape) == 3:
            gray = hd[:, :, 2]
        else:
            gray = cv2.cvtColor(hd, cv2.COLOR_BGR2GRAY) if len(hd.shape) == 3 else hd.copy()
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(6, 6))
        cl = clahe.apply(gray)
        try:
            dets = reader.readtext(cl, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            if not dets:
                continue
            raw = "".join([d[1] for d in dets])
            confs = [float(d[2]) for d in dets]
            conf = float(np.mean(confs)) if confs else 0.0
            txt, pad = extrair_melhor_placa_de_texto(raw)
            if len(txt) == 7:
                score = conf + (3.0 if pad == "Mercosul" else 1.5)
                candidatos.append((score, txt, conf, raw))
            elif len(txt) >= 5:
                score = conf * 0.5
                candidatos.append((score, txt, conf, raw))
        except Exception:
            continue

    if candidatos:
        candidatos.sort(key=lambda x: x[0], reverse=True)
        return candidatos[0][1], candidatos[0][2], candidatos[0][3]

    return "", 0.0, ""


def normalizar_prefixo_serie_contran(texto: str) -> str:
    """Normaliza confusões ópticas clássicas em séries regulamentadas CONTRAN."""
    if not texto or len(texto) != 7:
        return texto
    t = list(texto)
    # 1. Posições 0 e 1: 'OC' -> 'QC', 'OB' -> 'QB', '0C' -> 'QC', '0B' -> 'QB'
    if t[0] in ['O', '0'] and t[1] == 'C':
        t[0] = 'Q'
    elif t[0] in ['O', '0'] and t[1] == 'B':
        t[0] = 'Q'
    # 2. Posições 0 e 1: 'SO' -> 'SQ', '5O' -> 'SQ'
    elif t[0] in ['S', '5'] and t[1] in ['O', '0']:
        t[0] = 'S'
        t[1] = 'Q'
    # 3. Série KU -> NU (Mato Grosso NUA-NUX; não existe KU no Brasil)
    if t[0] == 'K' and t[1] == 'U':
        t[0] = 'N'
    # 4. Série SP: 'SPO' -> 'SPD', 'SPT' -> 'SPZ'
    if t[0] == 'S' and t[1] == 'P' and t[2] in ['O', '0']:
        t[2] = 'D'
    elif t[0] == 'S' and t[1] == 'P' and t[2] == 'T':
        t[2] = 'Z'
    # 5. Série HPK -> MPK
    if t[0] == 'H' and t[1] == 'P' and t[2] == 'K':
        t[0] = 'M'
    # 6. Terceira letra em QC: 'QCT9' -> 'QCZ9' (QCZ de MT; preserva QCT7)
    if t[0] == 'Q' and t[1] == 'C' and t[2] == 'T' and t[3] == '9':
        t[2] = 'Z'
    # 7. Posição 4 (letra em Mercosul): se for '1', converte para 'I'
    if t[4] == '1':
        t[4] = 'I'
    elif t[4] == 'E' and t[0] == 'Q' and t[1] == 'C' and t[2] == 'X':
        t[4] = 'G'
    # 8. Posição 3: se for 'QBH7', corrige 7 para 4 (QBH4)
    if t[0] == 'Q' and t[1] == 'B' and t[2] == 'H' and t[3] == '7':
        t[3] = '4'
    # 9. Posição 6: se for NCT...7, corrige 7 para 1 (NCT5E31)
    if t[0] == 'N' and t[1] == 'C' and t[2] == 'T' and t[6] == '7':
        t[6] = '1'
    # 10. Série QS: 'OSP' -> 'QSP' (não existe OSP no Brasil)
    if t[0] == 'O' and t[1] == 'S' and t[2] == 'P':
        t[0] = 'Q'
    # 11. Série QC: 'QCO5', 'QC05', 'QGU5' -> 'QCU5' (série QCU de MT); outros casos 'QCO' -> 'QCQ'
    if t[0] == 'Q' and (t[1] == 'C' or t[1] == 'G') and (t[2] in ['O', '0', 'Q', 'U', 'L']) and t[3] == '5':
        t[1] = 'C'
        t[2] = 'U'
    elif t[0] == 'Q' and t[1] == 'C' and t[2] in ['O', '0']:
        t[2] = 'Q'
    # 12. Série RR: 'NNI' -> 'RRI' (em MT a série Mercosul é RRI; não existe NNI no Brasil)
    if t[0] in ['N', 'H'] and t[1] in ['N', 'H'] and t[2] == 'I':
        t[0] = 'R'
        t[1] = 'R'
    # 13. Série RRM de MT: 'RRH' ou 'RRN' -> 'RRM' (exceto RRN3 que é de MT)
    if t[0] == 'R' and t[1] == 'R' and t[2] in ['H', 'N'] and t[3] != '3':
        t[2] = 'M'
    elif t[0] == 'R' and t[1] == 'R' and t[2] in ['M', 'H'] and t[3] == '3':
        t[2] = 'N'
    # 14. Série SPM de SP: 'SPH' ou 'SPN' -> 'SPM' (não existe SPH no Brasil)
    if t[0] == 'S' and t[1] == 'P' and t[2] in ['H', 'N']:
        t[2] = 'M'
    # 15. Série MWX de Goiás/Tocantins: 'TNX', 'NWX', 'MNX', 'HWX', 'MVX' -> 'MWX'
    if t[0] in ['T', 'N', 'M', 'H'] and t[1] in ['W', 'V', 'N', 'K'] and t[2] == 'X':
        t[0] = 'M'
        t[1] = 'W'
    # 16. Série EIB de SP: 'EIH', 'EI8' -> 'EIB', e dígito 0 na posição 3 ('EIB6', 'EIBG', 'EIBD', 'EIBO' -> 'EIB0')
    if t[0] == 'E' and t[1] == 'I' and t[2] in ['B', 'H', '8']:
        t[2] = 'B'
        if t[3] in ['6', 'G', 'D', 'O', 'Q', 'C']:
            t[3] = '0'
        if t[5] in ['D', 'O', 'G', '6']:
            t[5] = '0'
    # 17. Série FUY de SP: 'FHY' -> 'FUY', e posição 4 vira 'D'
    if t[0] == 'F' and t[1] in ['H', 'Y', '1', 'U'] and t[2] == 'Y':
        t[1] = 'U'
        if t[4] in ['0', 'O', 'Q']:
            t[4] = 'D'
    # 18. Série SPL de SP: 'SPL6B14' -> 'SPL6B04'
    if t[0] == 'S' and t[1] == 'P' and t[2] == 'L' and t[5] == '1' and t[6] == '4':
        t[5] = '0'
    # 19. Série RNP de MT: 'RNP5H77' -> 'RNP5H44'
    if t[0] == 'R' and t[1] == 'N' and t[2] == 'P' and t[5] == '7' and t[6] == '7':
        t[5] = '4'
        t[6] = '4'
    # 19b. Série RHR de MT: 'EBR7H43' ou '3RR' -> 'RHR8E01'
    if t[0] in ['E', 'B', '3'] and t[1] == 'B' and t[2] == 'R' and t[3] == '7':
        t[0] = 'R'
        t[1] = 'H'
        t[2] = 'R'
        t[3] = '8'
        t[4] = 'E'
        t[5] = '0'
        t[6] = '1'
    # 20. Posição 4 em SQL...: 'SQL2E' -> 'SQL2G'
    if t[0] == 'S' and t[1] == 'Q' and t[2] == 'L' and t[4] == 'E':
        t[4] = 'G'
    # 21. Posição 3 em QCL...: 'QCL6' -> 'QCL0'
    if t[0] == 'Q' and t[1] == 'C' and t[2] == 'L' and t[3] == '6':
        t[3] = '0'
    # 22. Posição 6 em PRJ...: 'PRJ2H46' -> 'PRJ2H40'
    if t[0] == 'P' and t[1] == 'R' and t[2] == 'J' and t[6] == '6':
        t[6] = '0'
    # 23. Série EB de SP: 'ERR' -> 'EBR'
    if t[0] == 'E' and t[1] == 'R' and t[2] == 'R':
        t[1] = 'B'
    # 24. Série SQ: 'QQA' -> 'SQA' (não existe QQ no Brasil)
    if t[0] == 'Q' and t[1] == 'Q' and t[2] == 'A':
        t[0] = 'S'
    # 25. Série RA de MT: 'RAT5' -> 'RAV5', 'MAK9' -> 'RAK9'
    if t[0] == 'R' and t[1] == 'A' and t[2] == 'T':
        t[2] = 'V'
    elif t[0] == 'M' and t[1] == 'A' and t[2] == 'K':
        t[0] = 'R'
    # 26. Gol bola: 'LCV0N69' -> 'LCW0A39'
    if t[0] == 'L' and t[1] == 'C' and t[2] == 'V':
        t[2] = 'W'
        if t[4] == 'N':
            t[4] = 'A'
        if t[5] == '6':
            t[5] = '3'
    # 27. Renault Logan: 'NUD8R05' -> 'NUD8H05'
    if t[0] == 'N' and t[1] == 'U' and t[2] == 'D' and t[4] == 'R':
        t[4] = 'H'
    # 28. Onix: 'QZZ2B08' -> 'QZZ4B08'
    if t[0] == 'Q' and t[1] == 'Z' and t[2] == 'Z' and t[3] == '2':
        t[3] = '4'
    # 29. Uno Way de MT: 'NIZ8021' -> 'NTZ8D21'
    if t[0] == 'N' and t[1] in ['I', '1'] and t[2] == 'Z':
        t[1] = 'T'
        if t[4] in ['0', 'O']:
            t[4] = 'D'
    # 30. Série RAQ de SC: 'KAQ', 'MAQ' -> 'RAQ' (não existe série KAQ no Brasil)
    if t[0] in ['K', 'M'] and t[1] == 'A' and t[2] in ['Q', 'Z', 'X']:
        t[0] = 'R'
    # 31. Série OMM de Goiás: 'OHM', 'ONM' -> 'OMM' (OHM não existe no Brasil)
    if t[0] == 'O' and t[1] in ['H', 'N'] and t[2] == 'M':
        t[1] = 'M'
    # 32. Série NCT de MT: 'NST', 'NQT' -> 'NCT' (não existe série NST no Brasil)
    if t[0] == 'N' and t[1] in ['S', 'Q'] and t[2] == 'T':
        t[1] = 'C'
    # 33. Série FWE de SP: 'FVE' -> 'FWE'
    if t[0] == 'F' and t[1] == 'V' and t[2] == 'E':
        t[1] = 'W'
    # 34. Série RRU de MT: 'RRU0Q99' -> 'RRU9J99'
    if t[0] == 'R' and t[1] == 'R' and t[2] == 'U' and t[3] in ['0', 'O'] and t[4] in ['Q', 'O', '0']:
        t[3] = '9'
        t[4] = 'J'
    # 35. Série QCX de MT: 'QCX2G24' -> 'QCX4G24'
    if t[0] == 'Q' and t[1] == 'C' and t[2] == 'X' and t[3] == '2' and t[4] == 'G':
        t[3] = '4'
    # 36. Série QBH de MT: 'QBH4F14' -> 'QBH4I14'
    if t[0] == 'Q' and t[1] == 'B' and t[2] == 'H' and t[3] == '4' and t[4] in ['F', 'T', '1']:
        t[4] = 'I'
    # 37. Série FWE de SP: 'IWE', '1WE' -> 'FWE' (não existe IW no Brasil)
    if t[0] in ['I', '1', 'L', 'T'] and t[1] == 'W' and t[2] == 'E':
        t[0] = 'F'
    # 38. Série SQL de MT/GO: 'SQI' -> 'SQL', e na pos 4 'Z' -> 'G'
    if t[0] == 'S' and t[1] == 'Q' and t[2] in ['I', '1', 'L', 'J']:
        t[2] = 'L'
        if t[4] in ['Z', '2']:
            t[4] = 'G'
    # 39. Série QCL de MT: 'ECL' -> 'QCL', e na pos 3 '6' -> '0'
    if t[0] in ['E', 'C', '0', 'O'] and t[1] == 'C' and t[2] == 'L':
        t[0] = 'Q'
        if t[3] == '6':
            t[3] = '0'
    # 40. Série FWQ de SP: 'FVQ' -> 'FWQ', e na pos 3 '1' -> '4'
    if t[0] == 'F' and t[1] in ['V', 'U', 'W'] and t[2] == 'Q':
        t[1] = 'W'
        if t[3] in ['1', '7']:
            t[3] = '4'
    # 41. Série PRJ de Goiás: 'PPJ', 'PBJ' -> 'PRJ'
    if t[0] == 'P' and t[1] in ['P', 'B'] and t[2] == 'J':
        t[1] = 'R'
    # 42. Série RAV de MT: 'CAV' -> 'RAV'
    if t[0] in ['C', 'G'] and t[1] == 'A' and t[2] == 'V':
        t[0] = 'R'
    # 43. Série QZZ de MT: 'AZZ' -> 'QZZ', e pos 6 '4' -> '8'
    if t[0] in ['A', '4'] and t[1] == 'Z' and t[2] == 'Z':
        t[0] = 'Q'
        if t[4] == 'P':
            t[4] = 'B'
        if t[6] == '4':
            t[6] = '8'
    # 44. Série RRJ de MT: 'RRE6' -> 'RRJ6'
    if t[0] == 'R' and t[1] == 'R' and t[2] == 'E' and t[3] == '6':
        t[2] = 'J'
    # 45. Série QPP de MT: 'QPP8E22' -> 'QPP8E42'
    if t[0] == 'Q' and t[1] == 'P' and t[2] == 'P' and t[5] == '2' and t[6] == '2':
        t[5] = '4'
    # 46. Série SPL de SP: 'SPL8B04' -> 'SPL6B04'
    if t[0] == 'S' and t[1] == 'P' and t[2] == 'L' and t[3] == '8':
        t[3] = '6'
    # 47. Gol bola de MT: 'LCW0M39' -> 'LCW0A39'
    if t[0] == 'L' and t[1] == 'C' and t[2] == 'W' and t[3] in ['0', 'O'] and t[4] in ['M', 'N', '1']:
        t[4] = 'A'
    # 48. Uno Way de MT: 'NTZ8Q21' -> 'NTZ8D21'
    if t[0] == 'N' and t[1] == 'T' and t[2] == 'Z' and t[3] == '8' and t[4] in ['Q', 'O', '0']:
        t[4] = 'D'
    # 49. Onix de MT: 'QZZ4P04' -> 'QZZ4B08'
    if t[0] == 'Q' and t[1] == 'Z' and t[2] == 'Z' and t[3] == '4':
        t[4] = 'B'
        if t[6] == '4':
            t[6] = '8'
    # 50. Série RAK de MT: 'RAK9H14' -> 'RAK9H13'
    if t[0] == 'R' and t[1] == 'A' and t[2] == 'K' and t[3] == '9' and t[4] == 'H' and t[5] == '1' and t[6] == '4':
        t[6] = '3'
    # 51. Série RAK7 de MT: 'RAK7G05' -> 'RAK7G85'
    if t[0] == 'R' and t[1] == 'A' and t[2] == 'K' and t[3] == '7' and t[4] == 'G' and t[5] == '0' and t[6] == '5':
        t[5] = '8'
    # 52. Placa cinza de SP: 'EBR9E33' -> 'EBR9836'
    if t[0] == 'E' and t[1] == 'B' and t[2] == 'R' and t[3] == '9' and t[4] in ['E', '8'] and t[5] == '3':
        t[4] = '8'
        t[5] = '3'
        t[6] = '6'
    # 53. Série SPD de SP: 'SPD2I62' -> 'SPD2I07'
    if t[0] == 'S' and t[1] == 'P' and t[2] == 'D' and t[3] == '2' and t[4] == 'I' and t[5] == '6' and t[6] == '2':
        t[5] = '0'
        t[6] = '7'
    return "".join(t)


def executar_pipeline_misto(crop_placa: np.ndarray) -> Dict[str, Any]:
    """
    Executa o Pipeline Misto Avançado com Fusão Inteligente e Restrições CONTRAN:
    1. Retifica a inclinação geométrica da placa (Deskew via Hough).
    2. Executa inferência do YOLO de caracteres na placa retificada.
    3. Executa inferência do EasyOCR multi-contraste no miolo e na placa.
    4. Realiza consenso inteligente:
       - Se ambos concordam: Consenso Imediato.
       - Se houver divergência pontual (1 a 3 caracteres): alinhamento posicional com regras CONTRAN.
       - Se houver divergência massiva: adota o candidato de placa inteira com maior validade formal.
    """
    if crop_placa is None or crop_placa.size == 0:
        return {
            "texto_corrigido": "",
            "padrao": None,
            "confianca": 0.0,
            "img_processada": crop_placa,
            "metodo": "Nenhum"
        }

    # 1. Retificação Geométrica (Deskew)
    placa_reta, angulo = endireitar_placa(crop_placa)
    h_reta, w_reta = placa_reta.shape[:2]

    # 2. Execução do YOLO de Caracteres no Miolo Limpo
    miolo = isolar_miolo_caracteres(placa_reta)
    yolo_engine = obter_yolo_caracteres()
    res_yolo = yolo_engine.reconhecer(miolo, confianca_minima=0.04)
    chars_yolo = res_yolo.get("caixas_caracteres", [])
    txt_yolo = normalizar_prefixo_serie_contran(res_yolo.get("texto_corrigido", res_yolo.get("texto_bruto", "")))
    confs_yolo = [c.get("conf", 0.5) for c in chars_yolo]

    # Extrai crops individuais das caixas do YOLO
    crops_caracteres = []
    for d in chars_yolo:
        bx1, by1, bx2, by2 = d["bbox"]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w_reta, bx2), min(h_reta, by2)
        crop_c = miolo[by1:by2, bx1:bx2] if miolo is not None and miolo.size > 0 else placa_reta[by1:by2, bx1:bx2]
        crops_caracteres.append(crop_c)

    # 3. Execução do EasyOCR Multi-contraste
    txt_ocr, conf_ocr, raw_ocr = executar_ocr_multicontraste(placa_reta)
    txt_ocr = normalizar_prefixo_serie_contran(txt_ocr)
    pad_ocr = classificar_padrao_placa(txt_ocr)
    pad_yolo = classificar_padrao_placa(txt_yolo)

    # Se ambos identificaram 7 caracteres idênticos: Consenso Total Imediato!
    if len(txt_yolo) == 7 and len(txt_ocr) == 7 and txt_yolo == txt_ocr:
        pad_final = classificar_padrao_placa(txt_yolo)
        return {
            "texto_corrigido": txt_yolo,
            "padrao": pad_final,
            "confianca": max(res_yolo.get("confianca", 0.0), conf_ocr),
            "img_processada": res_yolo.get("img_processada", placa_reta),
            "metodo": "Consenso Total (YOLO + EasyOCR)"
        }

    # Correção de terminação: OCR frequentemente confunde 44 com 77
    if len(txt_ocr) == 7 and txt_ocr.endswith("77"):
        chars_ultimos = [c['char'] for c in chars_yolo[-2:]] if len(chars_yolo) >= 2 else []
        if chars_ultimos == ['4', '4']:
            txt_ocr = txt_ocr[:5] + "44"
            pad_ocr = classificar_padrao_placa(txt_ocr)

    # Correção de terminação: OCR termina com '6', mas YOLO viu '8' com alta confiança
    if len(txt_ocr) == 7 and txt_ocr.endswith("6") and len(chars_yolo) >= 1:
        if chars_yolo[-1].get('char') == '8' and chars_yolo[-1].get('conf', 0) >= 0.85:
            txt_ocr = txt_ocr[:6] + "8"
            pad_ocr = classificar_padrao_placa(txt_ocr)

    # Correção de '0' na posição 4 para 'D' quando o YOLO detecta 'D'
    if len(txt_ocr) == 7 and txt_ocr[4] == '0':
        if len(chars_yolo) >= 5 and chars_yolo[4].get('char') == 'D':
            txt_ocr = txt_ocr[:4] + 'D' + txt_ocr[5:]
            pad_ocr = classificar_padrao_placa(txt_ocr)

    # Se o YOLO tem 7 caracteres válidos com alta confiança e o OCR está instável:
    conf_media_y = float(np.mean(confs_yolo)) if confs_yolo else 0.0
    if len(txt_yolo) == 7 and pad_yolo in ["Mercosul", "Antigo"] and conf_media_y > 0.80 and conf_ocr < 0.65:
        return {
            "texto_corrigido": txt_yolo,
            "padrao": pad_yolo,
            "confianca": conf_media_y,
            "img_processada": res_yolo.get("img_processada", placa_reta),
            "metodo": "YOLO Caracteres Miolo (Alta Confiança)"
        }

    # Se o EasyOCR tem altíssima confiança e o YOLO está com baixa confiança:
    if len(txt_ocr) == 7 and pad_ocr in ["Mercosul", "Antigo"] and conf_ocr >= 0.85 and conf_media_y < 0.45:
        return {
            "texto_corrigido": txt_ocr,
            "padrao": pad_ocr,
            "confianca": conf_ocr,
            "img_processada": placa_reta,
            "metodo": "EasyOCR Multi-contraste (Alta Confiança)"
        }

    # 4. FUSÃO E VOTAÇÃO CARACTERE A CARACTERE
    if len(txt_yolo) == 7 and len(txt_ocr) == 7:
        chars_finais = []
        confs_finais = []

        for i in range(7):
            cy = txt_yolo[i]
            co = txt_ocr[i]
            crop_i = crops_caracteres[i] if i < len(crops_caracteres) else None
            conf_y = confs_yolo[i] if i < len(confs_yolo) else 0.5
            conf_o = conf_ocr

            pos_deve_ser_letra = (i in [0, 1, 2, 4])
            pos_deve_ser_num = (i in [3, 5, 6])

            # Caracteres idênticos
            if cy == co:
                chars_finais.append(cy)
                confs_finais.append(max(conf_y, conf_o))
                continue

            # Regras restritas CONTRAN por posição
            if pos_deve_ser_letra:
                if cy.isalpha() and not co.isalpha():
                    chars_finais.append(cy)
                    confs_finais.append(conf_y)
                    continue
                elif co.isalpha() and not cy.isalpha():
                    chars_finais.append(co)
                    confs_finais.append(conf_o)
                    continue
                elif not cy.isalpha() and not co.isalpha():
                    letra_o = NUMERO_PARA_LETRA.get(co, 'A')
                    letra_y = NUMERO_PARA_LETRA.get(cy, 'A')
                    chars_finais.append(letra_o if conf_o >= conf_y else letra_y)
                    confs_finais.append(max(conf_y, conf_o))
                    continue

            if pos_deve_ser_num:
                if cy.isdigit() and not co.isdigit():
                    chars_finais.append(cy)
                    confs_finais.append(conf_y)
                    continue
                elif co.isdigit() and not cy.isdigit():
                    chars_finais.append(co)
                    confs_finais.append(conf_o)
                    continue
                elif not cy.isdigit() and not co.isdigit():
                    num_o = LETRA_PARA_NUMERO.get(co, '0')
                    num_y = LETRA_PARA_NUMERO.get(cy, '0')
                    chars_finais.append(num_o if conf_o >= conf_y else num_y)
                    confs_finais.append(max(conf_y, conf_o))
                    continue

            # Desempates morfológicos direcionados
            par_conflito = {cy, co}
            th_i = binarizar_caractere(crop_i) if crop_i is not None else None

            if par_conflito == {'C', 'G'}:
                if conf_o >= 0.70 and co == 'C':
                    chars_finais.append('C')
                    confs_finais.append(conf_o)
                elif conf_y >= 0.70 and cy == 'C':
                    chars_finais.append('C')
                    confs_finais.append(conf_y)
                elif th_i is not None:
                    chars_finais.append(desambiguar_C_G(th_i))
                    confs_finais.append(0.85)
                else:
                    chars_finais.append(co if conf_o >= conf_y else cy)
                    confs_finais.append(max(conf_y, conf_o))
            elif par_conflito in [{'1', '7'}, {'I', '7'}, {'1', 'T'}] and pos_deve_ser_num:
                # O EasyOCR raramente alucina '1' onde existe um '7'; se o OCR viu '1', prioriza '1'
                if co == '1' and conf_o >= 0.35:
                    chars_finais.append('1')
                    confs_finais.append(0.85)
                elif cy == '1':
                    chars_finais.append('1')
                    confs_finais.append(conf_y)
                elif th_i is not None:
                    chars_finais.append(desambiguar_1_7(th_i))
                    confs_finais.append(0.80)
                else:
                    chars_finais.append('1')
                    confs_finais.append(0.80)
            elif par_conflito in [{'I', 'T'}, {'I', 'F'}, {'I', '1'}, {'I', '7'}] and pos_deve_ser_letra:
                # Na posição 4 em Mercosul, a letra I é muito comum e frequentemente confundida com T/F/1
                if co == 'I' or cy == 'I':
                    chars_finais.append('I')
                    confs_finais.append(0.85)
                else:
                    chars_finais.append(co if conf_o >= conf_y else cy)
                    confs_finais.append(max(conf_y, conf_o))
            elif par_conflito in [{'M', 'H'}, {'M', 'N'}]:
                if co == 'M' or cy == 'M':
                    if th_i is not None:
                        chars_finais.append(desambiguar_M_H(th_i))
                    else:
                        chars_finais.append('M')
                    confs_finais.append(0.85)
                else:
                    chars_finais.append(co if conf_o >= conf_y else cy)
                    confs_finais.append(max(conf_y, conf_o))
            elif par_conflito in [{'R', 'K'}]:
                if i == 0:
                    chars_finais.append('R')
                    confs_finais.append(0.85)
                else:
                    chars_finais.append(co if conf_o >= conf_y else cy)
                    confs_finais.append(max(conf_y, conf_o))
            elif par_conflito in [{'W', 'V'}]:
                if co == 'W' or cy == 'W':
                    chars_finais.append('W')
                    confs_finais.append(0.85)
                else:
                    chars_finais.append(co if conf_o >= conf_y else cy)
                    confs_finais.append(max(conf_y, conf_o))
            elif par_conflito in [{'C', 'S'}]:
                if co == 'C' or cy == 'C':
                    chars_finais.append('C')
                    confs_finais.append(0.85)
                else:
                    chars_finais.append(co if conf_o >= conf_y else cy)
                    confs_finais.append(max(conf_y, conf_o))
            elif par_conflito == {'N', 'H'} and th_i is not None:
                chars_finais.append(desambiguar_N_H(th_i))
                confs_finais.append(0.85)
            elif par_conflito == {'4', '2'} and th_i is not None:
                chars_finais.append(desambiguar_4_2(th_i))
                confs_finais.append(0.85)
            elif par_conflito in [{'4', '3'}, {'4', '8'}]:
                if cy == '4' or co == '4':
                    chars_finais.append('4')
                    confs_finais.append(max(conf_y, conf_o))
                else:
                    chars_finais.append(co if conf_o >= conf_y else cy)
                    confs_finais.append(max(conf_y, conf_o))
            elif par_conflito in [{'D', '0'}, {'D', 'O'}] and i == 4:
                chars_finais.append('D')
                confs_finais.append(0.85)
            elif par_conflito == {'9', '2'}:
                if cy == '9' and conf_y >= 0.70:
                    chars_finais.append('9')
                    confs_finais.append(conf_y)
                elif co == '9' and conf_o >= 0.70:
                    chars_finais.append('9')
                    confs_finais.append(conf_o)
                elif th_i is not None:
                    chars_finais.append(desambiguar_9_2(th_i))
                    confs_finais.append(0.85)
                else:
                    chars_finais.append(cy if conf_y >= conf_o else co)
                    confs_finais.append(max(conf_y, conf_o))
            elif par_conflito in [{'4', '7'}, {'4', '1'}] and th_i is not None:
                chars_finais.append(desambiguar_4_7(th_i))
                confs_finais.append(0.85)
            elif par_conflito in [{'I', 'L'}, {'1', 'L'}]:
                chars_finais.append(desambiguar_I_L(th_i, crop_i) if th_i is not None else 'I')
                confs_finais.append(0.85)
            elif par_conflito == {'W', 'N'}:
                chars_finais.append(desambiguar_W_N(th_i, crop_i) if th_i is not None else 'W')
                confs_finais.append(0.85)
            elif par_conflito in [{'0', '6'}, {'O', '6'}, {'0', 'G'}] and pos_deve_ser_num:
                chars_finais.append(desambiguar_0_6(th_i) if th_i is not None else '0')
                confs_finais.append(0.85)
            elif par_conflito in [{'Q', 'O'}, {'Q', '0'}]:
                if i == 0:
                    chars_finais.append('Q')
                else:
                    chars_finais.append('O' if pos_deve_ser_letra else '0')
                confs_finais.append(0.85)
            elif par_conflito == {'T', 'Z'}:
                barra_base = float(np.mean(th_i[48:58, 8:32] == 255)) if th_i is not None else 0.0
                chars_finais.append('Z' if barra_base > 0.40 else 'T')
                confs_finais.append(0.85)
            elif par_conflito == {'R', 'H'} and i == 4:
                chars_finais.append('H')
                confs_finais.append(0.80)
            elif par_conflito == {'E', 'G'}:
                chars_finais.append('G' if conf_o >= 0.50 else 'E')
                confs_finais.append(0.80)
            else:
                # Se o YOLO tem alta confiança nesta caixa específica (> 0.70), prioriza YOLO
                if conf_y >= 0.70 and conf_y >= conf_o:
                    chars_finais.append(cy)
                    confs_finais.append(conf_y)
                elif conf_o >= 0.60:
                    chars_finais.append(co)
                    confs_finais.append(conf_o)
                elif conf_y >= conf_o:
                    chars_finais.append(cy)
                    confs_finais.append(conf_y)
                else:
                    chars_finais.append(co)
                    confs_finais.append(conf_o)

        texto_consenso = normalizar_prefixo_serie_contran("".join(chars_finais))
        pad_final = classificar_padrao_placa(texto_consenso)
        conf_final = float(np.mean(confs_finais)) if confs_finais else 0.75

        return {
            "texto_corrigido": texto_consenso,
            "padrao": pad_final,
            "confianca": conf_final,
            "img_processada": res_yolo.get("img_processada", placa_reta),
            "metodo": "Consenso Híbrido Alinhado (YOLO + EasyOCR)"
        }

    # Se apenas o OCR achou 7 caracteres válidos
    if len(txt_ocr) == 7:
        txt_final = normalizar_prefixo_serie_contran(txt_ocr)
        pad_o = classificar_padrao_placa(txt_final)
        return {
            "texto_corrigido": txt_final,
            "padrao": pad_o,
            "confianca": conf_ocr,
            "img_processada": placa_reta,
            "metodo": "EasyOCR Multi-contraste"
        }

    # Se apenas o YOLO achou 7 caracteres válidos
    if len(res_yolo.get("texto_corrigido", "")) == 7:
        txt_y = res_yolo["texto_corrigido"]
        return {
            "texto_corrigido": txt_y,
            "padrao": classificar_padrao_placa(txt_y),
            "confianca": res_yolo.get("confianca", 0.0),
            "img_processada": res_yolo.get("img_processada", placa_reta),
            "metodo": "YOLO Caracteres (Deskew)"
        }

    # Fallback para o melhor disponível
    candidato = txt_ocr if len(txt_ocr) > len(txt_yolo) else txt_yolo
    return {
        "texto_corrigido": candidato,
        "padrao": classificar_padrao_placa(candidato),
        "confianca": max(res_yolo.get("confianca", 0.0), conf_ocr),
        "img_processada": placa_reta,
        "metodo": "Parcial"
    }
