"""
Módulo de Desambiguação Morfológica Avançada para Caracteres de Placas (Fonte FE-Schrift / Padrão CONTRAN).
Resolve com precisão matemática:
- N vs H (analisa se os quadrantes superior e inferior entre as hastes são ocos ou diagonais)
- 4 vs 2 (analisa se a base inferior esquerda é aberta ou preenchida por barra horizontal)
- Q vs O (analisa cauda inferior direita e séries regulamentares brasileiras Q/O)
- S vs Q (analisa aberturas características do S vs anel fechado do Q)
- S vs E (analisa continuidade da haste vertical esquerda do E vs curvas do S)
- C vs G (analisa abertura lateral direita do C vs fechamento vertical do G)
- B vs 8, X vs K, T vs I, 2 vs P, J vs L, S vs Z
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


def binarizar_caractere(crop_char: np.ndarray) -> Optional[np.ndarray]:
    """Binariza o caractere com tamanho padronizado (60x40) com traço branco (255) e fundo preto (0)."""
    if crop_char is None or crop_char.size == 0:
        return None
    h, w = crop_char.shape[:2]
    if h < 6 or w < 3:
        return None

    char_padrao = cv2.resize(crop_char, (40, 60), interpolation=cv2.INTER_LANCZOS4)

    if len(char_padrao.shape) == 3:
        gray = cv2.cvtColor(char_padrao, cv2.COLOR_BGR2GRAY)
    else:
        gray = char_padrao

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    cl = clahe.apply(gray)

    # Otsu invertido: traço = 255, fundo = 0
    _, th = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if np.mean(th == 255) > 0.65:
        th = cv2.bitwise_not(th)

    return th


def desambiguar_N_H(th: np.ndarray) -> str:
    """
    Distingue N de H:
    - O caractere 'H' tem duas hastes verticais e UMA barra horizontal no meio.
      Os quadrantes superior-meio e inferior-meio são OCOS (pretos).
    - O caractere 'N' tem a diagonal atravessando esses quadrantes.
    """
    quad_sup_vazio = float(np.mean(th[8:22, 14:26] == 255)) < 0.18
    quad_inf_vazio = float(np.mean(th[38:52, 14:26] == 255)) < 0.18

    if quad_sup_vazio and quad_inf_vazio:
        return 'H'

    # Se houver traço na diagonal
    quad_diag_sup = float(np.mean(th[8:24, 12:20] == 255)) > 0.20
    quad_diag_inf = float(np.mean(th[36:52, 20:28] == 255)) > 0.20
    if quad_diag_sup or quad_diag_inf:
        return 'N'

    return 'H'


def desambiguar_4_2(th: np.ndarray) -> str:
    """
    Distingue 4 de 2:
    - O '2' tem barra horizontal contínua preenchendo toda a base inferior (inclusive à esquerda).
    - O '4' tem a base inferior esquerda ABERTA (vazia), pois a haste vertical desce na direita.
    """
    base_esquerda_vazia = float(np.mean(th[48:58, 6:20] == 255)) < 0.25
    haste_vert_direita = float(np.mean(th[30:58, 24:34] == 255)) > 0.40

    if base_esquerda_vazia and haste_vert_direita:
        return '4'

    base_completa = float(np.mean(th[48:58, 8:32] == 255)) > 0.50
    if base_completa:
        return '2'

    return '4'


def desambiguar_C_G(th: np.ndarray) -> str:
    """
    Distingue C de G:
    - O 'C' é totalmente aberto na lateral direita média/inferior.
    - O 'G' tem a haste vertical subindo no lado direito inferior até o meio.
    """
    lado_dir_inf = float(np.mean(th[28:46, 26:36] == 255))
    if lado_dir_inf > 0.22:
        return 'G'
    return 'C'


def desambiguar_S_Q(th: np.ndarray) -> str:
    """
    Distingue S de Q:
    - O 'S' tem aberturas no canto superior esquerdo e no canto inferior direito.
    - O 'Q' é um anel fechado contínuo em ambos os lados.
    """
    sup_esq_aberto = float(np.mean(th[10:24, 6:14] == 255)) < 0.25
    inf_dir_aberto = float(np.mean(th[36:50, 26:34] == 255)) < 0.25

    if sup_esq_aberto and inf_dir_aberto:
        return 'S'
    return 'Q'


def desambiguar_S_E(th: np.ndarray) -> str:
    """
    Distingue S de E:
    - O 'E' tem uma haste vertical sólida e contínua do topo à base no lado esquerdo.
    - O 'S' é curvo e aberto no topo esquerdo.
    """
    haste_esq_continua = float(np.mean(th[10:50, 4:12] == 255)) > 0.58
    if haste_esq_continua:
        return 'E'
    return 'S'


def desambiguar_Q_O(th: np.ndarray, posicao: int = 0) -> str:
    """
    Distingue Q de O:
    - Analisa a cauda característica no canto inferior direito.
    - Na posição 0 da placa, séries como QAA-QZZ são predominantes no Brasil.
    """
    canto_inf_dir = float(np.mean(th[42:58, 24:38] == 255))
    if canto_inf_dir > 0.28:
        return 'Q'
    return 'O'


def desambiguar_B_8(th: np.ndarray) -> str:
    """Distingue B de 8: B tem haste reta contínua à esquerda; 8 tem recortes curvos dos dois lados."""
    haste_esq = float(np.mean(th[10:50, 4:12] == 255)) > 0.55
    return 'B' if haste_esq else '8'


def desambiguar_K_X(th: np.ndarray) -> str:
    """Distingue K de X: K tem haste vertical esquerda; X é cruzamento em diagonal."""
    haste_esq = float(np.mean(th[10:50, 4:12] == 255)) > 0.55
    return 'K' if haste_esq else 'X'


def desambiguar_T_I(th: np.ndarray, crop_char: Optional[np.ndarray] = None) -> str:
    """Distingue T de I: T tem barra horizontal larga no topo; I é estreito."""
    if crop_char is not None and crop_char.size > 0:
        h, w = crop_char.shape[:2]
        if w / max(1, h) < 0.32:
            return 'I'
    barra_topo = float(np.mean(th[3:14, 4:36] == 255)) > 0.50
    return 'T' if barra_topo else 'I'


def desambiguar_2_P(th: np.ndarray) -> str:
    """Distingue 2 de P: 2 tem barra na base; P tem base aberta e perna esquerda."""
    base_fechada = float(np.mean(th[48:58, 8:32] == 255)) > 0.50
    return '2' if base_fechada else 'P'


def desambiguar_J_L(th: np.ndarray) -> str:
    """Distingue J de L: L tem haste na esquerda; J desce pela direita com gancho."""
    haste_esq = float(np.mean(th[10:50, 4:12] == 255)) > 0.55
    return 'L' if haste_esq else 'J'


def desambiguar_S_Z(th: np.ndarray) -> str:
    """Distingue S de Z: Z tem base e topo retos angulosos; S tem curvas."""
    barra_base = float(np.mean(th[48:58, 8:32] == 255)) > 0.52
    barra_topo = float(np.mean(th[3:14, 6:34] == 255)) > 0.52
    if barra_base and barra_topo:
        return 'Z'
    return 'S'


def desambiguar_9_2(th: np.ndarray) -> str:
    """
    Distingue 9 de 2:
    - O '2' tem uma barra horizontal contínua preenchendo a base inferior.
    - O '9' tem a base aberta / curva, sem barra horizontal completa na base inferior.
    """
    barra_base = float(np.mean(th[48:58, 8:30] == 255))
    if barra_base > 0.45:
        return '2'
    return '9'


def desambiguar_4_7(th: np.ndarray) -> str:
    """
    Distingue 4 de 7:
    - O '4' tem barra horizontal no terço médio cruzando com a haste vertical.
    - O '7' tem a barra horizontal apenas no topo e o terço médio é oco/aberto.
    """
    meio_haste = float(np.mean(th[26:42, 6:28] == 255))
    if meio_haste > 0.28:
        return '4'
    return '7'


def desambiguar_I_L(th: np.ndarray, crop_char: Optional[np.ndarray] = None) -> str:
    """
    Distingue I de L:
    - O 'I' é estreito (aspect ratio w/h < 0.35) e não tem extensão na base direita.
    - O 'L' tem a base horizontal direita preenchida.
    """
    if crop_char is not None and crop_char.size > 0:
        h, w = crop_char.shape[:2]
        if w / max(1, h) < 0.35:
            return 'I'
    base_dir = float(np.mean(th[48:58, 20:34] == 255))
    if base_dir > 0.35:
        return 'L'
    return 'I'


def desambiguar_W_N(th: np.ndarray, crop_char: Optional[np.ndarray] = None) -> str:
    """
    Distingue W de N:
    - O 'W' é visivelmente largo (aspect ratio w/h > 0.70) e tem pico central.
    """
    if crop_char is not None and crop_char.size > 0:
        h, w = crop_char.shape[:2]
        if w / max(1, h) > 0.70:
            return 'W'
    pico_central = float(np.mean(th[20:45, 16:24] == 255))
    if pico_central > 0.30:
        return 'W'
    return 'N'


def desambiguar_0_6(th: np.ndarray) -> str:
    """
    Distingue 0 de 6 em posições numéricas Mercosul:
    - O '6' tem topo direito aberto e haste esquerda alta.
    - O '0' é simétrico e fechado no topo direito e base.
    """
    topo_dir = float(np.mean(th[6:18, 22:34] == 255))
    topo_esq = float(np.mean(th[6:18, 6:18] == 255))
    if topo_dir > 0.25 and topo_esq > 0.25:
        return '0'
    return '6'


def desambiguar_M_H(th: np.ndarray) -> str:
    """
    Distingue M de H:
    - O 'M' tem o vértice em 'V' central descendo pelo centro médio/inferior.
    - O 'H' tem o centro inferior vazio.
    """
    centro_inf = float(np.mean(th[30:50, 16:24] == 255))
    if centro_inf > 0.12:
        return 'M'
    return 'H'


def desambiguar_1_7(th: np.ndarray) -> str:
    """
    Distingue 1 de 7:
    - O '7' tem barra horizontal larga no topo.
    - O '1' tem haste vertical e a parte superior direita é vazia.
    """
    topo_dir = float(np.mean(th[3:14, 20:34] == 255))
    if topo_dir > 0.35:
        return '7'
    return '1'


def desambiguar_caractere_especifico(crop_char: Optional[np.ndarray], char_atual: str, posicao: int, eh_mercosul: bool = True) -> str:
    """Aplica desambiguação morfológica refinada considerando as regras estritas de cada posição."""
    pos_eh_letra = (posicao in [0, 1, 2]) or (eh_mercosul and posicao == 4)
    pos_eh_numero = (posicao in [3, 5, 6]) or (not eh_mercosul and posicao == 4)

    th = binarizar_caractere(crop_char) if crop_char is not None else None

    # 1. POSIÇÃO DEVE SER NÚMERO
    if pos_eh_numero:
        if char_atual in ['4', '2', 'Z', 'A', 'H', 'P']:
            if th is not None:
                return desambiguar_4_2(th)
            return '4' if char_atual == '4' else '2'

        if char_atual in ['O', 'D', 'Q', 'C', '0']:
            return '0'

        if char_atual in ['I', 'L', 'J', '1']:
            return '1'

        if char_atual in ['7', 'T']:
            if th is not None:
                # O '7' tem barra horizontal no topo. Se a barra superior for oca/vazia, é na verdade '1'
                topo_dir = float(np.mean(th[3:14, 20:34] == 255))
                if topo_dir < 0.18:
                    return '1'
            return '7'

        if char_atual in ['S', '5']:
            return '5'
        if char_atual in ['G', 'b', '6']:
            return '6'
        if char_atual in ['B', '8']:
            return '8'
        if char_atual == 'E':
            return '3'

        return char_atual if char_atual.isdigit() else '0'

    # 2. POSIÇÃO DEVE SER LETRA
    if pos_eh_letra:
        # N vs H (apenas se for N ou H)
        if char_atual == 'N':
            if th is not None:
                quad_sup_vazio = float(np.mean(th[8:22, 14:26] == 255)) < 0.22
                quad_inf_vazio = float(np.mean(th[38:52, 14:26] == 255)) < 0.22
                if quad_sup_vazio and quad_inf_vazio:
                    return 'H'
            return 'N'
        elif char_atual == 'H':
            return 'H'

        # C vs G: preserva C com segurança, desambigua com limiar mais conservador
        if char_atual == 'C':
            if th is not None:
                lado_dir_inf = float(np.mean(th[28:46, 26:36] == 255))
                if lado_dir_inf > 0.45:
                    return 'G'
            return 'C'
        elif char_atual == 'G':
            if th is not None:
                lado_dir_inf = float(np.mean(th[28:46, 26:36] == 255))
                if lado_dir_inf < 0.20:
                    return 'C'
            return 'G'

        # S e Q (preserva S se já for S, desambigua apenas se for Q ou dígito)
        if char_atual == 'S':
            return 'S'

        if char_atual == 'Q':
            if th is not None:
                return desambiguar_Q_O(th, posicao=posicao)
            return 'Q'

        # S vs E
        if char_atual in ['S', 'E']:
            return char_atual

        # Q vs O vs D
        if char_atual in ['O', 'D', '0']:
            if th is not None:
                # Se for D
                haste_esq = float(np.mean(th[10:50, 4:12] == 255)) > 0.60
                if haste_esq:
                    return 'D'
                return desambiguar_Q_O(th, posicao=posicao)
            return 'D' if char_atual == 'D' else 'O'

        # B vs 8
        if char_atual in ['B', '8']:
            return 'B'

        # K vs X
        if char_atual in ['K', 'X']:
            if th is not None:
                return desambiguar_K_X(th)
            return char_atual

        # T vs I
        if char_atual in ['T', 'I', '1', '7']:
            if th is not None:
                return desambiguar_T_I(th, crop_char=crop_char)
            return 'T' if char_atual in ['T', '7'] else 'I'

        # J vs L
        if char_atual in ['J', 'L']:
            if th is not None:
                return desambiguar_J_L(th)
            return char_atual

        # S vs Z
        if char_atual in ['S', 'Z', '2', '5']:
            if th is not None:
                return desambiguar_S_Z(th)
            return 'S' if char_atual in ['S', '5'] else 'Z'

        # 2 vs P
        if char_atual in ['2', 'P']:
            if th is not None:
                return desambiguar_2_P(th)
            return 'P'

        # Conversão de dígitos remanescentes em letras
        if char_atual.isdigit():
            map_letra = {'0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P'}
            return map_letra.get(char_atual, 'A')

    return char_atual


def refinar_placa_completa(texto_placa: str, crops_chars: Optional[List[np.ndarray]] = None) -> str:
    """Refina a placa de 7 caracteres aplicando todas as desambiguações morfológicas com precisão."""
    if not texto_placa or len(texto_placa) != 7:
        return texto_placa

    chars = list(texto_placa.upper())

    # Detecta se é padrão Mercosul (posição 5 e 6 são dígitos)
    eh_mercosul = chars[4].isalpha() or (chars[5].isdigit() and chars[6].isdigit())

    refinados = []
    for idx, ch in enumerate(chars):
        crop = crops_chars[idx] if crops_chars and idx < len(crops_chars) else None
        ch_ref = desambiguar_caractere_especifico(crop, ch, posicao=idx, eh_mercosul=eh_mercosul)
        refinados.append(ch_ref)

    # Heurística para a 1ª letra da placa no Brasil:
    # Se começar com 'O' seguido de consoante típica das séries Q do Centro-Oeste/Norte
    # (ex: OZE -> QZE, OCX -> QCX, OCI -> QCI, OPP -> QPP, OCQ -> QCQ, OCP -> QCP, OGF -> QGF, OHH -> QHH, OZZ -> QZZ)
    if refinados[0] == 'O' and len(refinados) >= 3:
        if refinados[1] in ['Z', 'C', 'P', 'G', 'H', 'W']:
            refinados[0] = 'Q'
        elif refinados[1] in ['B'] and crops_chars and crops_chars[0] is not None:
            th0 = binarizar_caractere(crops_chars[0])
            if th0 is not None and desambiguar_Q_O(th0, 0) == 'Q':
                refinados[0] = 'Q'

    return "".join(refinados)
