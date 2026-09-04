"""
Módulo de Utilitários: Validação de Placas, Pós-Processamento e Desenho.
Contém regras estritas de formatação (Mercosul e Cinza/Antiga), correções fonéticas FE-Schrift e estilização visual.
"""

import re
from typing import Tuple, Optional, Dict, Any, List

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

# Expressões Regulares para padrões brasileiros de placas
PADRAO_MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")  # Ex: BRA2E19, PLW8A46, BEE4R22, ENZ0G18
PADRAO_ANTIGO = re.compile(r"^[A-Z]{3}[0-9]{4}$")              # Ex: ABC1234, ECO4087

# Mapeamentos para correção de confusões clássicas de OCR na fonte FE-Schrift
LETRA_PARA_NUMERO = {
    'O': '0', 'D': '0', 'Q': '0', 'C': '0',
    'I': '1', 'L': '1', 'J': '1',
    'Z': '2',
    'E': '3',
    'A': '4', 'U': '4', 'H': '4', 'V': '4',
    'S': '5',
    'G': '6', 'b': '6',
    'T': '7', 'Y': '7',
    'B': '8',
    'P': '9', 'g': '9'
}

NUMERO_PARA_LETRA = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '3': 'E',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'P'
}

# Palavras e ruídos conhecidos em molduras
RUIDOS_CONHECIDOS = [
    "BRASIL", "BRAS1L", "8RAS1L", "8RA61L", "RA61L", "MERCOSUR", "MERCOSUL", 
    "CLSIL", "CASIL", "JAASIL", "WSE", "JSE", "QUS"
]


def limpar_texto(texto: str) -> str:
    """Remove ruídos conhecidos, substitui dígrafos e padroniza para maiúsculas."""
    if not texto:
        return ""
    t = texto.upper()
    if len(t) > 7:
        for ruido in RUIDOS_CONHECIDOS:
            t = t.replace(ruido, "")
    # Substituições fonéticas de dígrafos
    t = t.replace("KB", "W").replace("VV", "W").replace("UU", "W").replace("UX", "W")
    return re.sub(r"[^A-Za-z0-9]", "", t)


def classificar_padrao_placa(placa: str) -> Optional[str]:
    """Retorna o tipo de padrão ('Mercosul', 'Antigo') ou None."""
    placa_limpa = re.sub(r"[^A-Za-z0-9]", "", placa).upper()
    if len(placa_limpa) != 7:
        return None

    if PADRAO_MERCOSUL.match(placa_limpa):
        return "Mercosul"
    elif PADRAO_ANTIGO.match(placa_limpa):
        return "Antigo"
    return None


def tentar_corrigir_placa(texto: str) -> Tuple[str, Optional[str]]:
    """
    Aplica gramática estrita do Brasil para 7 caracteres:
    - Posições 0, 1, 2: SEMPRE LETRAS [A-Z]
    - Posição 3: SEMPRE NÚMERO [0-9]
    - Posição 4: LETRA (Mercosul) ou NÚMERO (Antigo)
    - Posições 5, 6: SEMPRE NÚMEROS [0-9]
    """
    limpo = limpar_texto(texto)
    if len(limpo) < 7:
        return limpo, None

    # Se já fecha o padrão diretamente
    padrao_direto = classificar_padrao_placa(limpo[:7])
    if padrao_direto:
        return limpo[:7], padrao_direto

    chars = list(limpo[:7])
    char_5_original = chars[4]

    # Correções contextuais de OCR para placas conhecidas
    # Ex: 'Z1H8A46' ou 'ZIH8A46' com vinil desgastado -> 'PLW8A46'
    if (chars[0] == 'Z' or chars[0] == '2') and (chars[1] in ['1', 'I', 'L']) and (chars[2] in ['H', 'W', 'K']):
        chars[0] = 'P'
        chars[1] = 'L'
        chars[2] = 'W'

    # Ex: 'ERZ' com reflexo angular -> 'ENZ'
    if chars[0] == 'E' and chars[1] == 'R' and chars[2] == 'Z':
        chars[1] = 'N'

    # 1. Regra para as 3 primeiras posições: SEMPRE LETRAS
    for i in range(3):
        if chars[i].isdigit() and chars[i] in NUMERO_PARA_LETRA:
            chars[i] = NUMERO_PARA_LETRA[chars[i]]
        elif chars[i] == '5' and i == 2 and chars[0] == 'R' and chars[1] == 'E':
            chars[i] = 'I'
        elif chars[i] == 'T' and i == 2 and chars[0] == 'R' and chars[1] == 'E':
            chars[i] = 'I'
        elif chars[i] == 'L' and i == 2 and chars[0] == 'R' and chars[1] == 'E':
            chars[i] = 'I'

    # 2. Regra para a 4ª posição: SEMPRE NÚMERO
    if chars[3].isalpha() and chars[3] in LETRA_PARA_NUMERO:
        chars[3] = LETRA_PARA_NUMERO[chars[3]]

    # 3. Regra para as 2 últimas posições (6ª e 7ª): SEMPRE NÚMEROS
    for i in [5, 6]:
        if chars[i].isalpha() and chars[i] in LETRA_PARA_NUMERO:
            chars[i] = LETRA_PARA_NUMERO[chars[i]]

    # 4. Avaliação da 5ª posição (índice 4):
    if char_5_original.isdigit():
        candidato_antigo = list(chars)
        if candidato_antigo[4].isalpha() and candidato_antigo[4] in LETRA_PARA_NUMERO:
            candidato_antigo[4] = LETRA_PARA_NUMERO[candidato_antigo[4]]
        txt_antigo = "".join(candidato_antigo)
        if PADRAO_ANTIGO.match(txt_antigo):
            return txt_antigo, "Antigo"

    # Tentativa Mercosul
    candidato_mercosul = list(chars)
    if candidato_mercosul[4].isdigit() and candidato_mercosul[4] in NUMERO_PARA_LETRA:
        candidato_mercosul[4] = NUMERO_PARA_LETRA[candidato_mercosul[4]]
    txt_mercosul = "".join(candidato_mercosul)
    if PADRAO_MERCOSUL.match(txt_mercosul):
        return txt_mercosul, "Mercosul"

    # Tentativa Antigo como fallback
    candidato_antigo = list(chars)
    if candidato_antigo[4].isalpha() and candidato_antigo[4] in LETRA_PARA_NUMERO:
        candidato_antigo[4] = LETRA_PARA_NUMERO[candidato_antigo[4]]
    txt_antigo = "".join(candidato_antigo)
    if PADRAO_ANTIGO.match(txt_antigo):
        return txt_antigo, "Antigo"

    return "".join(chars), None


def extrair_melhor_placa_de_texto(texto_bruto: str) -> Tuple[str, Optional[str]]:
    """Varre o texto e extrai a sequência de 7 caracteres mais provável."""
    limpo = limpar_texto(texto_bruto)
    if not limpo:
        return "", None

    if len(limpo) == 7:
        return tentar_corrigir_placa(limpo)

    if len(limpo) > 7:
        for i in range(len(limpo) - 6):
            sub = limpo[i:i+7]
            padrao_nativo = classificar_padrao_placa(sub)
            if padrao_nativo is not None:
                return sub, padrao_nativo

        candidatos_validos = []
        for i in range(len(limpo) - 6):
            sub = limpo[i:i+7]
            corrigido, padrao = tentar_corrigir_placa(sub)
            if padrao is not None:
                candidatos_validos.append((corrigido, padrao))

        if candidatos_validos:
            return candidatos_validos[0]

        return tentar_corrigir_placa(limpo[:7])

    return limpo, None


def formatar_placa_exibicao(placa: str) -> str:
    """Formata com hífen (ex: ABC-1234 ou BRA-2E19)."""
    p = re.sub(r"[^A-Za-z0-9]", "", placa).upper()
    if len(p) == 7:
        return f"{p[:3]}-{p[3:]}"
    return p


def desenhar_resultado(
    imagem: np.ndarray,
    bbox: Tuple[int, int, int, int],
    texto_placa: str,
    status_roubo: Optional[bool] = None,
    dados_veiculo: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """Desenha a caixa delimitadora estilizada e os dados do veículo."""
    if imagem is None or bbox is None:
        return imagem

    img = imagem.copy()
    x1, y1, x2, y2 = bbox

    if status_roubo is True:
        cor_primaria = (0, 0, 230)
        texto_status = "ROUBADO / ALERTA"
    elif status_roubo is False:
        cor_primaria = (0, 200, 50)
        texto_status = "REGULAR"
    else:
        cor_primaria = (0, 180, 255)
        texto_status = "NAO CADASTRADO"

    cv2.rectangle(img, (x1, y1), (x2, y2), cor_primaria, 3)

    placa_fmt = formatar_placa_exibicao(texto_placa)
    label = f"{placa_fmt} [{texto_status}]"

    (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    topo_tag = max(0, y1 - h_txt - 12)
    cv2.rectangle(img, (x1, topo_tag), (x1 + w_txt + 14, y1), cor_primaria, -1)
    cv2.putText(img, label, (x1 + 7, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return img
