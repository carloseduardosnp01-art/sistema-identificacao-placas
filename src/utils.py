"""
Módulo de Utilitários: Validação de Placas, Pós-Processamento e Desenho.
Contém regras de formatação (Mercosul e Cinza/Antiga), correções de OCR e estilização visual.
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
PADRAO_MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")  # Ex: BRA2E19
PADRAO_ANTIGO = re.compile(r"^[A-Z]{3}[0-9]{4}$")              # Ex: ABC1234

# Mapeamentos para correção de confusões clássicas de OCR na fonte FE-Schrift (Mercosul e Antigo)
LETRA_PARA_NUMERO = {
    'O': '0', 'D': '0', 'Q': '0', 'C': '0',
    'I': '1', 'L': '1', 'J': '1',
    'Z': '2',
    'E': '3',
    'A': '4', 'U': '4', 'H': '4',
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

# Palavras e ruídos conhecidos que aparecem em molduras de placas brasileiras
RUIDOS_CONHECIDOS = [
    "BRASIL", "BRAS1L", "8RAS1L", "8RA61L", "MERCOSUR", "MERCOSUL", 
    "CLSIL", "PSL", "SIL", "CASIL", "WSE", "BR"
]


def limpar_texto(texto: str) -> str:
    """Remove caracteres não alfanuméricos, substitui dígrafos comuns e converte para maiúsculo."""
    if not texto:
        return ""
    t = texto.upper()
    # Substituições de dígrafos que representam W na fonte Mercosul desgastada
    t = t.replace("KB", "W").replace("VV", "W").replace("UU", "W")
    return re.sub(r"[^A-Za-z0-9]", "", t)


def classificar_padrao_placa(placa: str) -> Optional[str]:
    """
    Retorna o tipo de padrão ('Mercosul', 'Antigo') ou None caso não seja válido.
    """
    placa_limpa = limpar_texto(placa)
    if len(placa_limpa) != 7:
        return None

    if PADRAO_MERCOSUL.match(placa_limpa):
        return "Mercosul"
    elif PADRAO_ANTIGO.match(placa_limpa):
        return "Antigo"
    return None


def tentar_corrigir_placa(texto: str) -> Tuple[str, Optional[str]]:
    """
    Aplica heurísticas de correção baseadas na posição dos caracteres
    para os formatos Mercosul (LLLNLNN) e Antigo (LLLNNNN).
    """
    limpo = limpar_texto(texto)
    if len(limpo) != 7:
        return limpo, None

    # Tentar validar diretamente
    padrao = classificar_padrao_placa(limpo)
    if padrao:
        return limpo, padrao

    chars = list(limpo)

    # Regra para as 3 primeiras posições: SEMPRE letras
    for i in range(3):
        if chars[i].isdigit() and chars[i] in NUMERO_PARA_LETRA:
            chars[i] = NUMERO_PARA_LETRA[chars[i]]
        # Correção específica: K na 2ª posição de placas como ENZ virando EKZ
        # Se for K mas não fecha padrão, o algoritmo pode testar variações se necessário

    # Regra para a 4ª posição: SEMPRE número
    if chars[3].isalpha() and chars[3] in LETRA_PARA_NUMERO:
        chars[3] = LETRA_PARA_NUMERO[chars[3]]

    # Regra para as 2 últimas posições (6ª e 7ª): SEMPRE números
    for i in [5, 6]:
        if chars[i].isalpha() and chars[i] in LETRA_PARA_NUMERO:
            chars[i] = LETRA_PARA_NUMERO[chars[i]]

    # Avaliação da 5ª posição (índice 4):
    char_5_orig = chars[4]

    # Se já é dígito, prioriza Antigo (LLLNNNN)
    if char_5_orig.isdigit():
        candidato_antigo = list(chars)
        texto_antigo = "".join(candidato_antigo)
        if PADRAO_ANTIGO.match(texto_antigo):
            return texto_antigo, "Antigo"

    # Se já é letra, prioriza Mercosul (LLLNLNN)
    if char_5_orig.isalpha():
        candidato_mercosul = list(chars)
        texto_mercosul = "".join(candidato_mercosul)
        if PADRAO_MERCOSUL.match(texto_mercosul):
            return texto_mercosul, "Mercosul"

    # Caso contrário, tenta converter
    if char_5_orig.isdigit() and char_5_orig in NUMERO_PARA_LETRA:
        candidato_mercosul = list(chars)
        candidato_mercosul[4] = NUMERO_PARA_LETRA[char_5_orig]
        texto_mercosul = "".join(candidato_mercosul)
        if PADRAO_MERCOSUL.match(texto_mercosul):
            return texto_mercosul, "Mercosul"

    if char_5_orig.isalpha() and char_5_orig in LETRA_PARA_NUMERO:
        candidato_antigo = list(chars)
        candidato_antigo[4] = LETRA_PARA_NUMERO[char_5_orig]
        texto_antigo = "".join(candidato_antigo)
        if PADRAO_ANTIGO.match(texto_antigo):
            return texto_antigo, "Antigo"

    # Se ainda não bateu, tenta avaliar se trocar K por N nas 3 primeiras posições resolve
    candidato_k_n = list(chars)
    for i in range(3):
        if candidato_k_n[i] == 'K':
            candidato_k_n[i] = 'N'
    texto_kn = "".join(candidato_k_n)
    if PADRAO_MERCOSUL.match(texto_kn):
        return texto_kn, "Mercosul"
    if PADRAO_ANTIGO.match(texto_kn):
        return texto_kn, "Antigo"

    return "".join(chars), None


def extrair_melhor_placa_de_texto(texto_bruto: str) -> Tuple[str, Optional[str]]:
    """
    Varre um texto bruto retornado pelo OCR e extrai rigorosamente os 7 caracteres
    principais da placa brasileira, eliminando ruídos de moldura e cabeçalho.
    """
    limpo = limpar_texto(texto_bruto)
    if not limpo:
        return "", None

    # 1. Substitui ligaduras comuns de OCR (ex: 1H -> W, VV -> W)
    limpo_ligaduras = limpo.replace("1H", "W").replace("VV", "W")

    # 2. Remove ruídos conhecidos
    for ruido in RUIDOS_CONHECIDOS:
        limpo_ligaduras = limpo_ligaduras.replace(ruido, "")
        limpo = limpo.replace(ruido, "")

    # Se tiver exatamente 7 caracteres, tenta corrigir
    if len(limpo_ligaduras) == 7:
        res, padrao = tentar_corrigir_placa(limpo_ligaduras)
        if padrao:
            return res, padrao

    if len(limpo) == 7:
        res, padrao = tentar_corrigir_placa(limpo)
        if padrao:
            return res, padrao

    # Se tiver mais de 7 caracteres, testa janelas deslizantes de tamanho 7
    candidatos_validos = []
    textos_para_testar = list(set([limpo, limpo_ligaduras]))

    for txt in textos_para_testar:
        if len(txt) >= 7:
            for i in range(len(txt) - 6):
                janela = txt[i:i + 7]
                texto_corrigido, padrao = tentar_corrigir_placa(janela)
                if padrao:
                    score = sum(1 for a, b in zip(janela, texto_corrigido) if a == b)
                    # Bônus se a janela começa no início da string
                    if i == 0:
                        score += 0.5
                    candidatos_validos.append((score, texto_corrigido, padrao))

    if candidatos_validos:
        candidatos_validos.sort(key=lambda x: x[0], reverse=True)
        return candidatos_validos[0][1], candidatos_validos[0][2]

    # Se ainda não encontrou padrão exato, NUNCA retorna mais de 7 caracteres
    # Pega os primeiros 7 caracteres da string limpa
    candidato_final = limpo_ligaduras[:7] if len(limpo_ligaduras) >= 7 else limpo[:7]
    res, padrao = tentar_corrigir_placa(candidato_final)
    return res or candidato_final, padrao


def formatar_placa_exibicao(placa: str) -> str:
    """Formata a placa para exibição amigável (ex: ABC-1234 ou BRA-2E19)."""
    limpa = limpar_texto(placa)
    if len(limpa) == 7:
        return f"{limpa[:3]}-{limpa[3:]}"
    return limpa


def desenhar_resultado(
    imagem: Any,
    bbox: Tuple[int, int, int, int],
    texto_placa: str,
    status_roubo: Optional[int] = None,
    dados_veiculo: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Desenha uma caixa delimitadora destacada e uma tarja de informações sobre a imagem.
    - Verde: Veículo Regular (status_roubo == 0)
    - Vermelho: Veículo Roubado/Furtado (status_roubo == 1)
    - Amarelo: Placa não encontrada no banco (status_roubo is None)
    """
    if cv2 is None or imagem is None:
        return imagem
    img_desenhada = imagem.copy()
    x1, y1, x2, y2 = bbox

    # Define a cor de acordo com o status
    if status_roubo == 1:
        cor = (0, 0, 255)       # Vermelho (BGR) - ALERTA DE ROUBO
        rotulo_status = "ROUBADO / FURTADO"
    elif status_roubo == 0:
        cor = (0, 200, 0)       # Verde (BGR) - REGULAR
        rotulo_status = "REGULAR"
    else:
        cor = (0, 215, 255)     # Amarelo/Laranja (BGR) - NÃO CADASTRADO
        rotulo_status = "NAO REGISTRADO"

    # Desenha retângulo na placa
    cv2.rectangle(img_desenhada, (x1, y1), (x2, y2), cor, 3)

    # Texto a ser exibido no cabeçalho da caixa
    texto_topo = f"{formatar_placa_exibicao(texto_placa)} [{rotulo_status}]"

    # Fundo do texto para contraste
    (w_texto, h_texto), _ = cv2.getTextSize(texto_topo, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y_topo_fundo = max(0, y1 - h_texto - 10)
    cv2.rectangle(img_desenhada, (x1, y_topo_fundo), (x1 + w_texto + 10, y1), cor, -1)

    # Texto
    cv2.putText(
        img_desenhada,
        texto_topo,
        (x1 + 5, y1 - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return img_desenhada
