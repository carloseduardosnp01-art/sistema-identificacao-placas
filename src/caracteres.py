"""
Módulo de Reconhecimento de Caracteres por YOLO (Estágio 2).
Executa o tratamento de imagem em Preto e Branco / Alto Contraste Adaptativo (CLAHE + Adaptativo Local)
e realiza a inferência dos caracteres (0-9, A-Z) com filtragem geométrica e ordenação horizontal.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_CARACTERES_PATH = BASE_DIR / "models" / "detector_caracteres.pt"

from .utils import classificar_padrao_placa, tentar_corrigir_placa, extrair_melhor_placa_de_texto
from .desambiguacao import refinar_placa_completa


def gerar_variacoes_alto_contraste(img_placa: np.ndarray, altura: int = 240) -> List[Tuple[str, np.ndarray]]:
    """
    Gera variações de pré-processamento de alto contraste inteligentes:
    1. CLAHE Suave + Sharpen (resistente a reflexos e sombras graduais)
    2. Binarização Adaptativa Gaussiana Local (limiar bloco a bloco, não afoga a placa)
    3. CLAHE Forte (para placas escuras ou sob forte sombra de para-choque)
    4. Binarização Otsu Inteligente (com correção automática de inversão)
    """
    if img_placa is None or img_placa.size == 0:
        return []

    h, w = img_placa.shape[:2]
    escala = float(altura) / max(1, h)
    w_novo = int(w * escala)
    crop_hd = cv2.resize(img_placa, (w_novo, altura), interpolation=cv2.INTER_LANCZOS4)

    # Converte para escala de cinza
    if len(crop_hd.shape) == 3 and crop_hd.shape[2] == 3:
        gray = cv2.cvtColor(crop_hd, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_hd.copy()

    variacoes = []

    # 1. CLAHE Suave + Sharpen
    clahe_suave = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    cl_suave = clahe_suave.apply(gray)
    kernel_sharp = np.array([[0, -0.4, 0], [-0.4, 2.6, -0.4], [0, -0.4, 0]], dtype=np.float32)
    sharp_suave = cv2.filter2D(cl_suave, -1, kernel_sharp)
    variacoes.append(("CLAHE_Suave_Sharpen", cv2.cvtColor(sharp_suave, cv2.COLOR_GRAY2RGB)))

    # 2. Binarização Adaptativa Gaussiana Local (nunca fica toda preta sob sombras)
    adapt = cv2.adaptiveThreshold(cl_suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    # Garante fundo branco e texto preto
    if np.mean(adapt == 255) < 0.40:
        adapt = cv2.bitwise_not(adapt)
    variacoes.append(("Adaptativo_Gaussiano", cv2.cvtColor(adapt, cv2.COLOR_GRAY2RGB)))

    # 3. CLAHE Forte
    clahe_forte = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl_forte = clahe_forte.apply(gray)
    variacoes.append(("CLAHE_Forte", cv2.cvtColor(cl_forte, cv2.COLOR_GRAY2RGB)))

    # 4. Otsu com verificação de proporção
    blur = cv2.medianBlur(gray, 3)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(otsu == 255) < 0.35:
        otsu = cv2.bitwise_not(otsu)
    variacoes.append(("Otsu_Normalizado", cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)))

    return variacoes


def tratar_placa_pb_alto_contraste(img_placa: np.ndarray, altura: int = 300) -> np.ndarray:
    """Compatibilidade retroativa: retorna a versão com CLAHE e Sharpen de Alto Contraste."""
    vars_img = gerar_variacoes_alto_contraste(img_placa, altura=altura)
    if vars_img:
        return vars_img[0][1]
    return img_placa


class ReconhecedorCaracteresYOLO:
    """Reconhecedor de Caracteres baseado em YOLO com Alto Contraste Adaptativo e Filtragem Geométrica."""

    def __init__(self, caminho_modelo: Optional[str] = None):
        if caminho_modelo and os.path.exists(caminho_modelo):
            self.caminho_modelo = caminho_modelo
        elif (BASE_DIR / "models" / "detector_caracteres.pt").exists():
            self.caminho_modelo = str(BASE_DIR / "models" / "detector_caracteres.pt")
        elif (BASE_DIR / "models" / "yolo_caracteres.pt").exists():
            self.caminho_modelo = str(BASE_DIR / "models" / "yolo_caracteres.pt")
        else:
            self.caminho_modelo = str(MODEL_CARACTERES_PATH)

        self.modelo = None
        self.disponivel = False

        if os.path.exists(self.caminho_modelo):
            try:
                from ultralytics import YOLO
                self.modelo = YOLO(self.caminho_modelo)
                self.disponivel = True
                print(f"[YOLO 2 Caracteres] Modelo carregado com sucesso: {self.caminho_modelo}")
            except Exception as e:
                print(f"[YOLO 2 Caracteres] Erro ao carregar modelo: {e}")
                self.disponivel = False
        else:
            print(f"[YOLO 2 Caracteres] Modelo não encontrado em '{self.caminho_modelo}'.")

    def reconhecer(self, img_placa: np.ndarray, confianca_minima: float = 0.12) -> Dict[str, Any]:
        """
        Identifica caracteres por YOLO avaliando múltiplas variações de contraste:
        1. Gera variações adaptativas de alto contraste (CLAHE, Adaptativo Local, Otsu corrigido)
        2. Executa inferência com filtragem geométrica (descarta caixas fora da linha dos caracteres)
        3. Ordena horizontalmente da esquerda para a direita
        4. Aplica desambiguação posicional (padrão Mercosul / Antigo)
        """
        if img_placa is None or img_placa.size == 0 or not self.disponivel:
            return {
                "texto_bruto": "",
                "texto_corrigido": "",
                "padrao": None,
                "confianca": 0.0,
                "caixas_caracteres": [],
                "img_processada": img_placa
            }

        melhor_resultado = None
        melhor_score = -1.0

        variacoes = gerar_variacoes_alto_contraste(img_placa, altura=240)

        for nome_var, img_var in variacoes:
            h_var, w_var = img_var.shape[:2]

            # Inferência em imgsz=640 e 1280
            for sz in [640, 1280]:
                resultados = self.modelo.predict(img_var, imgsz=sz, conf=confianca_minima, verbose=False)

                if not resultados or len(resultados[0].boxes) == 0:
                    continue

                boxes = resultados[0].boxes
                nomes_classes = self.modelo.names
                deteccoes = []

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    nome_classe = str(nomes_classes.get(cls_id, "")).upper()

                    largura_box = x2 - x1
                    altura_box = y2 - y1
                    y_centro = (y1 + y2) / 2.0

                    # Filtragem geométrica:
                    # 1. Descarta elementos que estejam no topo extremo (tarja azul "BRASIL", y_centro < 18%)
                    if y_centro < h_var * 0.16:
                        continue
                    # 2. Descarta elementos na base extrema (borda do para-choque, y_centro > 94%)
                    if y_centro > h_var * 0.94:
                        continue
                    # 3. Descarta caixas com altura muito pequena (< 20% da altura da placa)
                    if altura_box < h_var * 0.20:
                        continue

                    if nome_classe:
                        deteccoes.append({
                            "char": nome_classe,
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "conf": conf,
                            "x_centro": (x1 + x2) / 2.0,
                            "largura": largura_box
                        })

                if not deteccoes:
                    continue

                # Ordena os caracteres horizontalmente (da esquerda para a direita)
                deteccoes_ordenadas = sorted(deteccoes, key=lambda d: d["x_centro"])

                # Filtra sobreposições de caixas (mantém a de maior confiança)
                deteccoes_filtradas = []
                for d in deteccoes_ordenadas:
                    if not deteccoes_filtradas:
                        deteccoes_filtradas.append(d)
                        continue
                    ultimo_d = deteccoes_filtradas[-1]
                    distancia_x = abs(d["x_centro"] - ultimo_d["x_centro"])
                    largura_media = (d["largura"] + ultimo_d["largura"]) / 2.0

                    if distancia_x < largura_media * 0.45:
                        if d["conf"] > ultimo_d["conf"]:
                            deteccoes_filtradas[-1] = d
                    else:
                        deteccoes_filtradas.append(d)

                texto_bruto = "".join([d["char"] for d in deteccoes_filtradas])
                confs = [d["conf"] for d in deteccoes_filtradas]
                conf_media = float(np.mean(confs)) if confs else 0.0

                # Avalia candidatos de placa completa e sub-janelas
                candidato_sub = ""
                texto_corrigido, padrao = tentar_corrigir_placa(texto_bruto)
                if padrao is None and len(texto_bruto) > 7:
                    candidato_sub, padrao_sub = extrair_melhor_placa_de_texto(texto_bruto)
                    if padrao_sub is not None:
                        texto_corrigido, padrao = candidato_sub, padrao_sub

                # Extrai os recortes de cada caractere detectado
                todos_crops = []
                for d in deteccoes_filtradas:
                    bx1, by1, bx2, by2 = d["bbox"]
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(w_var, bx2), min(h_var, by2)
                    crop_c = img_var[by1:by2, bx1:bx2]
                    todos_crops.append(crop_c)

                # Identifica o offset da sub-janela caso haja mais de 7 caracteres brutos
                crops_alinhados = todos_crops
                if len(texto_bruto) > 7 and len(texto_corrigido) == 7:
                    idx_ini = 0
                    for k in range(len(texto_bruto) - 6):
                        sub_k = texto_bruto[k:k+7]
                        if sub_k == texto_corrigido or (candidato_sub and sub_k == candidato_sub):
                            idx_ini = k
                            break
                    crops_alinhados = todos_crops[idx_ini:idx_ini+7]
                elif len(todos_crops) >= 7:
                    crops_alinhados = todos_crops[:7]

                # Aplica desambiguação morfológica refinada
                if len(texto_corrigido) == 7:
                    texto_refinado = refinar_placa_completa(texto_corrigido, crops_alinhados)
                    padrao_ref = classificar_padrao_placa(texto_refinado)
                    if padrao_ref is not None:
                        texto_corrigido = texto_refinado
                        padrao = padrao_ref

                # Cálculo do score de qualidade
                score = conf_media
                if padrao == "Mercosul":
                    score += 20.0
                elif padrao == "Antigo":
                    score += 15.0

                if len(texto_corrigido) == 7:
                    score += 10.0
                if len(texto_bruto) == 7:
                    score += 5.0

                if score > melhor_score:
                    melhor_score = score
                    melhor_resultado = {
                        "texto_bruto": texto_bruto,
                        "texto_corrigido": texto_corrigido[:7],
                        "padrao": padrao,
                        "confianca": conf_media,
                        "caixas_caracteres": deteccoes_filtradas,
                        "img_processada": img_var,
                        "metodo_contraste": nome_var
                    }

                    # Se encontrou placa Mercosul perfeita com 7 caracteres, encerra o loop
                    if padrao == "Mercosul" and len(texto_corrigido) == 7 and conf_media >= 0.70:
                        break

            if melhor_resultado and melhor_resultado.get("padrao") == "Mercosul" and len(melhor_resultado.get("texto_corrigido", "")) == 7:
                break

        if melhor_resultado is None:
            img_padrao = variacoes[0][1] if variacoes else img_placa
            return {
                "texto_bruto": "",
                "texto_corrigido": "",
                "padrao": None,
                "confianca": 0.0,
                "caixas_caracteres": [],
                "img_processada": img_padrao,
                "metodo_contraste": "Nenhum"
            }

        # Anota as caixas delimitadoras na imagem processada
        img_anotada = melhor_resultado["img_processada"].copy()
        for d in melhor_resultado["caixas_caracteres"]:
            bx1, by1, bx2, by2 = d["bbox"]
            ch = d["char"]
            cf = d["conf"]

            cv2.rectangle(img_anotada, (bx1, by1), (bx2, by2), (0, 255, 200), 2)
            label = f"{ch} {int(cf * 100)}%"
            (tw, th_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img_anotada, (bx1, max(0, by1 - th_txt - 6)), (bx1 + tw + 4, by1), (0, 255, 200), -1)
            cv2.putText(
                img_anotada,
                label,
                (bx1 + 2, max(th_txt, by1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        melhor_resultado["img_processada"] = img_anotada
        return melhor_resultado
