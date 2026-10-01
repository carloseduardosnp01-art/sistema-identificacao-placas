"""
Módulo de Reconhecimento de Caracteres por YOLO (Estágio 2).
Executa o tratamento de imagem em Preto e Branco de Alto Contraste (conforme treinamento no Roboflow)
e realiza a inferência dos caracteres (0-9, A-Z) exclusivamente via YOLO, sem EasyOCR.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_CARACTERES_PATH = BASE_DIR / "models" / "detector_caracteres.pt"

from .utils import classificar_padrao_placa, tentar_corrigir_placa, extrair_melhor_placa_de_texto


def tratar_placa_pb_alto_contraste(img_placa: np.ndarray, altura: int = 300) -> np.ndarray:
    """
    Aplica o tratamento de imagem em Preto e Branco de Alto Contraste (estilo Roboflow puro):
    1. Redimensionamento adaptativo para manter arestas nítidas (altura padrão de 300px)
    2. Conversão para escala de cinza
    3. Redução suave de ruído (filtro de mediana) para atenuar o pontilhado holográfico
    4. Binarização Otsu gerando fundo branco puro (255) e caracteres pretos (0)
    5. Conversão para 3 canais (RGB) para entrada na rede YOLO
    """
    if img_placa is None or img_placa.size == 0:
        return img_placa

    h, w = img_placa.shape[:2]
    escala = float(altura) / max(1, h)
    crop_hd = cv2.resize(img_placa, (int(w * escala), altura), interpolation=cv2.INTER_LANCZOS4)

    # Converte para escala de cinza
    if len(crop_hd.shape) == 3 and crop_hd.shape[2] == 3:
        gray = cv2.cvtColor(crop_hd, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_hd.copy()

    # Filtro de mediana suave para eliminar micro-ruídos da trama do fundo
    blur = cv2.medianBlur(gray, 3)

    # Binarização Otsu para produzir alto contraste puro (P&B)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


class ReconhecedorCaracteresYOLO:
    """Reconhecedor de Caracteres baseado em modelo YOLO treinado com imagens em Alto Contraste (P&B)."""

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

    def reconhecer(self, img_placa: np.ndarray, confianca_minima: float = 0.15) -> Dict[str, Any]:
        """
        Executa a identificação dos caracteres exclusivamente com YOLO 2:
        1. Trata a imagem da placa para Preto e Branco com Alto Contraste (Otsu puro)
        2. Executa inferência do YOLO 2 na imagem binarizada em alta resolução (imgsz=1280)
        3. Se necessário, avalia escalas de altura complementares (300px, 240px, 200px)
        4. Ordena os caracteres da esquerda para a direita (eixo X)
        5. Filtra sobreposições de caixas delimitadoras
        6. Formata a placa (Mercosul / Antigo) e anota visualmente a imagem processada
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

        # Testa alturas de resolução para placas de diferentes distâncias
        alturas_teste = [300, 240, 200]

        for altura_crop in alturas_teste:
            img_pb = tratar_placa_pb_alto_contraste(img_placa, altura=altura_crop)
            resultados = self.modelo.predict(img_pb, imgsz=1280, conf=confianca_minima, verbose=False)

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

                if nome_classe:
                    deteccoes.append({
                        "char": nome_classe,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "conf": conf,
                        "x_centro": (x1 + x2) / 2.0,
                        "largura": x2 - x1
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

            # Avalia tanto a sequência completa quanto a melhor sub-janela de placa
            texto_corrigido, padrao = tentar_corrigir_placa(texto_bruto)
            if padrao is None and len(texto_bruto) > 7:
                candidato_sub, padrao_sub = extrair_melhor_placa_de_texto(texto_bruto)
                if padrao_sub is not None:
                    texto_corrigido, padrao = candidato_sub, padrao_sub

            # Cálculo de score de qualidade da leitura
            score = conf_media + (10.0 if padrao is not None else 0.0)
            if len(texto_corrigido) == 7:
                score += 5.0
            if len(texto_bruto) == 7:
                score += 3.0

            if score > melhor_score:
                melhor_score = score
                melhor_resultado = {
                    "texto_bruto": texto_bruto,
                    "texto_corrigido": texto_corrigido[:7],
                    "padrao": padrao,
                    "confianca": conf_media,
                    "caixas_caracteres": deteccoes_filtradas,
                    "img_processada": img_pb
                }

                # Se encontrou placa completa de 7 caracteres com padrão válido e boa confiança, encerra
                if padrao is not None and len(texto_corrigido) == 7 and conf_media >= 0.75:
                    break

        if melhor_resultado is None:
            img_padrao_pb = tratar_placa_pb_alto_contraste(img_placa, altura=300)
            return {
                "texto_bruto": "",
                "texto_corrigido": "",
                "padrao": None,
                "confianca": 0.0,
                "caixas_caracteres": [],
                "img_processada": img_padrao_pb
            }

        # Anota visualmente as caixas delimitadoras e as classes na imagem P&B tratada (estilo Roboflow)
        img_anotada = melhor_resultado["img_processada"].copy()
        for d in melhor_resultado["caixas_caracteres"]:
            bx1, by1, bx2, by2 = d["bbox"]
            ch = d["char"]
            cf = d["conf"]

            # Caixa delimitadora ciano/verde brilhante conforme Roboflow
            cv2.rectangle(img_anotada, (bx1, by1), (bx2, by2), (0, 255, 200), 2)

            # Rótulo com letra e confiança
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
