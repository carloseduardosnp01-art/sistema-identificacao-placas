"""
Módulo de Reconhecimento de Caracteres por YOLO (Estágio 2).
Executa inferência direta de caixas delimitadoras e classes de letras/números (0-9, A-Z)
em imagens de placas em cores naturais (RGB), sem binarização ou filtros manuais.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_CARACTERES_PATH = BASE_DIR / "models" / "detector_caracteres.pt"

from .utils import classificar_padrao_placa, tentar_corrigir_placa


class ReconhecedorCaracteresYOLO:
    """Reconhecedor de Caracteres baseado em modelo YOLO treinado."""

    def __init__(self, caminho_modelo: Optional[str] = None):
        self.caminho_modelo = caminho_modelo or str(MODEL_CARACTERES_PATH)
        self.modelo = None
        self.disponivel = False

        if os.path.exists(self.caminho_modelo):
            try:
                from ultralytics import YOLO
                self.modelo = YOLO(self.caminho_modelo)
                self.disponivel = True
                print(f"[YOLO Caracteres] Modelo carregado com sucesso: {self.caminho_modelo}")
            except Exception as e:
                print(f"[YOLO Caracteres] Erro ao carregar modelo: {e}")
                self.disponivel = False
        else:
            print(f"[YOLO Caracteres] Modelo não encontrado em '{self.caminho_modelo}'. Usando modo de compatibilidade.")

    def reconhecer(self, img_placa: np.ndarray, confianca_minima: float = 0.25) -> Dict[str, Any]:
        """
        Reconhece os caracteres da placa em RGB nativo:
        1. Executa inferência do YOLO no recorte da placa
        2. Ordena os caracteres da esquerda para a direita (coordenada X)
        3. Formata e valida a placa (Mercosul / Antiga)
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

        # Converte para RGB se necessário
        if len(img_placa.shape) == 3 and img_placa.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_placa, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_placa.copy()

        # Inferência com o modelo YOLO
        resultados = self.modelo.predict(img_rgb, conf=confianca_minima, verbose=False)
        if not resultados or len(resultados[0].boxes) == 0:
            return {
                "texto_bruto": "",
                "texto_corrigido": "",
                "padrao": None,
                "confianca": 0.0,
                "caixas_caracteres": [],
                "img_processada": img_rgb
            }

        boxes = resultados[0].boxes
        nomes_classes = self.modelo.names

        deteccoes = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            nome_classe = str(nomes_classes.get(cls_id, "")).upper()

            if nome_classe:
                deteccoes.append({
                    "char": nome_classe,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": conf,
                    "x_centro": (x1 + x2) / 2.0
                })

        # Ordena os caracteres pela posição horizontal da esquerda para a direita
        deteccoes_ordenadas = sorted(deteccoes, key=lambda d: d["x_centro"])

        # Filtra sobreposições muito próximas (duplicações)
        deteccoes_filtradas = []
        for d in deteccoes_ordenadas:
            if not deteccoes_filtradas:
                deteccoes_filtradas.append(d)
                continue
            ultimo_d = deteccoes_filtradas[-1]
            distancia_x = abs(d["x_centro"] - ultimo_d["x_centro"])
            largura_media = (d["bbox"][2] - d["bbox"][0] + ultimo_d["bbox"][2] - ultimo_d["bbox"][0]) / 2.0

            if distancia_x < largura_media * 0.4:
                # Mantém o de maior confiança
                if d["conf"] > ultimo_d["conf"]:
                    deteccoes_filtradas[-1] = d
            else:
                deteccoes_filtradas.append(d)

        texto_bruto = "".join([d["char"] for d in deteccoes_filtradas])
        confs = [d["conf"] for d in deteccoes_filtradas]
        conf_media = float(np.mean(confs)) if confs else 0.0

        texto_corrigido, padrao = tentar_corrigir_placa(texto_bruto)

        return {
            "texto_bruto": texto_bruto,
            "texto_corrigido": texto_corrigido[:7],
            "padrao": padrao,
            "confianca": conf_media,
            "caixas_caracteres": deteccoes_filtradas,
            "img_processada": img_rgb
        }
