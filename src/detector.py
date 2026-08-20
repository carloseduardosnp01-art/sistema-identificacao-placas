"""
Módulo de Detecção de Placas com YOLO (Ultralytics).
Carrega o modelo treinado no Google Colab e realiza a inferência em imagens.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")


class DetectorPlacas:
    """Classe responsável por carregar o modelo YOLO e detectar placas de veículos."""

    def __init__(self, caminho_modelo: str = MODEL_PATH):
        self.caminho_modelo = caminho_modelo
        self.modelo = self._carregar_modelo()

    def _carregar_modelo(self) -> Optional[YOLO]:
        """Carrega o modelo YOLO se o arquivo existir, ou levanta aviso amigável."""
        if os.path.exists(self.caminho_modelo):
            print(f"[Detector] Carregando modelo customizado: {self.caminho_modelo}")
            return YOLO(self.caminho_modelo)
        else:
            print(f"[Detector] Modelo '{self.caminho_modelo}' não encontrado.")
            # Fallback para o modelo base yolov8n caso o best.pt ainda não tenha sido gerado
            print("[Detector] Usando 'yolov8n.pt' temporariamente como fallback.")
            return YOLO("yolov8n.pt")

    def detectar(self, imagem: np.ndarray, confianca_minima: float = 0.25) -> List[Dict[str, Any]]:
        """
        Executa a detecção de placas na imagem.
        Retorna uma lista de dicionários contendo:
        - 'bbox': (x1, y1, x2, y2) em inteiros
        - 'crop': imagem recortada da placa
        - 'confianca': nível de confiança da detecção YOLO
        """
        if self.modelo is None or imagem is None:
            return []

        # Executa inferência
        resultados = self.modelo(imagem, conf=confianca_minima, verbose=False)
        placas_detectadas = []

        for resultado in resultados:
            caixas = resultado.boxes
            for box in caixas:
                # Coordenadas (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Garante que os limites estejam dentro da imagem
                h, w = imagem.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Margem pequena de segurança ao redor do recorte (padding)
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.05)
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(w, x2 + pad_x)
                crop_y2 = min(h, y2 + pad_y)

                crop = imagem[crop_y1:crop_y2, crop_x1:crop_x2]

                if crop.size > 0:
                    placas_detectadas.append({
                        "bbox": (x1, y1, x2, y2),
                        "crop": crop,
                        "confianca": conf
                    })

        return placas_detectadas
