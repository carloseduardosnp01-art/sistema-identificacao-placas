"""
Script para inspecionar fotos com múltiplas placas detectadas.
"""
import os
import sys
sys.path.append(".")
import cv2
from src.detector import DetectorPlacas
from src.pipeline_misto import executar_pipeline_misto

detector = DetectorPlacas()
pasta = "data/drive_fotos/Fotos placas"

fotos_multiplas = [
    "WhatsApp Image 2026-10-07 at 21.37.52 (1).jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.56.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.57.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.40.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.41 (1).jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.42.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.44 (2).jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.45 (2).jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.46 (1).jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.49 (1).jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.52.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.53 (1).jpeg"
]

pasta_multi = "data/inspecao_multiplas"
os.makedirs(pasta_multi, exist_ok=True)

for f in fotos_multiplas:
    img = cv2.imread(os.path.join(pasta, f))
    dets = detector.detectar(img)
    print(f"\n=== {f} ({len(dets)} detecções) ===")
    for idx, d in enumerate(dets):
        crop = d["crop"]
        conf_det = d["confianca"]
        h, w = crop.shape[:2]
        res = executar_pipeline_misto(crop)
        txt = res.get("texto_corrigido", "")
        pad = res.get("padrao", "Nenhum")
        conf_rec = res.get("confianca", 0.0)
        
        # Salva o crop para verificarmos se é legível
        nome_base = f.replace(".jpeg", "").replace(".jpg", "").replace(" ", "_")
        crop_name = f"{nome_base}_p{idx+1}_{txt if txt else 'VAZIO'}.jpg"
        cv2.imwrite(os.path.join(pasta_multi, crop_name), crop)
        
        print(f"  Placa #{idx+1} [{w}x{h}, conf_det={conf_det:.2f}]: txt='{txt}' ({pad}) conf_rec={conf_rec:.2f} -> {crop_name}")
