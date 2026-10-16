import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector import DetectorPlacas
from src.ocr_hibrido import executar_ocr_hibrido

det = DetectorPlacas()
fotos = ['01_QCU5C70.jpeg', '02_MWX0J06.jpeg', '03_EIB6H02.jpeg', '44_EBR9836.jpeg']

print("=== TESTANDO NOVO MOTOR HIBRIDO (LIVE 63 FPS + AUDITORIA CONTRAN) ===")
for nome in fotos:
    caminho = f'data/drive_fotos/Fotos placas/{nome}'
    img = cv2.imread(caminho)
    placas = det.detectar(img)
    if not placas:
        print(f"[{nome}] Nenhuma placa detectada")
        continue
    crop = placas[0]['crop']
    
    # 1. Modo Live (Fast-Plate puro)
    res_live = executar_ocr_hibrido(crop, modo='live')
    # 2. Modo Híbrido Automático
    res_hib = executar_ocr_hibrido(crop, modo='hibrido')
    
    print(f"[{nome}]")
    print(f"   -> MODO LIVE:    {res_live['texto_corrigido']} em {res_live['tempo_ms']:.2f} ms")
    print(f"   -> MODO HIBRIDO: {res_hib['texto_corrigido']} ({res_hib['metodo']}) em {res_hib['tempo_ms']:.2f} ms")
