import os
import sys
import cv2
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.detector import DetectorPlacas
from src.pipeline_misto import endireitar_placa, obter_yolo_caracteres, executar_ocr_multicontraste

det = DetectorPlacas()
yolo = obter_yolo_caracteres()

alvos = ['22_SQC5J75.jpeg', '23_QBH4I14.jpeg', '26_QCX4G24.jpeg', '27_RNK0J18_SQO1I83.jpeg', '30_RNK0J18.jpeg']
pasta = 'data/drive_fotos/Fotos placas'

for a in alvos:
    p = os.path.join(pasta, a)
    img = cv2.imread(p)
    placas = det.detectar(img)
    if not placas:
        print(f"{a}: Placa não detectada")
        continue
    crop = placas[0]['crop']
    reta, ang = endireitar_placa(crop)
    res_y = yolo.reconhecer(reta)
    txt_o, conf_o, raw_o = executar_ocr_multicontraste(reta)
    
    chars_y = [f"{c['char']}({c['conf']:.2f})" for c in res_y.get('caixas_caracteres', [])]
    
    print(f"=== {a} ===")
    print(f"   YOLO Caixas:   {' '.join(chars_y)}")
    print(f"   YOLO Bruto:    {res_y.get('texto_bruto')} | Corrigido: {res_y.get('texto_corrigido')}")
    print(f"   EasyOCR:       Raw='{raw_o}' -> '{txt_o}' (conf={conf_o:.2f})")
