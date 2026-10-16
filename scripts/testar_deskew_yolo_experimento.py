"""
Experimento de Retificação de Ângulo (Deskew) + YOLO de Caracteres.
Testa especificamente as 36 fotos que antes ativavam o EasyOCR,
para avaliar se o YOLO consegue identificar corretamente os 7 caracteres
quando a placa é endireitada (0.0° horizontal) e a tarja superior é tratada.
"""

import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.detector import DetectorPlacas
from src.caracteres import ReconhecedorCaracteresYOLO, gerar_variacoes_alto_contraste
from src.desambiguacao import refinar_placa_completa
from src.utils import classificar_padrao_placa, tentar_corrigir_placa, extrair_melhor_placa_de_texto, formatar_placa_exibicao

def endireitar_placa_hough(img_placa):
    """
    Calcula a inclinação da placa via transformada probabilística de Hough
    e rotaciona a imagem para deixá-la 100% reta e horizontal (0.0°).
    """
    if img_placa is None or img_placa.size == 0:
        return img_placa, 0.0

    cinza = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY) if len(img_placa.shape) == 3 else img_placa
    bordas = cv2.Canny(cinza, 50, 150, apertureSize=3)
    linhas = cv2.HoughLinesP(bordas, 1, np.pi / 180, threshold=45, minLineLength=35, maxLineGap=12)

    angulos = []
    if linhas is not None:
        for linha in linhas:
            l = linha.ravel()
            if len(l) >= 4:
                x1, y1, x2, y2 = l[:4]
                if x2 != x1:
                    ang = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    # Considera apenas inclinações veiculares realistas (-35° a +35°)
                    if -35.0 < ang < 35.0:
                        angulos.append(ang)

    angulo_estimado = float(np.median(angulos)) if angulos else 0.0

    # Só rotaciona se o ângulo for significativo (> 1.5 grau)
    if abs(angulo_estimado) < 1.5:
        return img_placa, 0.0

    h, w = img_placa.shape[:2]
    centro = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(centro, angulo_estimado, 1.0)
    placa_reta = cv2.warpAffine(img_placa, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return placa_reta, angulo_estimado

def executar_experimento():
    pasta_saida = BASE_DIR / "data" / "experimento_deskew_yolo"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # Carrega a lista das 36 amostras que ativaram EasyOCR na V4
    csv_v4 = BASE_DIR / "data" / "resultados_testes_v4" / "relatorio_testes_v4.csv"
    if not csv_v4.exists():
        print("[!] Arquivo de resultados V4 não encontrado.")
        return

    df_v4 = pd.read_csv(csv_v4)
    ocr_v4 = df_v4[df_v4["Método"].str.contains("EasyOCR", na=False)]

    detector = DetectorPlacas()
    yolo_char = ReconhecedorCaracteresYOLO()

    print("=" * 85)
    print("🔬 EXPERIMENTO: RETIFICAÇÃO GEOMÉTRICA (DESKEW) + YOLO DE CARACTERES")
    print(f"Total de fotos avaliadas que antes usavam EasyOCR: {len(ocr_v4)}")
    print(f"Resultados visuais serão salvos em: {pasta_saida}")
    print("=" * 85)

    resultados = []

    for idx, row in ocr_v4.iterrows():
        nome_arquivo = row["Arquivo"]
        leitura_anterior = row["Placa Identificada"]

        # Localiza arquivo na pasta drive
        arquivos_busca = list(BASE_DIR.glob(f"data/drive_fotos/**/{nome_arquivo}"))
        if not arquivos_busca:
            continue
        caminho_img = arquivos_busca[0]

        img = cv2.imread(str(caminho_img))
        if img is None:
            continue

        placas = detector.detectar(img, confianca_minima=0.25)
        if not placas:
            continue

        # Pega a placa correspondente
        crop_original = placas[0]["crop"]

        # 1. Aplica o Deskew (retificação da rotação)
        crop_reto, angulo = endireitar_placa_hough(crop_original)

        # 2. Executa o YOLO de Caracteres na placa retificada
        res_yolo_reto = yolo_char.reconhecer(crop_reto)

        # 3. Também executa na placa original (sem deskew) para medir ganho
        res_yolo_sem_deskew = yolo_char.reconhecer(crop_original)

        chars_sem_deskew = len(res_yolo_sem_deskew.get("caixas_caracteres", []))
        chars_com_deskew = len(res_yolo_reto.get("caixas_caracteres", []))

        placa_yolo = res_yolo_reto.get("texto_corrigido", "")
        padrao_yolo = res_yolo_reto.get("padrao") or "Desconhecido"
        conf_yolo = res_yolo_reto.get("confianca", 0.0)

        # Salva imagem anotada
        img_anotada = res_yolo_reto.get("img_processada", crop_reto).copy()
        caminho_salvo = pasta_saida / f"deskew_{nome_arquivo}.png"
        cv2.imwrite(str(caminho_salvo), img_anotada)

        print(f"[{len(resultados)+1:02d}/{len(ocr_v4)}] {nome_arquivo[:22]:22} | "
              f"Ângulo: {angulo:+5.1f}° | Chars: {chars_sem_deskew} ➔ {chars_com_deskew} | "
              f"V4 (OCR): {leitura_anterior:8} | YOLO c/ Deskew: {placa_yolo:8} ({padrao_yolo}) | Conf: {conf_yolo*100:.1f}%")

        resultados.append({
            "Arquivo": nome_arquivo,
            "Ângulo Corrigido (graus)": f"{angulo:+.1f}°",
            "Chars YOLO Sem Deskew": chars_sem_deskew,
            "Chars YOLO Com Deskew": chars_com_deskew,
            "Leitura V4 (EasyOCR)": leitura_anterior,
            "Leitura YOLO (Com Deskew)": placa_yolo,
            "Padrão": padrao_yolo,
            "Confiança YOLO": f"{conf_yolo*100:.1f}%"
        })

    df_res = pd.DataFrame(resultados)
    df_res.to_csv(pasta_saida / "relatorio_experimento_deskew.csv", index=False, encoding="utf-8-sig")
    df_res.to_html(pasta_saida / "relatorio_experimento_deskew.html", index=False, escape=False)

    print("\n" + "=" * 85)
    print("📊 RESUMO DO EXPERIMENTO:")
    print("=" * 85)
    print(f"Total avaliado: {len(df_res)}")
    print(f"Placas com 7 caracteres identificados pelo YOLO após Deskew: {sum(df_res['Chars YOLO Com Deskew'] == 7)}")
    print(f"Placas no Padrão Mercosul validadas pelo YOLO: {sum(df_res['Padrão'] == 'Mercosul')}")
    print(f"\n✅ Relatório CSV salvo em: {pasta_saida / 'relatorio_experimento_deskew.csv'}")
    print(f"✅ Imagens visuais salvas em: {pasta_saida}")

if __name__ == "__main__":
    executar_experimento()
