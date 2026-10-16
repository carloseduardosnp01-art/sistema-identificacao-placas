"""
Script de Comparação Direta: 100% YOLO vs 100% EasyOCR.
Executa os dois motores de forma independente e isolada sobre todas as 71 imagens do dataset,
gerando um comparativo lado a lado para análise detalhada de desempenho.
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
from src.caracteres import ReconhecedorCaracteresYOLO
from src.desambiguacao import refinar_placa_completa
from src.utils import classificar_padrao_placa, extrair_melhor_placa_de_texto, formatar_placa_exibicao

# 1. Função de Deskew (Retificação Geométrica)
def endireitar_placa_hough(img_placa):
    """Retifica a rotação da placa para 0.0° horizontal."""
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
                    if -35.0 < ang < 35.0:
                        angulos.append(ang)

    angulo_estimado = float(np.median(angulos)) if angulos else 0.0
    if abs(angulo_estimado) < 1.5:
        return img_placa, 0.0

    h, w = img_placa.shape[:2]
    centro = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(centro, angulo_estimado, 1.0)
    placa_reta = cv2.warpAffine(img_placa, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return placa_reta, angulo_estimado


# 2. Execução Isolada do YOLO
def reconhecer_apenas_yolo(yolo_engine, crop_placa):
    """Executa exclusivamente o YOLO de caracteres (com Deskew)."""
    crop_reto, angulo = endireitar_placa_hough(crop_placa)
    res = yolo_engine.reconhecer(crop_reto)
    texto = res.get("texto_corrigido") or res.get("texto_bruto", "")
    padrao = res.get("padrao") or "Desconhecido"
    conf = res.get("confianca", 0.0)
    img_anotada = res.get("img_processada", crop_reto)
    return texto, padrao, conf, img_anotada, angulo


# 3. Execução Isolada do EasyOCR
def reconhecer_apenas_easyocr(reader_engine, crop_placa):
    """Executa exclusivamente o EasyOCR (com Deskew, remoção da tarja BRASIL e CLAHE)."""
    crop_reto, angulo = endireitar_placa_hough(crop_placa)
    h, w = crop_reto.shape[:2]

    # Isola o miolo dos caracteres (descartando os 20% superiores onde fica 'BRASIL' e a bandeira)
    miolo = crop_reto[int(h * 0.20):int(h * 0.95), int(w * 0.03):int(w * 0.97)]
    if miolo.size == 0:
        miolo = crop_reto

    mh, mw = miolo.shape[:2]
    escala = 220.0 / max(1, mh)
    miolo_hd = cv2.resize(miolo, (int(mw * escala), 220), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(miolo_hd, cv2.COLOR_BGR2GRAY) if len(miolo_hd.shape) == 3 else miolo_hd.copy()

    # CLAHE Suave + Sharpen
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    cl = clahe.apply(gray)
    kernel_sharp = np.array([[0, -0.4, 0], [-0.4, 2.6, -0.4], [0, -0.4, 0]], dtype=np.float32)
    sh = cv2.filter2D(cl, -1, kernel_sharp)

    try:
        dets = reader_engine.readtext(sh, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", detail=1)
        textos = []
        confs = []
        for d in dets:
            textos.append(d[1])
            confs.append(float(d[2]))

        raw = "".join(textos)
        conf_media = float(np.mean(confs)) if confs else 0.0

        corr, padrao = extrair_melhor_placa_de_texto(raw)
        if len(corr) == 7:
            corr_ref = refinar_placa_completa(corr)
            pad_ref = classificar_padrao_placa(corr_ref)
            if pad_ref is not None:
                corr = corr_ref
                padrao = pad_ref

        img_anotada = cv2.cvtColor(sh, cv2.COLOR_GRAY2RGB)
        return corr, padrao, conf_media, img_anotada
    except Exception as e:
        return "", "Erro", 0.0, crop_reto


def rodar_comparativo_completo():
    pasta_saida = BASE_DIR / "data" / "comparativo_yolo_vs_ocr"
    pasta_yolo = pasta_saida / "anotacoes_yolo"
    pasta_ocr = pasta_saida / "anotacoes_ocr"
    pasta_yolo.mkdir(parents=True, exist_ok=True)
    pasta_ocr.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(list(set(glob.glob(str(BASE_DIR / "data/drive_fotos/**/*.jpeg"), recursive=True))))
    if not arquivos:
        print("[!] Nenhuma foto encontrada.")
        return

    print("=" * 85)
    print("⚔️ COMPARATIVO CIENTÍFICO LADO A LADO: 100% YOLO vs 100% EASYOCR")
    print(f"Total de imagens a serem testadas: {len(arquivos)}")
    print(f"Diretório de saída: {pasta_saida}")
    print("=" * 85)

    detector = DetectorPlacas()
    yolo_engine = ReconhecedorCaracteresYOLO()

    import easyocr
    print("[EasyOCR] Carregando motor de rede neural...")
    reader_engine = easyocr.Reader(["en", "pt"], gpu=False)

    resultados = []

    for idx, caminho_img in enumerate(arquivos, 1):
        nome_arq = os.path.basename(caminho_img)
        img = cv2.imread(caminho_img)
        if img is None:
            continue

        placas = detector.detectar(img, confianca_minima=0.25)
        if not placas:
            resultados.append({
                "Nº": idx,
                "Arquivo": nome_arq,
                "YOLO Placa": "Não detectado",
                "YOLO Padrão": "-",
                "YOLO Conf": "-",
                "OCR Placa": "Não detectado",
                "OCR Padrão": "-",
                "OCR Conf": "-",
                "Concordância": "Nenhum",
                "Ângulo": "0.0°"
            })
            continue

        crop = placas[0]["crop"]

        # 1. Executa exclusivamente o YOLO
        yolo_placa, yolo_pad, yolo_conf, yolo_img, ang = reconhecer_apenas_yolo(yolo_engine, crop)

        # 2. Executa exclusivamente o EasyOCR
        ocr_placa, ocr_pad, ocr_conf, ocr_img = reconhecer_apenas_easyocr(reader_engine, crop)

        # Verifica se ambos concordam
        yolo_fmt = formatar_placa_exibicao(yolo_placa) if yolo_placa else "N/A"
        ocr_fmt = formatar_placa_exibicao(ocr_placa) if ocr_placa else "N/A"

        concordam = "✅ IGUAIS" if (yolo_fmt == ocr_fmt and len(yolo_placa) == 7) else ("⚡ PARCIAL" if set(yolo_placa).intersection(set(ocr_placa)) and len(yolo_placa) >= 5 else "❌ DIFERENTES")

        # Salva anotações
        cv2.imwrite(str(pasta_yolo / f"yolo_{nome_arq}.png"), yolo_img)
        cv2.imwrite(str(pasta_ocr / f"ocr_{nome_arq}.png"), ocr_img)

        print(f"[{idx:02d}/{len(arquivos)}] {nome_arq[:22]:22} | "
              f"YOLO: {yolo_fmt:8} ({yolo_conf*100:.1f}%) | "
              f"OCR: {ocr_fmt:8} ({ocr_conf*100:.1f}%) | {concordam}")

        resultados.append({
            "Nº": idx,
            "Arquivo": nome_arq,
            "YOLO Placa": yolo_fmt,
            "YOLO Padrão": yolo_pad,
            "YOLO Confiança": f"{yolo_conf*100:.1f}%",
            "EasyOCR Placa": ocr_fmt,
            "EasyOCR Padrão": ocr_pad,
            "EasyOCR Confiança": f"{ocr_conf*100:.1f}%",
            "Concordância": concordam,
            "Ângulo Corrigido": f"{ang:+.1f}°"
        })

    df = pd.DataFrame(resultados)
    csv_path = pasta_saida / "comparativo_yolo_vs_ocr.csv"
    html_path = pasta_saida / "comparativo_yolo_vs_ocr.html"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_html(html_path, index=False, escape=False)

    print("\n" + "=" * 85)
    print("📊 RESUMO COMPARATIVO FINAL:")
    print("=" * 85)
    print(f"Total de fotos avaliadas: {len(df)}")
    print("\nStatus de Concordância:")
    print(df["Concordância"].value_counts().to_string())
    print("\nPlacas Válidas com 7 Caracteres:")
    yolo_7 = sum(df["YOLO Placa"].str.replace("-", "").str.len() == 7)
    ocr_7 = sum(df["EasyOCR Placa"].str.replace("-", "").str.len() == 7)
    print(f"  - 100% YOLO fechou 7 caracteres: {yolo_7} de {len(df)} ({yolo_7/len(df)*100:.1f}%)")
    print(f"  - 100% EasyOCR fechou 7 caracteres: {ocr_7} de {len(df)} ({ocr_7/len(df)*100:.1f}%)")
    print(f"\n✅ Relatório CSV salvo em: {csv_path}")
    print(f"✅ Relatório HTML salvo em: {html_path}")

if __name__ == "__main__":
    rodar_comparativo_completo()
