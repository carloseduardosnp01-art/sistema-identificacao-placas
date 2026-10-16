"""
Script v3 para teste do sistema de identificação de placas com Desambiguação Morfológica Avançada.
Resolve confusões clássicas: D/O/Q, 4/2, T/I, M/H/N, X/K, B/8, J/L, S/Z, 2/P.
Salva todos os novos resultados na pasta 'data/resultados_testes_v3'.
"""

import os
import sys
import glob
import cv2
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.detector import DetectorPlacas
from src.ocr import extrair_texto_placa
from src.database import consultar_placa
from src.utils import formatar_placa_exibicao

def rodar_teste_pasta_v3(pasta_fotos="data/drive_fotos", pasta_saida="data/resultados_testes_v3"):
    caminho_pasta = BASE_DIR / pasta_fotos
    caminho_saida = BASE_DIR / pasta_saida
    caminho_saida.mkdir(parents=True, exist_ok=True)

    extensoes = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    arquivos = []
    for ext in extensoes:
        arquivos.extend(glob.glob(str(caminho_pasta / "**" / ext), recursive=True))

    arquivos = sorted(list(set(arquivos)))

    if not arquivos:
        print(f"[!] Nenhuma imagem encontrada em '{caminho_pasta}'.")
        return

    print("=" * 80)
    print("🚀 INICIANDO TESTES V3 COM DESAMBIGUAÇÃO MORFOLÓGICA AVANÇADA")
    print(f"📁 Pasta de entrada: {caminho_pasta}")
    print(f"📂 Pasta de saída: {caminho_saida}")
    print(f"📸 Total de imagens: {len(arquivos)}")
    print("=" * 80)

    detector = DetectorPlacas()
    resultados = []

    for idx, caminho_img in enumerate(arquivos, 1):
        nome_arquivo = os.path.basename(caminho_img)
        img = cv2.imread(caminho_img)
        if img is None:
            continue

        placas_detectadas = detector.detectar(img, confianca_minima=0.25)
        img_anotada = img.copy()

        if not placas_detectadas:
            resultados.append({
                "Arquivo": nome_arquivo,
                "Status Detecção": "Nenhuma placa detectada",
                "Placa Identificada": "-",
                "Padrão": "Não detectado",
                "Confiança Detecção": "-",
                "Confiança Leitura": "-",
                "Método": "Nenhum",
                "Status Veículo": "Não verificado"
            })
            continue

        for p_idx, p in enumerate(placas_detectadas, 1):
            bbox = p["bbox"]
            crop = p["crop"]
            conf_det = p["confianca"]

            # Extração com Desambiguação Morfológica V3
            res = extrair_texto_placa(crop)

            placa_final = res.get("texto_corrigido") or res.get("texto_bruto", "")
            padrao_final = res.get("padrao") or "Desconhecido"
            conf_leitura = res.get("confianca", 0.0)
            metodo_usado = res.get("metodo", "YOLO/OCR")

            # Consulta no banco de dados SQLite
            consulta = consultar_placa(placa_final) if placa_final else None
            status_veiculo = "Não cadastrado / Sem alerta"
            roubado = False

            if consulta:
                if consulta.get("roubado"):
                    status_veiculo = "🚨 ROUBADO / FURTADO"
                    roubado = True
                else:
                    status_veiculo = "✅ Regular"

            placa_formatada = formatar_placa_exibicao(placa_final) if placa_final else "N/A"
            texto_rotulo = f"{placa_formatada} ({conf_leitura*100:.0f}%)"
            cor = (0, 0, 255) if roubado else (0, 255, 0)

            # Desenha bounding box e rótulo
            cv2.rectangle(img_anotada, (bbox[0], bbox[1]), (bbox[2], bbox[3]), cor, 3)
            cv2.putText(img_anotada, texto_rotulo, (bbox[0], max(25, bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2, cv2.LINE_AA)

            print(f"[{idx:02d}/{len(arquivos)}] {nome_arquivo[:25]:25} | "
                  f"Placa: {placa_formatada:9} ({padrao_final:11}) | "
                  f"Conf: {conf_leitura*100:.1f}% | {metodo_usado}")

            resultados.append({
                "Arquivo": nome_arquivo,
                "Status Detecção": f"OK ({conf_det*100:.1f}%)",
                "Placa Identificada": placa_formatada,
                "Padrão": padrao_final,
                "Confiança Detecção": f"{conf_det*100:.1f}%",
                "Confiança Leitura": f"{conf_leitura*100:.1f}%",
                "Método": metodo_usado,
                "Status Veículo": status_veiculo
            })

        # Salva imagem anotada
        caminho_img_anotada = caminho_saida / f"resultado_{nome_arquivo}"
        cv2.imwrite(str(caminho_img_anotada), img_anotada)

    # Gera relatórios finais V3
    df = pd.DataFrame(resultados)
    caminho_csv = caminho_saida / "relatorio_testes_v3.csv"
    caminho_html = caminho_saida / "relatorio_testes_v3.html"

    df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
    df.to_html(caminho_html, index=False, escape=False)

    print("\n" + "=" * 80)
    print("📊 RESUMO FINAL DOS TESTES V3:")
    print("=" * 80)
    print(f"Total de registros avaliados: {len(df)}")
    print("\nDistribuição por Padrão de Placa:")
    print(df["Padrão"].value_counts().to_string())
    print("\nMétodos utilizados:")
    print(df["Método"].value_counts().to_string())
    print(f"\n✅ Relatório CSV salvo em: {caminho_csv}")
    print(f"✅ Relatório HTML salvo em: {caminho_html}")
    print(f"✅ Imagens com anotações salvas em: {caminho_saida}")

if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else "data/drive_fotos"
    rodar_teste_pasta_v3(pasta)
