"""
Script para testar o sistema de identificação de placas em lote com novas imagens.
Processa fotos da pasta 'data/drive_fotos' ou qualquer caminho passado por parâmetro.
"""

import os
import sys
import glob
import cv2
import pandas as pd
from pathlib import Path

# Configura diretório base
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.detector import DetectorPlacas
from src.caracteres import ReconhecedorCaracteresYOLO
from src.ocr import extrair_texto_placa
from src.database import consultar_placa
from src.utils import formatar_placa_exibicao, desenhar_resultado

def rodar_teste_pasta(pasta_fotos="data/drive_fotos", pasta_saida="data/resultados_testes"):
    caminho_pasta = BASE_DIR / pasta_fotos
    caminho_saida = BASE_DIR / pasta_saida
    caminho_saida.mkdir(parents=True, exist_ok=True)

    extensoes = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    arquivos = []
    for ext in extensoes:
        arquivos.extend(glob.glob(str(caminho_pasta / "**" / ext), recursive=True))

    if not arquivos:
        print(f"[!] Nenhuma imagem encontrada em '{caminho_pasta}'.")
        print(f"Coloque as fotos baixadas do Drive em: {caminho_pasta}")
        return

    print("=" * 80)
    print(f"🚗 INICIANDO TESTE DO SISTEMA DE IDENTIFICAÇÃO DE PLACAS")
    print(f"📁 Pasta de entrada: {caminho_pasta}")
    print(f"📸 Total de imagens encontradas: {len(arquivos)}")
    print("=" * 80)

    # Carrega modelos
    detector = DetectorPlacas()
    reconhecedor_yolo = ReconhecedorCaracteresYOLO()

    resultados = []

    for idx, caminho_img in enumerate(arquivos, 1):
        nome_arquivo = os.path.basename(caminho_img)
        img = cv2.imread(caminho_img)
        if img is None:
            print(f"[{idx}/{len(arquivos)}] ❌ Falha ao carregar {nome_arquivo}")
            continue

        print(f"\n[{idx}/{len(arquivos)}] 🔍 Processando: {nome_arquivo}")
        placas_detectadas = detector.detectar(img, confianca_minima=0.25)
        print(f"   ↳ Placas detectadas pelo YOLO: {len(placas_detectadas)}")

        img_anotada = img.copy()

        if not placas_detectadas:
            resultados.append({
                "Arquivo": nome_arquivo,
                "Status Detecção": "Nenhuma placa detectada",
                "Placa Identificada": "-",
                "Padrão": "-",
                "Confiança Detecção": "-",
                "Confiança Leitura": "-",
                "Status Veículo": "Não verificado"
            })
            continue

        for p_idx, p in enumerate(placas_detectadas, 1):
            bbox = p["bbox"]
            crop = p["crop"]
            conf_det = p["confianca"]

            # 1. Tenta com YOLO 2 de Caracteres
            res_caracteres = reconhecedor_yolo.reconhecer(crop) if reconhecedor_yolo.disponivel else None
            
            placa_final = ""
            padrao_final = ""
            conf_leitura = 0.0
            metodo_usado = ""

            if res_caracteres and res_caracteres.get("texto_corrigido"):
                placa_final = res_caracteres["texto_corrigido"]
                padrao_final = res_caracteres.get("padrao") or "Desconhecido"
                conf_leitura = res_caracteres.get("confianca", 0.0)
                metodo_usado = "YOLO Caracteres"
            else:
                # 2. Fallback para EasyOCR se YOLO Caracteres não extrair
                res_ocr = extrair_texto_placa(crop)
                placa_final = res_ocr.get("texto_corrigido") or res_ocr.get("texto_bruto", "")
                padrao_final = res_ocr.get("padrao") or "Desconhecido"
                conf_leitura = res_ocr.get("confianca", 0.0)
                metodo_usado = "EasyOCR (Fallback)"

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
            
            # Anota na imagem original
            cv2.rectangle(img_anotada, (bbox[0], bbox[1]), (bbox[2], bbox[3]), cor, 3)
            cv2.putText(img_anotada, texto_rotulo, (bbox[0], max(25, bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2, cv2.LINE_AA)

            print(f"   ↳ [Placa {p_idx}] {placa_formatada} | Padrão: {padrao_final} | "
                  f"Detecção: {conf_det*100:.1f}% | Leitura: {conf_leitura*100:.1f}% ({metodo_usado}) | {status_veiculo}")

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

    # Exibe resumo final em tabela
    df = pd.DataFrame(resultados)
    print("\n" + "=" * 80)
    print("📊 RESUMO FINAL DOS TESTES:")
    print("=" * 80)
    print(df.to_string(index=False))

    # Salva CSV e HTML para consulta amigável
    caminho_csv = caminho_saida / "relatorio_testes.csv"
    caminho_html = caminho_saida / "relatorio_testes.html"
    df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
    df.to_html(caminho_html, index=False, escape=False)

    print(f"\n✅ Relatório CSV salvo em: {caminho_csv}")
    print(f"✅ Relatório HTML salvo em: {caminho_html}")
    print(f"✅ Imagens com anotações salvas em: {caminho_saida}")

if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else "data/drive_fotos"
    rodar_teste_pasta(pasta)
