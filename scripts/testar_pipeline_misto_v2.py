"""
Script de Validação e Benchmark do Pipeline Misto V2 (Consenso YOLO + EasyOCR Multi-contraste).
Processa todas as 71 imagens da pasta do Google Drive, gera anotações visuais,
compila métricas de desempenho e salva relatórios em CSV e HTML.
"""

import os
import cv2
import time
import pandas as pd
import numpy as np
from pathlib import Path
from src.detector import DetectorPlacas
from src.pipeline_misto import executar_pipeline_misto, endireitar_placa
from src.utils import classificar_padrao_placa, formatar_placa_exibicao

PASTA_FOTOS = Path("data/drive_fotos/Fotos placas")
PASTA_SAIDA = Path("data/resultados_pipeline_misto_v2")
PASTA_IMGS_SAIDA = PASTA_SAIDA / "crops_reconhecidos"

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
PASTA_IMGS_SAIDA.mkdir(parents=True, exist_ok=True)

detector = DetectorPlacas()

arquivos = sorted([f for f in os.listdir(PASTA_FOTOS) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"=== INICIANDO BENCHMARK PIPELINE MISTO V2 EM {len(arquivos)} IMAGENS ===")

resultados = []
inicio_total = time.time()

for idx, nome_arquivo in enumerate(arquivos, 1):
    caminho = PASTA_FOTOS / nome_arquivo
    img = cv2.imread(str(caminho))
    
    if img is None:
        print(f"[{idx}/{len(arquivos)}] Erro ao ler imagem: {nome_arquivo}")
        continue
        
    t0 = time.time()
    dets = detector.detectar(img)
    t_det = time.time() - t0
    
    if not dets:
        resultados.append({
            "Indice": idx,
            "Arquivo": nome_arquivo,
            "Detector": "Falha",
            "Placa_Reconhecida": "",
            "Padrao": "Nenhum",
            "Confianca": 0.0,
            "Metodo": "Nenhum",
            "Tempo_ms": round(t_det * 1000, 1),
            "Imagem_Salva": ""
        })
        print(f"[{idx}/{len(arquivos)}] {nome_arquivo} => Placa nao detectada")
        continue

    crop_placa = dets[0]["crop"]
    
    t1 = time.time()
    res_misto = executar_pipeline_misto(crop_placa)
    t_rec = time.time() - t1
    
    placa_txt = res_misto.get("texto_corrigido", "")
    padrao = res_misto.get("padrao", "Nenhum")
    conf = res_misto.get("confianca", 0.0)
    metodo = res_misto.get("metodo", "Nenhum")
    
    # Salva imagem processada com texto da placa sobreposto
    img_anotada = crop_placa.copy()
    h, w = img_anotada.shape[:2]
    # Cria uma barra inferior para exibir o texto com alto contraste
    barra = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(barra, f"{placa_txt} ({padrao})", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    img_final = np.vstack([img_anotada, barra])
    
    nome_salvo = f"placa_{idx:02d}_{placa_txt if placa_txt else 'SEM_PLACA'}.jpg"
    caminho_salvo = PASTA_IMGS_SAIDA / nome_salvo
    cv2.imwrite(str(caminho_salvo), img_final)
    
    tempo_total = round((t_det + t_rec) * 1000, 1)
    
    resultados.append({
        "Indice": idx,
        "Arquivo": nome_arquivo,
        "Detector": "Sucesso",
        "Placa_Reconhecida": placa_txt,
        "Padrao": padrao if padrao else "Indefinido",
        "Confianca": round(conf, 3),
        "Metodo": metodo,
        "Tempo_ms": tempo_total,
        "Imagem_Salva": nome_salvo
    })
    
    print(f"[{idx}/{len(arquivos)}] {nome_arquivo} => {placa_txt} ({padrao}) | Conf: {conf:.2f} | Metodo: {metodo} | {tempo_total}ms")

tempo_gasto = round(time.time() - inicio_total, 2)
print(f"\n=== PROCESSAMENTO CONCLUIDO EM {tempo_gasto}s ===")

# Compila DataFrame e CSV
df = pd.DataFrame(resultados)
csv_path = PASTA_SAIDA / "relatorio_pipeline_misto_v2.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Relatorio CSV salvo em: {csv_path}")

# Metricas
total_imgs = len(df)
placas_detectadas = len(df[df["Detector"] == "Sucesso"])
placas_7_chars = len(df[df["Placa_Reconhecida"].str.len() == 7])
mercosul_count = len(df[df["Padrao"] == "Mercosul"])
antigo_count = len(df[df["Padrao"] == "Antigo"])
consenso_hibrido_count = len(df[df["Metodo"].str.contains("Consenso")])
ocr_multi_count = len(df[df["Metodo"].str.contains("EasyOCR")])
yolo_count = len(df[df["Metodo"].str.contains("YOLO")])

# Gera Relatório HTML com visualização gráfica e imagens
html_path = PASTA_SAIDA / "relatorio_pipeline_misto_v2.html"
linhas_tabela = ""
for r in resultados:
    cor_badge = "#28a745" if r["Padrao"] == "Mercosul" else ("#17a2b8" if r["Padrao"] == "Antigo" else "#dc3545")
    img_tag = f"<img src='crops_reconhecidos/{r['Imagem_Salva']}' width='200' style='border-radius:4px;'/>" if r["Imagem_Salva"] else "N/A"
    linhas_tabela += f"""
    <tr>
        <td>{r['Indice']}</td>
        <td><small>{r['Arquivo']}</small></td>
        <td><b style='font-size:1.15em; letter-spacing:1px;'>{r['Placa_Reconhecida']}</b></td>
        <td><span style='background-color:{cor_badge}; color:white; padding:3px 8px; border-radius:12px; font-size:0.85em;'>{r['Padrao']}</span></td>
        <td>{r['Confianca']}</td>
        <td><small>{r['Metodo']}</small></td>
        <td>{r['Tempo_ms']} ms</td>
        <td>{img_tag}</td>
    </tr>
    """

html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Relatório Pipeline Misto V2 - Reconhecimento de Placas</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; margin: 20px; color: #333; }}
        h1 {{ color: #1a73e8; margin-bottom: 5px; }}
        .cards {{ display: flex; gap: 15px; margin-bottom: 25px; flex-wrap: wrap; }}
        .card {{ background: white; padding: 15px 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); flex: 1; min-width: 160px; text-align: center; }}
        .card h3 {{ margin: 0; font-size: 0.9em; color: #666; }}
        .card p {{ margin: 8px 0 0; font-size: 1.8em; font-weight: bold; color: #1a73e8; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; vertical-align: middle; }}
        th {{ background-color: #202124; color: white; font-weight: 600; font-size: 0.9em; }}
        tr:hover {{ background-color: #f1f3f4; }}
    </style>
</head>
<body>
    <h1>🚗 Relatório de Validação: Pipeline Misto V2 (YOLO + EasyOCR Multi-contraste)</h1>
    <p>Benchmark executado em <b>{total_imgs} fotos reais</b> do Google Drive (fotos de celular sob ângulos, sombras e reflexos).</p>
    
    <div class="cards">
        <div class="card">
            <h3>Total de Imagens</h3>
            <p>{total_imgs}</p>
        </div>
        <div class="card">
            <h3>Detecção de Placas</h3>
            <p>{placas_detectadas} / {total_imgs} ({placas_detectadas/total_imgs*100:.1f}%)</p>
        </div>
        <div class="card">
            <h3>Placas com 7 Caracteres</h3>
            <p style="color:#28a745;">{placas_7_chars} / {total_imgs} ({placas_7_chars/total_imgs*100:.1f}%)</p>
        </div>
        <div class="card">
            <h3>Padrão Mercosul Estrito</h3>
            <p style="color:#1a73e8;">{mercosul_count}</p>
        </div>
        <div class="card">
            <h3>Consenso Híbrido</h3>
            <p style="color:#6f42c1;">{consenso_hibrido_count}</p>
        </div>
        <div class="card">
            <h3>EasyOCR Multi-contraste</h3>
            <p style="color:#fd7e14;">{ocr_multi_count}</p>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Nome do Arquivo</th>
                <th>Placa Reconhecida</th>
                <th>Padrão</th>
                <th>Confiança</th>
                <th>Método de Decisão</th>
                <th>Tempo</th>
                <th>Recorte Anotado</th>
            </tr>
        </thead>
        <tbody>
            {linhas_tabela}
        </tbody>
    </table>
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Relatorio HTML salvo em: {html_path}")
print("=== RESUMO FINAL ===")
print(f"Total: {total_imgs} imagens")
print(f"Placas com 7 Caracteres: {placas_7_chars} ({placas_7_chars/total_imgs*100:.1f}%)")
print(f"Padrão Mercosul: {mercosul_count}")
print(f"Padrão Antigo: {antigo_count}")
