import os
import sys
import re
import time
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Adiciona raiz do projeto ao path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.detector import DetectorPlacas
from src.ocr import extrair_texto_placa

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def extrair_placas_gabarito(nome_arquivo: str):
    base = os.path.splitext(nome_arquivo)[0]
    partes = base.split('_')
    placas = []
    for p in partes:
        p_limpa = re.sub(r'[^A-Z0-9]', '', p.upper())
        if len(p_limpa) == 7:
            placas.append(p_limpa)
    return placas

def calcular_distancia(s1, s2):
    """Calcula quantidade de caracteres diferentes entre duas strings de mesmo tamanho."""
    if len(s1) != len(s2):
        return max(len(s1), len(s2))
    return sum(1 for a, b in zip(s1, s2) if a != b)

def main():
    pasta_fotos = os.path.join(BASE_DIR, "data", "drive_fotos", "Fotos placas")
    pasta_relatorios = os.path.join(BASE_DIR, "relatorios_benchmark")
    os.makedirs(pasta_relatorios, exist_ok=True)

    print_flush("=" * 80)
    print_flush("INICIANDO BENCHMARK COMPLETO DO PIPELINE ALPR (FOTOS 01 A 71)")
    print_flush(f"Diretório das fotos: {pasta_fotos}")
    print_flush(f"Diretório de saída:  {pasta_relatorios}")
    print_flush("=" * 80)

    detector = DetectorPlacas()
    arquivos = sorted([f for f in os.listdir(pasta_fotos) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    linhas_planilha = []
    
    total_imagens = len(arquivos)
    total_placas_esperadas = 0
    total_exatos = 0
    total_parciais = 0
    total_erros = 0
    
    tempo_inicio_geral = time.time()

    for i, nome_arq in enumerate(arquivos, 1):
        caminho = os.path.join(pasta_fotos, nome_arq)
        gabarito = extrair_placas_gabarito(nome_arq)
        total_placas_esperadas += len(gabarito)

        t0 = time.time()
        img = cv2.imread(caminho)
        if img is None:
            print_flush(f"[{i:02d}/{total_imagens:02d}] Falha ao ler imagem: {nome_arq}")
            continue

        placas_detectadas = detector.detectar(img, confianca_minima=0.20)
        
        predicoes = []
        metodos = []
        confiancas = []

        for p in placas_detectadas:
            res_ocr = extrair_texto_placa(p["crop"])
            txt = res_ocr.get("texto_corrigido") or ""
            metodo = res_ocr.get("metodo", "N/A")
            conf = res_ocr.get("confianca", 0.0)
            if txt:
                predicoes.append(txt)
                metodos.append(metodo)
                confiancas.append(conf)

        tempo_proc = time.time() - t0

        # Mapeamento de cada placa esperada contra as predições
        for g in gabarito:
            if not predicoes:
                status = "NENHUMA_DETECCAO"
                melhor_pred = "-"
                metodo_usado = "-"
                conf_usada = 0.0
                dist = 7
                total_erros += 1
            else:
                melhor_pred = min(predicoes, key=lambda p: calcular_distancia(p, g))
                idx_pred = predicoes.index(melhor_pred)
                metodo_usado = metodos[idx_pred]
                conf_usada = confiancas[idx_pred]
                dist = calcular_distancia(melhor_pred, g)

                if dist == 0:
                    status = "EXATO"
                    total_exatos += 1
                elif dist == 1:
                    status = "PARCIAL_1_CHAR"
                    total_parciais += 1
                else:
                    status = f"DIVERGENTE_{dist}_CHARS"
                    total_erros += 1

            linhas_planilha.append({
                "Foto_ID": i,
                "Arquivo": nome_arq,
                "Placa_Gabarito": g,
                "Placa_Predita": melhor_pred,
                "Status": status,
                "Distancia_Erros": dist,
                "Metodo_OCR": metodo_usado,
                "Confianca": f"{conf_usada:.2f}" if conf_usada else "-",
                "Todas_Predicoes": ", ".join(predicoes) if predicoes else "Nenhuma",
                "Tempo_s": f"{tempo_proc:.2f}"
            })

        print_flush(f"[{i:02d}/{total_imagens:02d}] {nome_arq} | Gabarito: {gabarito} | Pred: {predicoes} | Tempo: {tempo_proc:.2f}s")

    tempo_total = time.time() - tempo_inicio_geral

    # Salva Planilha Excel e CSV
    df = pd.DataFrame(linhas_planilha)
    caminho_excel = os.path.join(pasta_relatorios, "benchmark_placas_completo.xlsx")
    caminho_csv = os.path.join(pasta_relatorios, "benchmark_placas_completo.csv")
    
    df.to_excel(caminho_excel, index=False)
    df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")

    print(f"\n[OK] Planilhas salvas com sucesso:")
    print(f"     -> {caminho_excel}")
    print(f"     -> {caminho_csv}")

    # Geração dos Gráficos com Matplotlib
    pct_exatos = (total_exatos / max(1, total_placas_esperadas)) * 100
    pct_parciais = (total_parciais / max(1, total_placas_esperadas)) * 100
    pct_erros = (total_erros / max(1, total_placas_esperadas)) * 100

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico de Pizza / Donut
    categorias = ['Exato (100%)', 'Parcial (1 char dif)', 'Divergente / Não detectado']
    valores = [total_exatos, total_parciais, total_erros]
    cores = ['#2ecc71', '#f39c12', '#e74c3c']

    wedges, texts, autotexts = ax1.pie(
        valores,
        labels=categorias,
        autopct='%1.1f%%',
        startangle=140,
        colors=cores,
        explode=(0.05, 0.05, 0.05),
        wedgeprops=dict(width=0.45, edgecolor='w')
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_weight('bold')
    ax1.set_title("Distribuição Global de Precisão ALPR\n(Total: 85 Placas em 71 Fotos)", fontsize=13, fontweight='bold', pad=15)

    # Gráfico de Barras com contagem absoluta
    bars = ax2.bar(categorias, valores, color=cores, width=0.55, edgecolor='#333333', linewidth=1.2)
    ax2.set_ylabel("Quantidade de Placas", fontsize=11, fontweight='bold')
    ax2.set_title("Contagem Absoluta por Categoria de Reconhecimento", fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylim(0, max(valores) + 12)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f'{h} ({h/total_placas_esperadas*100:.1f}%)',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    caminho_grafico = os.path.join(pasta_relatorios, "benchmark_grafico_precisao.png")
    plt.savefig(caminho_grafico, dpi=300)
    plt.close()

    print(f"[OK] Gráfico salvo com sucesso:")
    print(f"     -> {caminho_grafico}")

    print("\n" + "=" * 80)
    print("RESUMO CONSOLIDADO DO BENCHMARK")
    print(f"Total de Imagens:           {total_imagens}")
    print(f"Total de Placas Esperadas:  {total_placas_esperadas}")
    print(f"Acertos Exatos (100%):      {total_exatos} ({pct_exatos:.2f}%)")
    print(f"Acertos Parciais (1 char):  {total_parciais} ({pct_parciais:.2f}%)")
    print(f"Erros / Divergentes:        {total_erros} ({pct_erros:.2f}%)")
    print(f"Tempo Total de Execução:    {tempo_total:.2f}s ({tempo_total/total_imagens:.2f}s por imagem)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
