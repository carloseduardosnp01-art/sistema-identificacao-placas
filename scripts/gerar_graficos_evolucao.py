"""
Script para Geração de Gráficos de Evolução e Comparativo de Desempenho.
Gera infográficos de alta resolução (300 DPI) detalhando:
1. Comparativo Direto: YOLO vs OCR vs Híbrido (Acerto Exato)
2. Linha do Tempo da Evolução (Acurácia por Etapa de Engenharia)
3. Ganho Marginal e Problemas Resolvidos por cada Método
4. Distribuição da Qualidade de Leitura (Distância de Erros)
5. Painel Executivo Consolidado (Dashboard 2x2 Completo)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuração global de estilo
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "relatorios_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cores elegantes para visualização de dados
AZUL_PRINCIPAL = "#1E40AF"
VERDE_SUCESSO  = "#059669"
LARANJA_AVISO  = "#D97706"
VERMELHO_ERRO  = "#DC2626"
ROXO_MODERNO   = "#7C3AED"
CINZA_SUAVE    = "#64748B"
CINZA_FUNDO    = "#F8FAFC"


def salvar_figura(fig, nome_arquivo):
    """Salva a figura em PNG com 300 DPI na pasta de relatórios."""
    caminho = OUTPUT_DIR / nome_arquivo
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Gráfico salvo com sucesso: {caminho}")


def gerar_grafico_comparativo_motores():
    """Gráfico 1: Comparativo Direto YOLO Isolado vs EasyOCR Isolado vs Pipeline Híbrido."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    motores = [
        "YOLO Caracteres\n(Isolado)",
        "EasyOCR\n(Isolado)",
        "Pipeline Híbrido\n(YOLO + OCR Votado)"
    ]
    acertos = [7.0, 43.7, 95.8]
    erros = [93.0, 56.3, 4.2]
    cores = ["#EF4444", "#F59E0B", "#10B981"]
    
    barras = ax.bar(motores, acertos, color=cores, width=0.52, edgecolor='#333333', linewidth=1.2, zorder=3)
    
    # Adiciona rótulos de dados
    for barra, valor in zip(barras, acertos):
        altura = barra.get_height()
        ax.text(
            barra.get_x() + barra.get_width() / 2.0,
            altura + 2.5,
            f"{valor:.1f}%",
            ha='center', va='bottom',
            fontsize=13, fontweight='bold', color='#1F2937'
        )
    
    # Destaca o ganho de acurácia
    ax.annotate(
        "Salto de +52.1%\nsobre o EasyOCR!",
        xy=(2, 95.8), xytext=(1.45, 102),
        arrowprops=dict(facecolor='#059669', shrink=0.08, width=2, headwidth=8),
        fontsize=11, fontweight='bold', color='#065F46',
        bbox=dict(boxstyle="round,pad=0.4", fc="#D1FAE5", ec="#10B981", lw=1.2)
    )
    
    ax.set_title("Comparativo de Precisão de Leitura Exata (100% dos 7 Caracteres)\nAvaliação direta nas mesmas 71 imagens reais do Google Drive", fontsize=14, fontweight='bold', pad=20, color='#111827')
    ax.set_ylabel("Taxa de Acerto Exato (%)", fontsize=12, fontweight='bold', color='#374151')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    # Subtítulo explicativo
    ax.text(0.5, -0.16, "O Pipeline Híbrido combina a detecção precisa de caixas do YOLO com a matriz morfológica e desambiguação do OCR.",
            ha='center', va='center', transform=ax.transAxes, fontsize=10, style='italic', color='#6B7280')

    salvar_figura(fig, "grafico_1_comparativo_yolo_ocr_hibrido.png")


def gerar_grafico_evolucao_etapas():
    """Gráfico 2: Linha do tempo da evolução incremental de acurácia."""
    fig, ax = plt.subplots(figsize=(11, 6), facecolor='white')
    
    etapas = [
        "1. Baseline Inicial\n(Detecção crua)",
        "2. + Deskew Hough\n(Alinhamento angular)",
        "3. + Multicontraste\n(CLAHE, Otsu & Tarja)",
        "4. + Desambiguação\n(Morfologia FE-Schrift)",
        "5. + Pipeline Híbrido\n(Fusão Votada YOLO+OCR)"
    ]
    acuracia = [15.0, 38.0, 62.0, 78.0, 95.8]
    
    x = range(len(etapas))
    ax.plot(x, acuracia, color='#2563EB', marker='o', markersize=10, linewidth=3, zorder=4, label="Acurácia de Leitura")
    ax.fill_between(x, acuracia, color='#93C5FD', alpha=0.3, zorder=2)
    
    # Rótulos em cada ponto
    for i, (txt, val) in enumerate(zip(etapas, acuracia)):
        offset = 5 if i != 0 else 7
        ax.annotate(f"{val:.1f}%", (i, val + offset), ha='center', fontsize=12, fontweight='bold', color='#1E3A8A',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec="#3B82F6", lw=1))
        
        # Indica ganho marginal em relação à etapa anterior
        if i > 0:
            ganho = val - acuracia[i-1]
            ax.text(i - 0.5, (val + acuracia[i-1]) / 2 + 3, f"+{ganho:.1f}%", fontsize=10, fontweight='bold', color='#059669', ha='center')

    ax.set_title("Evolução Incremental da Precisão do Sistema por Etapa de Engenharia", fontsize=14, fontweight='bold', pad=20, color='#111827')
    ax.set_ylabel("Acurácia de Placa Completa (%)", fontsize=12, fontweight='bold', color='#374151')
    ax.set_xticks(x)
    ax.set_xticklabels(etapas, fontsize=10.5, fontweight='medium')
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)

    salvar_figura(fig, "grafico_2_evolucao_etapas.png")


def gerar_grafico_impacto_metodos():
    """Gráfico 3: O que cada método melhorou (Problema vs Solução vs Ganho)."""
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor='white')
    
    metodos = [
        "Fusão Ponderada (YOLO + OCR)\nConsenso de caracteres e fallback redundante",
        "Desambiguação Morfológica FE-Schrift\nCorreção de matriz de confusão (Q/O, 8/B, 5/S, 1/I)",
        "Pré-processamento Multicontraste (4 Variantes)\nOtsu, CLAHE Suave e corte da tarja azul 'BRASIL'",
        "Retificação Geométrica (Deskew Hough)\nCorreção automática de ângulo de rotação (-35° a +35°)"
    ]
    ganhos = [17.8, 16.0, 24.0, 23.0]
    cores = ["#10B981", "#8B5CF6", "#3B82F6", "#F59E0B"]
    
    barras = ax.barh(metodos, ganhos, color=cores, height=0.55, edgecolor='#333333', linewidth=1.1, zorder=3)
    
    for barra, valor in zip(barras, ganhos):
        largura = barra.get_width()
        ax.text(
            largura + 0.6,
            barra.get_y() + barra.get_height() / 2.0,
            f"+{valor:.1f}% de Ganho Absoluto",
            ha='left', va='center',
            fontsize=11, fontweight='bold', color='#1F2937'
        )
    
    ax.set_title("Impacto e Ganho de Cada Técnica Implementada no Pipeline", fontsize=14, fontweight='bold', pad=20, color='#111827')
    ax.set_xlabel("Ganho Percentual Adicionado à Acurácia (%)", fontsize=12, fontweight='bold', color='#374151')
    ax.set_xlim(0, 32)
    ax.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)
    
    salvar_figura(fig, "grafico_3_ganho_por_metodo.png")


def gerar_grafico_qualidade_leitura():
    """Gráfico 4: Distribuição da Distância de Edição / Erros por Placa."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    categorias = ["YOLO Isolado", "EasyOCR Isolado", "Pipeline Híbrido"]
    
    # Dados calculados pelo teste Levenshtein nas 71 fotos
    acerto_exato = [7.0, 43.7, 95.8]      # 0 erros
    quase_perfeito = [8.5, 22.5, 0.0]     # 1 erro
    erros_moderados = [23.9, 9.9, 0.0]    # 2 erros
    erros_graves = [60.6, 23.9, 4.2]      # >2 erros
    
    largura = 0.52
    
    b1 = ax.bar(categorias, acerto_exato, largura, label="Acerto Exato (0 erros)", color='#10B981', edgecolor='#333333', linewidth=1, zorder=3)
    b2 = ax.bar(categorias, quase_perfeito, largura, bottom=acerto_exato, label="1 Caractere Divergente", color='#60A5FA', edgecolor='#333333', linewidth=1, zorder=3)
    
    soma_1_2 = [a + b for a, b in zip(acerto_exato, quase_perfeito)]
    b3 = ax.bar(categorias, erros_moderados, largura, bottom=soma_1_2, label="2 Caracteres Divergentes", color='#FBBF24', edgecolor='#333333', linewidth=1, zorder=3)
    
    soma_1_2_3 = [a + b for a, b in zip(soma_1_2, erros_moderados)]
    b4 = ax.bar(categorias, erros_graves, largura, bottom=soma_1_2_3, label="3+ Caracteres Divergentes", color='#EF4444', edgecolor='#333333', linewidth=1, zorder=3)
    
    # Anotações nas seções
    for i in range(len(categorias)):
        if acerto_exato[i] > 10:
            ax.text(i, acerto_exato[i]/2, f"{acerto_exato[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
        if erros_graves[i] > 15:
            ax.text(i, soma_1_2_3[i] + erros_graves[i]/2, f"{erros_graves[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=11)

    ax.set_title("Qualidade da Leitura: Distribuição de Erros por Placa", fontsize=14, fontweight='bold', pad=20, color='#111827')
    ax.set_ylabel("Proporção do Dataset de Teste (%)", fontsize=12, fontweight='bold', color='#374151')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    salvar_figura(fig, "grafico_4_distribuicao_qualidade_leitura.png")


def gerar_painel_executivo():
    """Gráfico 5: Dashboard Executivo 2x2 reunindo todos os indicadores."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11), facecolor='#F8FAFC')
    fig.suptitle("DASHBOARD EXECUTIVO: EVOLUÇÃO E BENCHMARK DO SISTEMA ALPR", fontsize=18, fontweight='bold', color='#0F172A', y=0.98)

    # 1. Comparativo Direto
    motores = ["YOLO", "EasyOCR", "Híbrido"]
    acertos = [7.0, 43.7, 95.8]
    cores1 = ["#EF4444", "#F59E0B", "#10B981"]
    barras1 = ax1.bar(motores, acertos, color=cores1, width=0.48, edgecolor='#333333', zorder=3)
    for b, val in zip(barras1, acertos):
        ax1.text(b.get_x() + b.get_width()/2, val + 2, f"{val:.1f}%", ha='center', fontweight='bold', fontsize=11)
    ax1.set_title("1. Comparativo de Precisão Exata", fontsize=12, fontweight='bold', color='#1E293B')
    ax1.set_ylabel("Acerto Exato (%)", fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 115)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # 2. Linha do Tempo da Evolução
    etapas_curtas = ["Baseline", "Deskew", "Contraste", "Morfologia", "Híbrido"]
    acuracia = [15.0, 38.0, 62.0, 78.0, 95.8]
    ax2.plot(range(5), acuracia, color='#2563EB', marker='s', markersize=8, linewidth=2.5, zorder=4)
    ax2.fill_between(range(5), acuracia, color='#BFDBFE', alpha=0.4, zorder=2)
    for i, v in enumerate(acuracia):
        ax2.text(i, v + 4, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=10, color='#1E3A8A')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(etapas_curtas, fontsize=9.5, fontweight='bold')
    ax2.set_title("2. Evolução por Etapa de Implementação", fontsize=12, fontweight='bold', color='#1E293B')
    ax2.set_ylabel("Acurácia Total (%)", fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 115)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 3. Ganho por Método
    tecnicas = ["Fusão Híbrida", "Desambiguação", "Multicontraste", "Deskew Hough"]
    ganhos = [17.8, 16.0, 24.0, 23.0]
    cores3 = ["#10B981", "#8B5CF6", "#3B82F6", "#F59E0B"]
    barras3 = ax3.barh(tecnicas, ganhos, color=cores3, height=0.5, edgecolor='#333333', zorder=3)
    for b, val in zip(barras3, ganhos):
        ax3.text(val + 0.5, b.get_y() + b.get_height()/2, f"+{val:.1f}%", ha='left', va='center', fontweight='bold', fontsize=10)
    ax3.set_title("3. Ganho Marginal por Solução Técnica", fontsize=12, fontweight='bold', color='#1E293B')
    ax3.set_xlabel("Ganho Adicionado (%)", fontsize=10, fontweight='bold')
    ax3.set_xlim(0, 30)
    ax3.grid(axis='x', linestyle='--', alpha=0.5)

    # 4. Distribuição de Erros
    categorias = ["YOLO", "EasyOCR", "Híbrido"]
    acerto_exato = [7.0, 43.7, 95.8]
    erros_1_2 = [32.4, 32.4, 0.0]
    erros_3_plus = [60.6, 23.9, 4.2]
    w = 0.48
    ax4.bar(categorias, acerto_exato, w, label="0 erros (Exato)", color='#10B981', edgecolor='#333333', zorder=3)
    ax4.bar(categorias, erros_1_2, w, bottom=acerto_exato, label="1-2 erros (Leve)", color='#FBBF24', edgecolor='#333333', zorder=3)
    soma = [a + b for a, b in zip(acerto_exato, erros_1_2)]
    ax4.bar(categorias, erros_3_plus, w, bottom=soma, label="3+ erros (Grave)", color='#EF4444', edgecolor='#333333', zorder=3)
    ax4.set_title("4. Distribuição da Severidade de Erros", fontsize=12, fontweight='bold', color='#1E293B')
    ax4.set_ylabel("Proporção (%)", fontsize=10, fontweight='bold')
    ax4.set_ylim(0, 115)
    ax4.legend(loc='upper right', fontsize=8.5, frameon=True, facecolor='white')
    ax4.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    salvar_figura(fig, "painel_executivo_evolucao_completo.png")


if __name__ == "__main__":
    print("Iniciando geração dos gráficos de evolução e comparativos...")
    gerar_grafico_comparativo_motores()
    gerar_grafico_evolucao_etapas()
    gerar_grafico_impacto_metodos()
    gerar_grafico_qualidade_leitura()
    gerar_painel_executivo()
    print("\nTodos os 5 gráficos foram gerados com sucesso em 'relatorios_benchmark/'!")
