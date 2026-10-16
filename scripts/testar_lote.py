import os
import sys
import re
import cv2
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.detector import DetectorPlacas
from src.ocr import extrair_texto_placa

def extrair_placas_gabarito(nome_arquivo: str):
    """Extrai todas as placas presentes no nome do arquivo."""
    base = os.path.splitext(nome_arquivo)[0]
    # Remove prefixo numérico se houver (ex: '01_QCU5C70' -> 'QCU5C70')
    partes = base.split('_')
    placas = []
    for p in partes:
        # Placa válida tem 7 caracteres alfanuméricos
        p_limpa = re.sub(r'[^A-Z0-9]', '', p.upper())
        if len(p_limpa) == 7:
            placas.append(p_limpa)
    return placas

def calcular_distancia(s1, s2):
    """Calcula quantidade de caracteres diferentes entre duas strings de mesmo tamanho."""
    if len(s1) != len(s2):
        return max(len(s1), len(s2))
    return sum(1 for a, b in zip(s1, s2) if a != b)

def rodar_teste_lote(inicio: int = 1, fim: int = 10, pasta_fotos: str = "data/drive_fotos/Fotos placas"):
    detector = DetectorPlacas()
    arquivos = sorted([f for f in os.listdir(pasta_fotos) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Filtra pelo intervalo de índices
    lote_arquivos = []
    for f in arquivos:
        match = re.match(r'^(\d+)_', f)
        if match:
            idx = int(match.group(1))
            if inicio <= idx <= fim:
                lote_arquivos.append((idx, f))
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK PIPELINE ALPR - LOTE: Fotos {inicio:02d} até {fim:02d} ({len(lote_arquivos)} imagens)")
    print(f"{'='*70}\n")
    
    total_imagens = len(lote_arquivos)
    total_placas_gabarito = 0
    acertos_exatos = 0
    acertos_quase = 0 # 1 caractere de erro
    erros = 0
    
    resultados_detalhados = []
    
    for idx, nome_arq in lote_arquivos:
        caminho = os.path.join(pasta_fotos, nome_arq)
        gabarito = extrair_placas_gabarito(nome_arq)
        total_placas_gabarito += len(gabarito)
        
        img = cv2.imread(caminho)
        if img is None:
            print(f"[{idx:02d}] Erro ao carregar imagem: {nome_arq}")
            continue
            
        # 1. Detecção de placas
        placas_detectadas = detector.detectar(img, confianca_minima=0.20)
        
        predicoes = []
        metodos = []
        for p in placas_detectadas:
            res_ocr = extrair_texto_placa(p["crop"])
            txt = res_ocr.get("texto_corrigido") or ""
            metodo = res_ocr.get("metodo", "N/A")
            if txt:
                predicoes.append(txt)
                metodos.append(metodo)
                
        # Avaliação contra o gabarito
        status_geral = []
        for g in gabarito:
            if not predicoes:
                status_geral.append(f"Nenhuma placa detectada (esperado: {g})")
                erros += 1
            else:
                # Encontra a predição mais próxima
                melhor_pred = min(predicoes, key=lambda p: calcular_distancia(p, g))
                dist = calcular_distancia(melhor_pred, g)
                
                if dist == 0:
                    acertos_exatos += 1
                    status_geral.append(f"EXATO: {melhor_pred} == {g}")
                elif dist == 1:
                    acertos_quase += 1
                    status_geral.append(f"QUASE (1 char dif): {melhor_pred} vs {g}")
                else:
                    erros += 1
                    status_geral.append(f"DIVERGENTE ({dist} chars): {melhor_pred} vs {g}")
                    
        print(f"[{idx:02d}] {nome_arq}")
        print(f"     Gabarito: {gabarito}")
        print(f"     Predito:  {predicoes} | Métodos: {metodos}")
        print(f"     Status:   {', '.join(status_geral)}")
        print("-" * 70)
        
        resultados_detalhados.append({
            "idx": idx,
            "arquivo": nome_arq,
            "gabarito": gabarito,
            "predicoes": predicoes,
            "metodos": metodos,
            "status": status_geral
        })
        
    print(f"\n{'='*70}")
    print(f"RESUMO DO LOTE ({inicio:02d} a {fim:02d})")
    print(f"Total Imagens: {total_imagens}")
    print(f"Total Placas Esperadas: {total_placas_gabarito}")
    print(f"Acertos Exatos (100%):  {acertos_exatos} ({acertos_exatos/max(1, total_placas_gabarito)*100:.1f}%)")
    print(f"Acertos Parciais (1 char dif): {acertos_quase}")
    print(f"Erros/Divergentes:      {erros}")
    print(f"{'='*70}\n")
    return resultados_detalhados

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testar lote de fotos ALPR")
    parser.add_argument("--inicio", type=int, default=1, help="Número inicial da foto")
    parser.add_argument("--fim", type=int, default=10, help="Número final da foto")
    args = parser.parse_args()
    
    rodar_teste_lote(args.inicio, args.fim)
