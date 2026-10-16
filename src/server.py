"""
Servidor Web Ultraleve com FastAPI e Uvicorn para o Sistema de Identificação de Placas.
Substitui o Streamlit oferecendo carregamento instantâneo, baixo consumo de memória
e interface visual moderna via Tailwind CSS / Glassmorphism.
"""

import os
import sys
import time
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Configurações de diretórios
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.detector import DetectorPlacas
from src.ocr_hibrido import executar_ocr_hibrido
from src.database import (
    inicializar_banco,
    consultar_placa,
    cadastrar_veiculo,
    listar_veiculos,
    normalizar_placa
)
from src.utils import desenhar_resultado

# Inicializa banco de dados SQLite
inicializar_banco()

app = FastAPI(title="VisionALPR API", description="Sistema de Identificação de Placas e Segurança Veicular")

# Configuração de templates e estáticos
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates = Jinja2Templates(directory=templates_dir)

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Carrega detector YOLO na inicialização do serviço
detector = DetectorPlacas()

class VeiculoInput(BaseModel):
    placa: str
    marca: str
    modelo: str
    cor: str
    ano: int
    status_roubo: int = 0
    data_ocorrencia: str = None
    boletim_ocorrencia: str = None
    cidade: str = None
    estado: str = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Renderiza a interface web moderna e ultraleve."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/detectar")
async def api_detectar(arquivo: UploadFile = File(...), modo: str = Form("hibrido")):
    """
    Recebe uma foto, executa detecção YOLO, OCR Híbrido (FastPlate + Paddle) e consulta de segurança.
    Modos suportados: 'hibrido' (padrão inteligente), 'live' (15ms ultra-rápido) ou 'deep' (auditoria cirúrgica 94%).
    """
    try:
        t_inicio = time.time()
        conteudo = await arquivo.read()
        nparr = np.frombuffer(conteudo, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Arquivo de imagem inválido.")

        # 1. Detecção YOLO da placa veicular
        placas_detectadas = detector.detectar(img, confianca_minima=0.20)
        
        resultados_placas = []
        img_anotada = img.copy()

        for p in placas_detectadas:
            crop = p["crop"]
            # 2. Executa o Motor Híbrido Unificado (FastPlate 15ms + PaddleOCR)
            res_ocr = executar_ocr_hibrido(crop, modo=modo)
            texto_placa = res_ocr.get("texto_corrigido") or ""
            confianca = res_ocr.get("confianca", 0.0)
            padrao = res_ocr.get("padrao", "Desconhecido")
            metodo = res_ocr.get("metodo", "Motor Híbrido")

            # Codifica crop em Base64 para exibição no frontend
            _, buffer_crop = cv2.imencode(".jpg", crop)
            crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer_crop).decode("utf-8")

            # Consulta no banco de dados de segurança
            info_veiculo = consultar_placa(texto_placa) if texto_placa else None
            status_roubo = (info_veiculo["status_roubo"] == 1) if info_veiculo else False

            p["texto"] = texto_placa
            p["info_veiculo"] = dict(info_veiculo) if info_veiculo else None
            p["padrao"] = padrao

            # Anota a bounding box na imagem completa
            bbox = p.get("bbox", (0, 0, img.shape[1], img.shape[0]))
            img_anotada = desenhar_resultado(
                img_anotada,
                bbox=bbox,
                texto_placa=texto_placa if texto_placa else "PLACA",
                status_roubo=status_roubo if info_veiculo else None,
                dados_veiculo=dict(info_veiculo) if info_veiculo else None
            )

            if texto_placa:
                resultados_placas.append({
                    "placa": texto_placa,
                    "padrao": padrao,
                    "metodo": metodo,
                    "confianca": float(confianca),
                    "crop_base64": crop_b64,
                    "info_veiculo": dict(info_veiculo) if info_veiculo else None
                })

        # Codifica imagem anotada em Base64
        _, buffer_anotada = cv2.imencode(".jpg", img_anotada, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_anotada_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer_anotada).decode("utf-8")

        tempo_total = time.time() - t_inicio

        return JSONResponse({
            "sucesso": True,
            "tempo_segundos": f"{tempo_total:.2f}",
            "total_placas": len(resultados_placas),
            "imagem_anotada": img_anotada_b64,
            "placas": resultados_placas
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"sucesso": False, "erro": str(e)}, status_code=500)

@app.post("/api/detectar_live")
async def api_detectar_live(arquivo: UploadFile = File(...)):
    """
    Endpoint otimizado para Streaming ao Vivo / Webcam (60 FPS).
    Executa YOLO + Fast-Plate-OCR em ~20-30ms e retorna apenas metadados e coordenadas de bbox.
    """
    try:
        t0 = time.perf_counter()
        conteudo = await arquivo.read()
        nparr = np.frombuffer(conteudo, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"sucesso": False, "erro": "Frame inválido"}, status_code=400)

        # Detecção rápida com YOLO
        placas_detectadas = detector.detectar(img, confianca_minima=0.25)
        lista_resultados = []

        for p in placas_detectadas:
            crop = p["crop"]
            # Modo live: Fast-Plate-OCR (~10-15ms)
            res_fast = executar_ocr_hibrido(crop, modo="live")
            texto_placa = res_fast.get("texto_corrigido") or ""
            
            if texto_placa:
                info_veiculo = consultar_placa(texto_placa)
                status_roubo = (info_veiculo["status_roubo"] == 1) if info_veiculo else False
                
                lista_resultados.append({
                    "bbox": [int(v) for v in p.get("bbox", [0, 0, 0, 0])],
                    "placa": texto_placa,
                    "padrao": res_fast.get("padrao", "Desconhecido"),
                    "confianca": float(round(res_fast.get("confianca", 0.0), 2)),
                    "status_roubo": status_roubo,
                    "veiculo": dict(info_veiculo) if info_veiculo else None
                })

        tempo_ms = (time.perf_counter() - t0) * 1000.0

        return JSONResponse({
            "sucesso": True,
            "tempo_ms": round(tempo_ms, 1),
            "placas": lista_resultados
        })

    except Exception as e:
        return JSONResponse({"sucesso": False, "erro": str(e)}, status_code=500)

@app.get("/api/veiculos")
async def api_listar_veiculos():
    """Retorna todos os veículos cadastrados na base SQLite."""
    veiculos = listar_veiculos()
    return {"veiculos": [dict(v) for v in veiculos]}

@app.post("/api/veiculos")
async def api_cadastrar_veiculo(veiculo: VeiculoInput):
    """Cadastra ou atualiza um veículo na base de dados de segurança."""
    sucesso = cadastrar_veiculo(
        placa=veiculo.placa,
        marca=veiculo.marca,
        modelo=veiculo.modelo,
        cor=veiculo.cor,
        ano=veiculo.ano,
        status_roubo=veiculo.status_roubo,
        data_ocorrencia=veiculo.data_ocorrencia,
        boletim_ocorrencia=veiculo.boletim_ocorrencia,
        cidade=veiculo.cidade,
        estado=veiculo.estado
    )
    if sucesso:
        return {"sucesso": True, "mensagem": f"Veículo {veiculo.placa} cadastrado com sucesso!"}
    else:
        raise HTTPException(status_code=500, detail="Erro ao salvar veículo no banco.")

if __name__ == "__main__":
    import uvicorn
    print("\n[VisionALPR] Iniciando servidor web ultraleve em http://127.0.0.1:8000")
    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=True)
