"""
Interface Web Interativa com Streamlit para o Sistema de Identificação de Placas.
Permite upload de fotos, detecção de placas com YOLO, extração via OCR e checagem de roubo/furto.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Adiciona o diretório raiz ao path do Python para importações relativas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.detector import DetectorPlacas, MODEL_PATH
from src.ocr import extrair_texto_placa
from src.database import (
    inicializar_banco,
    consultar_placa,
    cadastrar_veiculo,
    listar_veiculos,
    normalizar_placa
)
from src.utils import desenhar_resultado, formatar_placa_exibicao

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Sistema de Identificação de Placas",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializa o banco de dados se necessário
inicializar_banco()


@st.cache_resource
def carregar_detector():
    """Carrega o detector de placas em cache para não recarregar a cada interação."""
    return DetectorPlacas()


detector = carregar_detector()


# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.title("🚗 Sistema de Placas")
st.sidebar.markdown("---")

# Status dos Modelos YOLO (Pipeline de 2 Estágios)
st.sidebar.markdown("#### 🧠 Modelos de Deep Learning")
modelo_detector_existe = os.path.exists(MODEL_PATH)
if modelo_detector_existe:
    st.sidebar.success("✅ **YOLO 1 (Detector)**: `models/best.pt`")
else:
    st.sidebar.warning("⚠️ **YOLO 1 (Detector)**: Usando base (`yolov8n.pt`)")

caminho_yolo_char = os.path.join(BASE_DIR, "models", "detector_caracteres.pt")
modelo_char_existe = os.path.exists(caminho_yolo_char)
if modelo_char_existe:
    st.sidebar.success("✅ **YOLO 2 (Caracteres)**: `detector_caracteres.pt`")
else:
    st.sidebar.info("ℹ️ **YOLO 2 (Caracteres)**: Modo Fallback OCR Ativo (Pronto para receber `detector_caracteres.pt`)")

confianca_yolo = st.sidebar.slider(
    "Confiança Mínima do YOLO:",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
    help="Define o limite de sensibilidade do YOLO para detectar placas e caracteres."
)

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Sobre o Projeto")
st.sidebar.info(
    "Sistema de Visão Computacional baseado em YOLO para detecção de placas e "
    "leitura de caracteres Mercosul/Antigos com validação em base de veículos roubados."
)


# --- CORPO PRINCIPAL ---
st.title("🛡️ Sistema de Identificação de Placas e Consulta de Roubos")
st.markdown("Detecção em tempo real de placas veiculares com verificação automática de segurança.")

aba1, aba2, aba3 = st.tabs([
    "📸 Identificação por Foto",
    "🗄️ Base de Dados de Veículos",
    "🎓 Instruções de Treinamento (Colab)"
])


# --- ABA 1: IDENTIFICAÇÃO POR FOTO ---
with aba1:
    st.subheader("Análise de Placas e Consulta de Segurança")
    
    modo_entrada = st.radio(
        "Como deseja fornecer a imagem?",
        ["📁 Fazer upload de foto do computador", "🖼️ Escolher imagem da Galeria de Testes (Dataset)"],
        horizontal=True
    )

    img_pil = None

    if modo_entrada == "📁 Fazer upload de foto do computador":
        arquivo_imagem = st.file_uploader(
            "Selecione uma imagem (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"]
        )
        if arquivo_imagem is not None:
            img_pil = Image.open(arquivo_imagem).convert("RGB")
    else:
        # Busca imagens disponíveis na pasta de exemplos e test_images
        pastas_exemplos = [os.path.join(BASE_DIR, "data", "exemplos"), os.path.join(BASE_DIR, "data", "test_images")]
        arquivos_disponiveis = []
        for p in pastas_exemplos:
            if os.path.exists(p):
                for arq in os.listdir(p):
                    if arq.lower().endswith((".png", ".jpg", ".jpeg")) and not arq.startswith("rodosol_") and not arq.startswith("ufpr_"):
                        arquivos_disponiveis.append(os.path.join(p, arq))

        if arquivos_disponiveis:
            mapa_nomes = {os.path.basename(c): c for c in arquivos_disponiveis}
            opcao_selecionada = st.selectbox(
                "Selecione um veículo do banco de testes para analisar:",
                options=list(mapa_nomes.keys())
            )
            if opcao_selecionada:
                caminho_escolhido = mapa_nomes[opcao_selecionada]
                img_pil = Image.open(caminho_escolhido).convert("RGB")
        else:
            st.info("Nenhuma imagem encontrada na galeria.")

    if img_pil is not None:
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        col_orig, col_proc = st.columns([1, 1])

        with col_orig:
            st.image(img_pil, caption="Imagem do Veículo Selecionada", use_container_width=True)

        with st.spinner("Processando com YOLO e OCR..."):
            placas = detector.detectar(img_bgr, confianca_minima=confianca_yolo)

        if not placas:
            st.warning("⚠️ Nenhuma placa foi detectada na imagem com a confiança atual. Tente diminuir a sensibilidade na barra lateral.")
        else:
            img_resultado = img_bgr.copy()
            detalhes_placas = []

            for i, p in enumerate(placas):
                bbox = p["bbox"]
                crop = p["crop"]
                conf_yolo = p["confianca"]

                # Extrai texto com OCR
                resultado_ocr = extrair_texto_placa(crop)
                texto_placa = resultado_ocr["texto_corrigido"] or resultado_ocr["texto_bruto"]

                # Consulta no banco de dados
                dados_veiculo = consultar_placa(texto_placa) if texto_placa else None
                status_roubo = dados_veiculo["status_roubo"] if dados_veiculo else None

                # Desenha o resultado na imagem final
                img_resultado = desenhar_resultado(
                    img_resultado,
                    bbox,
                    texto_placa or "PLACA",
                    status_roubo,
                    dados_veiculo
                )

                detalhes_placas.append({
                    "indice": i + 1,
                    "crop": crop,
                    "img_proc": resultado_ocr["img_processada"],
                    "texto": texto_placa,
                    "padrao": resultado_ocr["padrao"],
                    "conf_ocr": resultado_ocr["confianca"],
                    "conf_yolo": conf_yolo,
                    "dados_veiculo": dados_veiculo
                })

            # Exibe imagem com as detecções desenhadas
            img_resultado_rgb = cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB)
            with col_proc:
                st.image(img_resultado_rgb, caption="Resultado do Processamento", use_container_width=True)

            st.markdown("---")
            st.subheader("📋 Relatório de Análise das Placas")

            for det in detalhes_placas:
                c1, c2, c3 = st.columns([1, 1, 2])

                with c1:
                    st.write(f"**Recorte da Placa #{det['indice']}**")
                    if det["crop"] is not None and det["crop"].size > 0:
                        st.image(cv2.cvtColor(det["crop"], cv2.COLOR_BGR2RGB), use_container_width=True)

                with c2:
                    st.write("**Pré-processamento OCR**")
                    if det["img_proc"] is not None and det["img_proc"].size > 0:
                        st.image(det["img_proc"], caption="Binarização / Filtros", use_container_width=True)

                with c3:
                    st.write("**Dados Identificados & Checagem:**")
                    texto = det["texto"]
                    dados = det["dados_veiculo"]

                    if not texto:
                        st.error("❌ Não foi possível extrair o texto dos caracteres da placa com nitidez.")
                        continue

                    st.markdown(f"### Placa: `{formatar_placa_exibicao(texto)}`")
                    st.write(f"- **Padrão:** {det['padrao'] or 'Não identificado'}")
                    st.write(f"- **Confiança YOLO:** {det['conf_yolo'] * 100:.1f}% | **Confiança OCR:** {det['conf_ocr'] * 100:.1f}%")

                    if dados:
                        if dados["status_roubo"] == 1:
                            st.markdown(f"""
                            <div style="background-color: #ff4b4b22; border-left: 5px solid #ff4b4b; padding: 12px; border-radius: 6px; margin-top: 10px;">
                                <h4 style="color: #ff4b4b; margin: 0 0 8px 0;">🚨 ALERTA: VEÍCULO ROUBADO / FURTADO!</h4>
                                <p style="margin: 3px 0;"><b>Veículo:</b> {dados['marca']} {dados['modelo']} ({dados['cor']} / {dados['ano']})</p>
                                <p style="margin: 3px 0;"><b>B.O.:</b> {dados.get('boletim_ocorrencia', 'BO-2024')}</p>
                                <p style="margin: 3px 0;"><b>Data:</b> {dados.get('data_ocorrencia', 'Não informada')}</p>
                                <p style="margin: 3px 0;"><b>Local:</b> {dados.get('cidade', '')} - {dados.get('estado', '')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background-color: #00cc6622; border-left: 5px solid #00cc66; padding: 12px; border-radius: 6px; margin-top: 10px;">
                                <h4 style="color: #00cc66; margin: 0 0 8px 0;">✅ SITUAÇÃO REGULAR: Sem queixa</h4>
                                <p style="margin: 3px 0;"><b>Veículo:</b> {dados['marca']} {dados['modelo']} ({dados['cor']} / {dados['ano']})</p>
                                <p style="margin: 3px 0;"><b>Local:</b> {dados.get('cidade', '')} - {dados.get('estado', '')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #ffaa0022; border-left: 5px solid #ffaa00; padding: 12px; border-radius: 6px; margin-top: 10px;">
                            <h5 style="color: #ffaa00; margin: 0;">⚠️ Placa <code>{formatar_placa_exibicao(texto)}</code> não cadastrada no banco de teste.</h5>
                        </div>
                        """, unsafe_allow_html=True)


# --- ABA 2: BASE DE DADOS DE VEÍCULOS ---
with aba2:
    st.subheader("Gerenciamento de Veículos e Registro de Roubos")

    col_form, col_tab = st.columns([1, 2])

    with col_form:
        st.markdown("#### Cadastrar / Atualizar Veículo")
        with st.form("form_cadastro_veiculo"):
            f_placa = st.text_input("Placa (ex: ABC1234 ou BRA2E19):").upper()
            f_marca = st.text_input("Marca:", value="Toyota")
            f_modelo = st.text_input("Modelo:", value="Corolla")
            f_cor = st.text_input("Cor:", value="Prata")
            f_ano = st.number_input("Ano:", min_value=1980, max_value=2026, value=2022)
            f_roubado = st.checkbox("Veículo com Queixa de Roubo/Furto?", value=False)
            f_bo = st.text_input("Boletim de Ocorrência (opcional):", value="BO-2024-00000")
            f_data = st.text_input("Data da Ocorrência (opcional):", value="2024-06-01")
            f_cidade = st.text_input("Cidade:", value="São Paulo")
            f_estado = st.text_input("Estado (UF):", value="SP", max_chars=2)

            btn_salvar = st.form_submit_button("Salvar no Banco")

            if btn_salvar:
                if not f_placa:
                    st.error("Informe a placa do veículo.")
                else:
                    sucesso = cadastrar_veiculo({
                        "placa": f_placa,
                        "marca": f_marca,
                        "modelo": f_modelo,
                        "cor": f_cor,
                        "ano": int(f_ano),
                        "status_roubo": 1 if f_roubado else 0,
                        "boletim_ocorrencia": f_bo if f_roubado else None,
                        "data_ocorrencia": f_data if f_roubado else None,
                        "cidade": f_cidade,
                        "estado": f_estado.upper()
                    })
                    if sucesso:
                        st.success(f"Veículo com placa {f_placa} cadastrado com sucesso!")
                        st.rerun()

    with col_tab:
        st.markdown("#### Veículos Registrados na Base")
        veiculos = listar_veiculos()
        if veiculos:
            df = pd.DataFrame(veiculos)
            # Formata status
            df["Status"] = df["status_roubo"].apply(lambda x: "🚨 ROUBADO" if x == 1 else "✅ REGULAR")
            colunas_exibir = ["placa", "Status", "marca", "modelo", "cor", "ano", "cidade", "estado", "boletim_ocorrencia"]
            st.dataframe(df[colunas_exibir], use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum veículo cadastrado no momento.")


# --- ABA 3: INSTRUÇÕES DE TREINAMENTO DOS MODELOS YOLO ---
with aba3:
    st.subheader("🎓 Treinamento dos Modelos de Deep Learning (YOLO)")
    st.markdown("Aprenda a treinar os dois estágios do pipeline com GPU no Google Colab ou diretamente no seu computador.")

    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "1️⃣ YOLO 1: Detector de Placas",
        "2️⃣ YOLO 2: Leitor de Caracteres (Roboflow)",
        "💻 Treinamento Local no PC"
    ])

    with sub_tab1:
        st.markdown("""
        ### 🎯 Estágio 1: Detector de Placas no Veículo
        * **Objetivo**: Achar e recortar a placa na cena completa (em qualquer ângulo e iluminação).
        * **Notebook**: [`colab/treinamento_yolo_placas.ipynb`](file:///c:/Users/Admin/Desktop/PROGRAMACAO/sistema-identificacao-placas/colab/treinamento_yolo_placas.ipynb)
        * **Dataset**: Open Images Dataset V7 (`Vehicle registration plate`).
        * **Arquivo Gerado**: `models/detector_placas.pt` (ou `models/best.pt`).
        """)

    with sub_tab2:
        st.markdown("""
        ### 🔤 Estágio 2: Leitor de Caracteres Mercosul & Antigos
        * **Objetivo**: Detectar individualmente cada uma das 36 classes de letras e números (`0-9`, `A-Z`) na fonte oficial FE-Schrift em cores naturais.
        * **Notebook**: [`colab/treinamento_yolo_caracteres.ipynb`](file:///c:/Users/Admin/Desktop/PROGRAMACAO/sistema-identificacao-placas/colab/treinamento_yolo_caracteres.ipynb)
        * **Dataset**: Roboflow Universe (Placas Brasil / Mercosul Caracteres).
        * **Arquivo Gerado**: Salve o modelo treinado como `models/detector_caracteres.pt`.
        """)

    with sub_tab3:
        st.markdown("""
        ### 🖥️ Treinando Diretamente no seu Computador
        Você também pode rodar o treinamento localmente no terminal:
        ```bash
        # Treinar com dataset do Roboflow
        python train_caracteres.py --roboflow_key SUA_CHAVE --roboflow_workspace placas-brasil --roboflow_project caracteres-mercosul --epochs 40

        # Ou treinar com dataset local já baixado
        python train_caracteres.py --data data/datasets_caracteres/data.yaml --epochs 40
        ```
        """)
