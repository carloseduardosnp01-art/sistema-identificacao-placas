# 🚗 Sistema de Identificação de Placas e Consulta de Veículos Roubados

Sistema completo de Reconhecimento Automático de Placas Veiculares (**ALPR/LPR**) com inteligência artificial, utilizando arquitetura web ultraleve e de alta performance (**FastAPI + Tailwind CSS**) e um **Pipeline Híbrido de Alto Desempenho** (Detecção YOLO + Reconhecimento Morfológico com Fusão de OCR e YOLO Caracteres), integrado à base de dados para consulta instantânea de veículos com queixa de **roubo ou furto**.

---

## 👥 Integrantes

- Caio Felipe G. Lopes
- Davi Alves Mares
- Carlos Eduardo Pena Fiel Menon de Freitas

---

## 🏗️ Arquitetura do Sistema

```
[Foto / Imagem do Veículo (Upload ou Câmera)]
           ↓
 1. Estágio 1 – Detecção das Placas (YOLO: models/detector_placas.pt)
           ↓ (Recorte de alta precisão da Bounding Box)
 2. Pré-processamento & Retificação (Deskew morfológico + 4 variantes de contraste)
           ↓
 3. Estágio 2 – Pipeline Híbrido de Caracteres (src/pipeline_misto.py)
           ├─ YOLO Caracteres (models/detector_caracteres.pt)
           ├─ EasyOCR especializado com desambiguação morfológica FE-Schrift
           └─ Votação por Confiança caractere a caractere com correção sintática
           ↓
 4. Validação Regex (Padrão Mercosul [ABC1D23] e Cinza Tradicional [ABC1234])
           ↓
 5. Consulta Instantânea ao Banco de Dados SQLite (data/veiculos.db)
           ↓
 6. Interface Web Moderna (FastAPI + Tailwind CSS) com Alerta de Segurança (🚨 Roubado / ✅ Regular)
```

---

## ⚡ Nova Interface Web Ultraleve (FastAPI + Tailwind CSS)

Substituindo frameworks monolíticos pesados que recarregavam todo o script a cada requisição, a nova base do sistema foi reconstruída sobre **FastAPI** e **Uvicorn**:
- **Consumo de Memória Reduzido:** Os modelos neurais permanecem carregados em memória na inicialização do servidor.
- **Inferência Instantânea:** Requisições assíncronas via `POST /api/detectar` com processamento em tempo real.
- **Design Moderno:** Interface responsiva com Glassmorphism, Tailwind CSS, badge de leitura estilizado no padrão Mercosul e feedback em tempo real.
- **Gestão de Banco de Dados Integrada:** Adicione e consulte veículos regulares ou roubados diretamente pela interface.

---

## 📊 Relatórios de Benchmark e Métricas de Precisão

O sistema foi validado em um teste cego de ponta a ponta sobre as 71 imagens reais do conjunto do Google Drive (fotos de tráfego, estacionamento, dia/noite e múltiplas placas):

| Métrica | Resultado Consolidado |
| :--- | :--- |
| **Total de Imagens Testadas** | 71 |
| **Total de Placas Esperadas (Gabarito)** | 85 placas |
| **Placas com Acerto Exato (100% dos 7 caracteres)** | **76 (89,41%)** |
| **Divergências / Placas ilegíveis no fundo** | 9 (10,59%) |
| **Tempo Médio por Imagem** | ~27 segundos |

Os relatórios detalhados e gráficos gerados estão disponíveis na pasta [`relatorios_benchmark/`](relatorios_benchmark/):
- **Planilha Completa Excel:** [`relatorios_benchmark/benchmark_placas_completo.xlsx`](relatorios_benchmark/benchmark_placas_completo.xlsx)
- **Planilha CSV:** [`relatorios_benchmark/benchmark_placas_completo.csv`](relatorios_benchmark/benchmark_placas_completo.csv)
- **Gráfico Comparativo de Desempenho:** [`relatorios_benchmark/benchmark_grafico_precisao.png`](relatorios_benchmark/benchmark_grafico_precisao.png)

---

## 📁 Estrutura do Projeto

```
sistema-identificacao-placas/
├── colab/                               # Notebooks de treinamento (Google Colab)
│   ├── treinamento_yolo_placas.ipynb            # Detector de placas (Open Images V7)
│   ├── treinamento_yolo_caracteres.ipynb        # Detector de caracteres
│   ├── treinamento_yolo_caracteres_ufpr.ipynb   # Caracteres com dataset UFPR
│   └── treinamento_yolo_ufpr_alpr.ipynb         # Pipeline UFPR-ALPR
├── data/
│   ├── fotos_drive/                     # Conjunto de 71 imagens reais para validação
│   └── veiculos.db                      # Banco SQLite de veículos e ocorrências
├── models/
│   ├── README.md                        # Instruções sobre os pesos
│   ├── detector_placas.pt               # Pesos do detector de placas
│   └── detector_caracteres.pt           # Pesos do detector de caracteres
├── relatorios_benchmark/                # Planilhas (.xlsx, .csv) e gráficos de acurácia
│   ├── benchmark_placas_completo.xlsx
│   ├── benchmark_placas_completo.csv
│   └── benchmark_grafico_precisao.png
├── scripts/
│   └── gerar_benchmark_completo.py      # Script de teste em lote e geração de gráficos
├── src/
│   ├── server.py                        # Servidor Web assíncrono FastAPI
│   ├── templates/index.html             # Interface web moderna com Tailwind CSS
│   ├── detector.py                      # Estágio 1: detecção de placas com YOLO
│   ├── caracteres.py                    # Estágio 2: tratamento e YOLO de caracteres
│   ├── pipeline_misto.py                # Pipeline Híbrido (YOLO + OCR ponderado)
│   ├── ocr.py                           # Módulo OCR alternativo
│   ├── database.py                      # Conexão SQLite e consulta de veículos
│   ├── utils.py                         # Validação Regex, correção e anotação visual
│   └── app.py                           # Interface legada em Streamlit
├── tests/
│   ├── test_modulos.py                  # Testes unitários (validação e banco)
│   └── test_images/                     # Imagens sintéticas de teste
├── iniciar_sistema.bat                  # Executável 1-clique para iniciar o servidor Web
├── rodar.bat                            # Atalho rápido para iniciar_sistema.bat
├── requirements.txt                     # Dependências do projeto
└── README.md
```

---

## 🚀 Como Executar o Projeto

### 1. Instalação das Dependências
Certifique-se de ter o Python 3.10+ instalado e execute:
```bash
pip install -r requirements.txt
```

### 2. Executando a Interface Web
No Windows, basta dar um duplo clique no arquivo [`iniciar_sistema.bat`](iniciar_sistema.bat) ou [`rodar.bat`](rodar.bat).

Ou execute manualmente via terminal:
```bash
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
```
Acesse no seu navegador: **`http://localhost:8000`**

### 3. Interface Legada (Streamlit)
Caso deseje rodar a versão anterior em Streamlit:
```bash
streamlit run src/app.py
```

### 4. Executando o Benchmark Completo
Para reproduzir os testes com todas as imagens e recalcular os gráficos e planilhas:
```bash
python scripts/gerar_benchmark_completo.py
```
Os novos relatórios serão salvos automaticamente na pasta `relatorios_benchmark/`.
