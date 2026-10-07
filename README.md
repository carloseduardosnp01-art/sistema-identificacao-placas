# 🚗 Sistema de Identificação de Placas e Consulta de Veículos Roubados

Sistema completo de Reconhecimento Automático de Placas Veiculares (**ALPR/LPR**) com inteligência artificial, usando um **pipeline duplo YOLO** (detecção da placa + reconhecimento de caracteres) e consulta instantânea em base de dados de veículos com queixa de **roubo ou furto**.

---

## 👥 Integrantes

- Caio Felipe G. Lopes
- Davi Alves Mares
- Carlos Eduardo Pena Fiel Menon de Freitas

---

## 🏗️ Arquitetura do Sistema

```
[Foto / Imagem do Veículo]
           ↓
 1. Estágio 1 – Detecção da Placa (YOLO: models/detector_placas.pt)
           ↓ (Recorte da Bounding Box)
 2. Pré-processamento P&B de alto contraste (cinza + mediana + Otsu)
           ↓
 3. Estágio 2 – Reconhecimento de Caracteres (YOLO: models/detector_caracteres.pt, classes 0-9 / A-Z)
           ↓ (Ordenação, validação e correção: BRA2E19 ou ABC1234)
 4. Validação Regex (Mercosul/Antiga) & Consulta ao Banco SQLite
           ↓
 5. Interface Gráfica (Streamlit) com Alerta de Segurança (🚨 Roubado / ✅ Regular)
```

O módulo `src/ocr.py` (EasyOCR + desambiguação morfológica para a fonte FE-Schrift) permanece disponível como método alternativo.

---

## 📁 Estrutura do Projeto

```
sistema-identificacao-placas/
├── colab/                               # Notebooks de treinamento (Google Colab)
│   ├── treinamento_yolo_placas.ipynb            # Detector de placas (Open Images V7)
│   ├── treinamento_yolo_caracteres.ipynb        # Detector de caracteres
│   ├── treinamento_yolo_caracteres_ufpr.ipynb   # Caracteres com dataset UFPR
│   └── treinamento_yolo_ufpr_alpr.ipynb         # Pipeline UFPR-ALPR
├── src/
│   ├── detector.py                      # Estágio 1: detecção de placas com YOLO
│   ├── caracteres.py                    # Estágio 2: tratamento P&B e YOLO de caracteres
│   ├── ocr.py                           # OCR alternativo (EasyOCR + heurísticas)
│   ├── database.py                      # Conexão SQLite e consulta de veículos
│   ├── utils.py                         # Validação Regex, correção e renderização
│   └── app.py                           # Interface web com Streamlit
├── data/
│   └── veiculos.db                      # Banco SQLite de veículos e ocorrências
├── models/
│   ├── README.md                        # Instruções sobre os pesos
│   ├── detector_placas.pt               # Pesos do detector de placas
│   └── detector_caracteres.pt           # Pesos do detector de caracteres
├── tests/
│   ├── test_modulos.py                  # Testes unitários (validação e banco)
│   └── test_images/                     # Imagens de teste
├── train_caracteres.py                  # Treino local do YOLO de caracteres (Roboflow/local)
├── copiar_modelo.py                     # Copia pesos treinados para models/
├── calibrar_ocr.py                      # Calibração fina dos parâmetros do EasyOCR
├── baixar_benchmark.py / baixar_exemplos.py / extrair_amostras_benchmark.py
├── test_pipeline.py / test_segmentacao.py / test_deskew.py   # Scripts de teste do pipeline
├── iniciar_sistema.bat / rodar.bat      # Atalhos para iniciar a aplicação no Windows
├── requirements.txt                     # Dependências
└── README.md
```

---

## 🚀 Como Executar o Projeto

### 1. Instalação das Dependências
```bash
pip install -r requirements.txt
```

### 2. Treinamento dos Modelos
- **Detector de placas:** execute [`colab/treinamento_yolo_placas.ipynb`](colab/treinamento_yolo_placas.ipynb) no Google Colab (GPU T4) e salve o resultado em `models/detector_placas.pt`.
- **Detector de caracteres:** use um dos notebooks `colab/treinamento_yolo_caracteres*.ipynb` ou treine localmente:
  ```bash
  python train_caracteres.py
  ```
  Em seguida, `python copiar_modelo.py` copia o peso treinado para `models/detector_caracteres.pt`.

### 3. Executando a Aplicação Web
No Windows, basta dar duplo clique em `iniciar_sistema.bat` (ou `rodar.bat`). Manualmente:
```bash
streamlit run src/app.py
```
Acesse: `http://localhost:8501`.

### 4. Executando os Testes
```bash
python tests/test_modulos.py
python test_pipeline.py
```
