# 🚗 Sistema de Identificação de Placas e Consulta de Veículos Roubados

Sistema completo de Reconhecimento Automático de Placas Veiculares (**ALPR/LPR**) com inteligência artificial, detecção de placas com **YOLO**, leitura óptica de caracteres com **OCR** e consulta instantânea em base de dados de veículos com queixa de **roubo ou furto**.

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
 1. Detecção da Placa (YOLOv8 / YOLO11)
           ↓ (Recorte da Bounding Box)
 2. Pré-processamento (OpenCV: CLAHE + Filtro Bilateral + Otsu)
           ↓
 3. Extração e Correção de Caracteres (EasyOCR + Heurísticas)
           ↓ (Ex: BRA2E19 ou ABC1234)
 4. Validação & Consulta ao Banco SQLite
           ↓
 5. Interface Gráfica (Streamlit) com Alerta de Segurança (🚨 Roubado / ✅ Regular)
```

---

## 📁 Estrutura do Projeto

```
sistema-identificacao-placas/
├── colab/
│   └── treinamento_yolo_placas.ipynb   # Notebook completo para baixar Open Images V7 e treinar no Colab
├── src/
│   ├── __init__.py
│   ├── detector.py                      # Módulo de inferência com YOLO (models/best.pt)
│   ├── ocr.py                           # Pré-processamento OpenCV e leitura com EasyOCR
│   ├── database.py                      # Conexão SQLite e consulta de veículos
│   ├── utils.py                         # Validação Regex (Mercosul/Antiga) e renderização
│   └── app.py                           # Interface web interativa com Streamlit
├── data/
│   └── veiculos.db                      # Banco SQLite com dados de veículos e ocorrências
├── models/
│   ├── README.md                        # Instruções para colocar o best.pt
│   └── best.pt                          # Pesos treinados no Google Colab
├── tests/
│   ├── __init__.py
│   └── test_modulos.py                  # Testes unitários dos módulos de validação e banco
├── requirements.txt                     # Dependências do projeto
└── README.md
```

---

## 🚀 Como Executar o Projeto

### 1. Instalação das Dependências
No terminal do seu ambiente Python:
```bash
pip install -r requirements.txt
```

### 2. Treinamento do Modelo no Google Colab
1. Abra o arquivo [`colab/treinamento_yolo_placas.ipynb`](colab/treinamento_yolo_placas.ipynb) no [Google Colab](https://colab.research.google.com/).
2. Ative a GPU em **Ambiente de Execução > Alterar tipo de ambiente de execução > GPU T4**.
3. Execute todas as células. O notebook baixa as imagens do **Open Images Dataset V7** (classe `Vehicle registration plate`), treina o modelo YOLO e gera o arquivo `best.pt`.
4. Baixe o `best.pt` e coloque-o na pasta `models/best.pt` deste projeto.

### 3. Executando a Aplicação Web
Para iniciar a interface gráfica do Streamlit:
```bash
streamlit run src/app.py
```
Acesse no navegador: `http://localhost:8501`.

### 4. Executando os Testes
Para rodar os testes unitários:
```bash
python tests/test_modulos.py
```
