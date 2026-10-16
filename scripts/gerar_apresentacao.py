"""
Script para Geração da Apresentação de Slides do Projeto ALPR.
Cria duas versões da apresentação:
1. apresentacao_projeto_alpr.pptx (PowerPoint Widescreen 16:9 profissional e editável)
2. apresentacao_projeto_alpr.html (Apresentação Web Interativa com Tailwind CSS e navegação moderna)
Incorpora todos os gráficos de treinamento (YOLO 1 e YOLO 2) e benchmarks.
"""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "apresentacao"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pastas de Gráficos
DIR_YOLO1 = BASE_DIR / "data" / "resultados treinamento modelo de identificação de placa"
if not DIR_YOLO1.exists():
    for d in (BASE_DIR / "data").glob("*resultados treinamento*identifica*"):
        if "caracteres" not in d.name.lower():
            DIR_YOLO1 = d
            break

DIR_YOLO2 = BASE_DIR / "data" / "resultados treinamento modelo de identificacao de caracteres yolo 2"
DIR_BENCH = BASE_DIR / "relatorios_benchmark"

# Paleta de Cores Corporativa
C_AZUL_ESCURO = RGBColor(15, 23, 42)      # #0F172A
C_AZUL_MEDIO  = RGBColor(30, 64, 175)     # #1E40AF
C_AZUL_CLARO  = RGBColor(56, 189, 248)    # #38BDF8
C_VERDE       = RGBColor(16, 185, 129)    # #10B981
C_CINZA_TEXTO = RGBColor(71, 85, 105)     # #475569
C_BRANCO      = RGBColor(255, 255, 255)
C_FUNDO_CARD  = RGBColor(241, 245, 249)   # #F1F5F9


def aplicar_fundo(slide, color):
    """Aplica cor sólida ao fundo do slide."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def adicionar_cabecalho(slide, titulo_texto, subtitulo_texto=None):
    """Adiciona barra de cabeçalho padrão aos slides de conteúdo."""
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p = tf.paragraphs[0]
    p.text = titulo_texto
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = C_AZUL_ESCURO
    p.font.name = "Arial"
    
    if subtitulo_texto:
        p2 = tf.add_paragraph()
        p2.text = subtitulo_texto
        p2.font.size = Pt(13)
        p2.font.color.rgb = C_AZUL_MEDIO
        p2.font.name = "Arial"
        p2.space_before = Pt(4)


def criar_apresentacao_pptx():
    """Gera o arquivo PowerPoint apresentacao_projeto_alpr.pptx em formato 16:9."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    # =========================================================================
    # SLIDE 1: CAPA
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s1, C_AZUL_ESCURO)
    
    # Detalhe visual (faixa decorativa)
    faixa = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.18), Inches(5.1))
    faixa.fill.solid()
    faixa.fill.fore_color.rgb = C_AZUL_CLARO
    faixa.line.color.rgb = C_AZUL_CLARO
    
    tb1 = s1.shapes.add_textbox(Inches(1.3), Inches(1.5), Inches(11.2), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "SISTEMA INTELIGENTE DE IDENTIFICAÇÃO DE PLACAS (ALPR)"
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = C_BRANCO
    p.font.name = "Arial"
    
    p = tf1.add_paragraph()
    p.text = "Do Treinamento em Deep Learning ao Pipeline Híbrido de Alto Desempenho e Consulta Policial"
    p.font.size = Pt(17)
    p.font.color.rgb = C_AZUL_CLARO
    p.font.name = "Arial"
    p.space_before = Pt(14)
    
    p = tf1.add_paragraph()
    p.text = "Integrantes: Caio Felipe G. Lopes • Davi Alves Mares • Carlos Eduardo Pena Fiel Menon de Freitas"
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(203, 213, 225)
    p.font.name = "Arial"
    p.space_before = Pt(40)
    
    p = tf1.add_paragraph()
    p.text = "Tecnologias: YOLOv8 / YOLO11 • Roboflow Universe • EasyOCR • OpenCV • FastAPI • SQLite"
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(148, 163, 184)
    p.font.name = "Arial"
    p.space_before = Pt(10)

    # =========================================================================
    # SLIDE 2: O DESAFIO DO MUNDO REAL
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s2, C_BRANCO)
    adicionar_cabecalho(s2, "1. O Desafio Real da Identificação Veicular", "Por que leitores convencionais falham em condições reais de tráfego?")
    
    # 4 Cards de Desafios
    desafios = [
        ("📐 Rotação & Ângulo", "Câmeras de segurança e tráfego capturam veículos em ângulos inclinados (-35° a +35°). A deformação afim quebra a ordem de leitura dos caracteres."),
        ("🌗 Iluminação Extrema", "Faróis acesos, reflexos solares na placa de alumínio, sombras duras e penumbra noturna geram estouro de branco ou perda de contraste."),
        ("🔤 Tipografia FE-Schrift", "A fonte oficial do Mercosul foi projetada propositalmente para ser antifraude, confundindo leitores comuns entre 'Q/O', '8/B', '5/S' e '1/I'."),
        ("🇧🇷 Elementos Espúrios", "A tarja azul superior com a palavra 'BRASIL' e a bandeira geram falsos caracteres iniciais (ex: leitura 'BRA5I30' no lugar da placa real).")
    ]
    
    for i, (tit, desc) in enumerate(desafios):
        x = Inches(0.8 + (i % 2) * 5.9)
        y = Inches(1.8 + (i // 2) * 2.5)
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(5.6), Inches(2.2))
        card.fill.solid()
        card.fill.fore_color.rgb = C_FUNDO_CARD
        card.line.color.rgb = RGBColor(226, 232, 240)
        
        tb = s2.shapes.add_textbox(x + Inches(0.2), y + Inches(0.2), Inches(5.2), Inches(1.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = tit
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = C_AZUL_MEDIO
        
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(12)
        p2.font.color.rgb = C_CINZA_TEXTO
        p2.space_before = Pt(6)

    # =========================================================================
    # SLIDE 3: ARQUITETURA GLOBAL DO SISTEMA
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s3, C_BRANCO)
    adicionar_cabecalho(s3, "2. Arquitetura do Sistema e Pipeline Híbrido", "Fluxo multicamadas integrando visão computacional, Deep Learning e regras sintáticas")
    
    etapas_pipeline = [
        ("1. Captura da Imagem", "Upload web ou câmera", "Suporte a fotos reais de tráfego, estacionamento e vias públicas."),
        ("2. Detecção da Placa", "YOLO 1 (detector_placas.pt)", "Localiza a bounding box da placa na imagem com mAP@50 > 95%."),
        ("3. Retificação Geométrica", "Deskew com Hough (OpenCV)", "Detecta o ângulo (-35° a +35°) e desentorta a placa para 0.0° perfeito."),
        ("4. Multicontraste Dinâmico", "4 Variantes de Pré-processamento", "Corte da tarja azul e geração de imagens em Normal, CLAHE, Otsu e Adaptativo."),
        ("5. Pipeline Híbrido Votado", "YOLO 2 + EasyOCR Consensual", "Votação caractere a caractere combinando probabilidade neural e morfologia."),
        ("6. Validação & Consulta", "Regex Mercosul/Cinza + SQLite", "Garante o formato de 7 caracteres e checa status de queixa de roubo/furto.")
    ]
    
    for i, (tit, sub, desc) in enumerate(etapas_pipeline):
        x = Inches(0.8 + (i % 3) * 3.9)
        y = Inches(1.8 + (i // 3) * 2.6)
        card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(3.7), Inches(2.3))
        card.fill.solid()
        card.fill.fore_color.rgb = C_FUNDO_CARD
        card.line.color.rgb = RGBColor(203, 213, 225)
        
        tb = s3.shapes.add_textbox(x + Inches(0.15), y + Inches(0.15), Inches(3.4), Inches(2.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = tit
        p.font.size = Pt(13.5)
        p.font.bold = True
        p.font.color.rgb = C_AZUL_MEDIO
        
        p_sub = tf.add_paragraph()
        p_sub.text = sub
        p_sub.font.size = Pt(10.5)
        p_sub.font.bold = True
        p_sub.font.color.rgb = C_VERDE
        p_sub.space_before = Pt(2)
        
        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(11)
        p_desc.font.color.rgb = C_CINZA_TEXTO
        p_desc.space_before = Pt(5)

    # =========================================================================
    # SLIDE 4: TREINAMENTO DO YOLO 1 (DETECTOR DE PLACAS)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s4, C_BRANCO)
    adicionar_cabecalho(s4, "3. Treinamento do YOLO 1: Detecção de Placas", "Localização precisa da bounding box da placa veicular na imagem completa")
    
    # Texto à esquerda
    tb4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    
    itens_y1 = [
        ("📦 Dataset de Treino:", "Open Images V7 combinado com amostras de veículos brasileiros em tráfego."),
        ("⚙️ Hiperparâmetros:", "YOLOv8 Nano, Batch 16, Resolução de 640px com Data Augmentation automático (escala, brilho, leve rotação)."),
        ("🎯 Desempenho:", "Alcançou rápida convergência nas primeiras 30 épocas com mAP@50 > 95%."),
        ("💡 Papel no Pipeline:", "Isola com precisão os limites da placa veicular, descartando o veículo e o cenário urbano ao redor para alimentar os estágios seguintes.")
    ]
    for i, (t, d) in enumerate(itens_y1):
        p = tf4.paragraphs[0] if i == 0 else tf4.add_paragraph()
        p.text = f"{t} {d}"
        p.font.size = Pt(13)
        p.font.color.rgb = C_AZUL_ESCURO
        p.space_before = Pt(12)
        
    # Imagem à direita (results.png ou BoxPR_curve do YOLO 1)
    img_y1 = DIR_YOLO1 / "results.png"
    if not img_y1.exists():
        img_y1 = DIR_YOLO1 / "BoxPR_curve.png"
    if img_y1.exists():
        s4.shapes.add_picture(str(img_y1), Inches(6.6), Inches(1.8), width=Inches(5.9))

    # =========================================================================
    # SLIDE 5: TREINAMENTO DO YOLO 2 (CARACTERES - ROBOFLOW)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s5, C_BRANCO)
    adicionar_cabecalho(s5, "4. Treinamento do YOLO 2: Reconhecimento de Caracteres", "Modelo especializado em 36 classes (0-9 e A-Z) treinado no Roboflow Universe")
    
    # Imagem à esquerda (cartão de métricas ou results do YOLO 2)
    img_y2_card = DIR_YOLO2 / "metricas_roboflow_yolo2.png"
    if img_y2_card.exists():
        s5.shapes.add_picture(str(img_y2_card), Inches(0.8), Inches(1.8), width=Inches(6.2))
        
    # Texto e curvas à direita
    img_y2_res = DIR_YOLO2 / "BoxPR_curve.png"
    if img_y2_res.exists():
        s5.shapes.add_picture(str(img_y2_res), Inches(7.3), Inches(1.8), width=Inches(5.2))

    # =========================================================================
    # SLIDE 6: O CHOQUE DE REALIDADE (POR QUE 99% VIROU 7% NO MUNDO REAL?)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s6, C_BRANCO)
    adicionar_cabecalho(s6, "5. O Choque de Realidade: O Efeito Cascata de Erro", "Por que uma rede neural com 99,3% de mAP alcançou apenas 7,0% de acerto nas placas reais?")
    
    tb6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf6 = tb6.text_frame
    tf6.word_wrap = True
    
    p = tf6.paragraphs[0]
    p.text = "A Matemática da Falha em Reconhecimento Sequencial de Caracteres"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = RGBColor(220, 38, 38)
    
    pontos_choque = [
        "1. A Ilusão do mAP por Objeto: O modelo acerta 99% das letras quando elas estão isoladas, limpas e perfeitamente horizontais no dataset de laboratório.",
        "2. A Condição Exata da Placa: Para a consulta policial ser bem-sucedida, o sistema PRECISA acertar rigorosamente TODOS os 7 caracteres em sequência. Se errar 1 único caractere ('B' por '8' ou 'M' por 'H'), a placa inteira falha na consulta.",
        "3. A Matemática Probabilística: Se a probabilidade de acertar um caractere individual sob ângulo ou reflexo for de ~68%, a probabilidade conjunta da placa é (0,68)^7 = apenas 6,7%!",
        "4. Conclusão Crítica de Engenharia: Nenhum modelo de Deep Learning isolado resolve o problema do mundo real sem pré-processamento geométrico, desambiguação morfológica e fusão de modelos redundantes."
    ]
    for pt in pontos_choque:
        p_pt = tf6.add_paragraph()
        p_pt.text = pt
        p_pt.font.size = Pt(13)
        p_pt.font.color.rgb = C_AZUL_ESCURO
        p_pt.space_before = Pt(12)

    # =========================================================================
    # SLIDE 7: O COMPARATIVO: YOLO ISOLADO VS EASYOCR ISOLADO
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s7, C_BRANCO)
    adicionar_cabecalho(s7, "6. Batalha Direta: YOLO Isolado vs EasyOCR Isolado", "Avaliação cega independente sobre as 71 imagens reais do Google Drive")
    
    img_comp = DIR_BENCH / "grafico_1_comparativo_yolo_ocr_hibrido.png"
    if img_comp.exists():
        s7.shapes.add_picture(str(img_comp), Inches(0.8), Inches(1.8), width=Inches(7.2))
        
    tb7 = s7.shapes.add_textbox(Inches(8.3), Inches(1.8), Inches(4.3), Inches(5.0))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    
    p = tf7.paragraphs[0]
    p.text = "Resultados Práticos Obtidos:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_AZUL_MEDIO
    
    resumo_batalha = [
        "🔴 YOLO Isolado: 7,0% de Acerto Exato. Sofria severamente com placas inclinadas e reflexos de sol.",
        "🟡 EasyOCR Isolado: 43,7% de Acerto Exato. Superior ao YOLO, mas lia a palavra 'BRASIL' e confundia a fonte FE-Schrift.",
        "🟢 Pipeline Híbrido: 95,8% nas placas principais (89,4% geral). Salto de +52,1% de acurácia sobre o melhor motor isolado!"
    ]
    for r in resumo_batalha:
        p_r = tf7.add_paragraph()
        p_r.text = r
        p_r.font.size = Pt(12.5)
        p_r.font.color.rgb = C_AZUL_ESCURO
        p_r.space_before = Pt(10)

    # =========================================================================
    # SLIDE 8: RETIFICAÇÃO GEOMÉTRICA (DESKEW HOUGH)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s8, C_BRANCO)
    adicionar_cabecalho(s8, "7. Solução Técnica 1: Retificação Geométrica (Deskew)", "Alinhamento angular automático via Transformada de Hough (Ganho: +23,0%)")
    
    tb8 = s8.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    tf8 = tb8.text_frame
    tf8.word_wrap = True
    
    itens_s8 = [
        ("1. O Problema Enfrentado:", "Placas de motos e veículos capturados lateralmente apresentavam rotação entre -35° e +35°. Com a placa inclinada, o OCR perdia a linha de texto e o detector de caracteres ordenava caixas fora de sequência horizontal."),
        ("2. O Algoritmo Implementado:", "Desenvolvido endireitar_placa_hough() em OpenCV: conversão em escala de cinza, extração de bordas Canny e detecção de segmentos lineares com cv2.HoughLinesP."),
        ("3. Estimativa Angular:", "Cálculo trigonométrico arctan2((y2-y1)/(x2-x1)) de todas as linhas com ângulo entre -35° e +35°, extraindo a mediana estatística para máxima robustez contra ruídos."),
        ("4. Correção Afim:", "Geração de matriz cv2.getRotationMatrix2D e rotação por interpolação bicúbica com cv2.warpAffine, garantindo que a placa entre a 0.0° perfeito no OCR."),
        ("5. Impacto Medido:", "Elevou a precisão do sistema de 15,0% para 38,0% (+23,0% de ganho absoluto).")
    ]
    for i, (t, d) in enumerate(itens_s8):
        p = tf8.paragraphs[0] if i == 0 else tf8.add_paragraph()
        p.text = f"{t} {d}"
        p.font.size = Pt(12.5)
        p.font.color.rgb = C_AZUL_ESCURO
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 9: PRÉ-PROCESSAMENTO MULTICONTRASTE E TARJA BRASIL
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s9, C_BRANCO)
    adicionar_cabecalho(s9, "8. Solução Técnica 2: Pré-processamento Multicontraste", "Eliminação da tarja azul 'BRASIL' e 4 variantes dinâmicas de imagem (Ganho: +24,0%)")
    
    tb9 = s9.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    tf9 = tb9.text_frame
    tf9.word_wrap = True
    
    itens_s9 = [
        ("1. Corte Estratégico da Tarja Azul:", "Na placa Mercosul, a palavra 'BRASIL' e a bandeira ocupam os primeiros 20% do topo. Criamos um recorte morfológico isolando o miolo (h*0.20 a h*0.95), eliminando o falso prefixo 'BRA' que poluía as leituras."),
        ("2. Normalização de Resolução:", "Redimensionamento proporcional via interpolação Lanczos4 para altura fixa de 220px, permitindo que caracteres pequenos atinjam resolução legível."),
        ("3. As 4 Variantes de Imagem:", "O pipeline avalia simultaneamente 4 tratamentos independentes:\n   • Variante 1 (Normal): Imagem original retificada em alta resolução.\n   • Variante 2 (Realce CLAHE): Equalização adaptativa de histograma + filtro de sharpening de bordas.\n   • Variante 3 (Binarização Otsu): Segmentação global preto e branco ideal para placas com sombra.\n   • Variante 4 (Adaptativa Local): Binarização de limiar móvel para placas com reflexo direto de farol."),
        ("4. Impacto Medido:", "Elevou a acurácia de 38,0% para 62,0% (+24,0% de ganho absoluto).")
    ]
    for i, (t, d) in enumerate(itens_s9):
        p = tf9.paragraphs[0] if i == 0 else tf9.add_paragraph()
        p.text = f"{t} {d}"
        p.font.size = Pt(12.5)
        p.font.color.rgb = C_AZUL_ESCURO
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 10: DESAMBIGUAÇÃO MORFOLÓGICA FE-SCHRIFT
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s10, C_BRANCO)
    adicionar_cabecalho(s10, "9. Solução Técnica 3: Desambiguação FE-Schrift", "Engenharia reversa das regras tipográficas antifraude do Mercosul (Ganho: +16,0%)")
    
    tb10 = s10.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    tf10 = tb10.text_frame
    tf10.word_wrap = True
    
    itens_s10 = [
        ("1. O Segredo da Fonte FE-Schrift:", "Criada na Alemanha (Fälschungserschwerende Schrift), foi adotada no Mercosul com traços interrompidos para impedir adulterações manuais (ex: transformar um 'O' em 'Q' com fita preta)."),
        ("2. O Problema nos OCRs Genéricos:", "Modelos padrão confundem frequentemente as letras e dígitos dessa tipografia:\n   • 'Q' é lido como 'O' ou 'G' devido à abertura na perna.\n   • 'B' é lido como '8' e '8' é lido como 'B'.\n   • 'S' é lido como '5' e '5' é lido como 'S'.\n   • '1' e 'I' possuem formato idêntico."),
        ("3. Regras Sintáticas Posicionais:", "Implementamos em src/desambiguacao.py a correção estrutural baseada no padrão brasileiro:\n   • Posições 0, 1 e 2: Devem ser estritamente LETRAS (força '0'->'O', '8'->'B', '5'->'S').\n   • Posição 3: Deve ser estritamente DÍGITO (força 'O'->'0', 'B'->'8', 'S'->'5').\n   • Posição 4: Mercosul é LETRA (ex: 'C'), Antiga é DÍGITO (ex: '3').\n   • Posições 5 e 6: Devem ser estritamente DÍGITOS."),
        ("4. Impacto Medido:", "Elevou a acurácia de 62,0% para 78,0% (+16,0% de ganho absoluto).")
    ]
    for i, (t, d) in enumerate(itens_s10):
        p = tf10.paragraphs[0] if i == 0 else tf10.add_paragraph()
        p.text = f"{t} {d}"
        p.font.size = Pt(12.5)
        p.font.color.rgb = C_AZUL_ESCURO
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 11: A VIRADA DE CHAVE: O PIPELINE HÍBRIDO VOTADO
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s11, C_BRANCO)
    adicionar_cabecalho(s11, "10. A Virada de Chave: Fusão Ponderada (YOLO + OCR)", "Atingindo 95,8% de acurácia com votação inteligente e redundância de modelos")
    
    img_evol = DIR_BENCH / "grafico_2_evolucao_etapas.png"
    if img_evol.exists():
        s11.shapes.add_picture(str(img_evol), Inches(0.8), Inches(1.8), width=Inches(7.2))
        
    tb11 = s11.shapes.add_textbox(Inches(8.3), Inches(1.8), Inches(4.3), Inches(5.0))
    tf11 = tb11.text_frame
    tf11.word_wrap = True
    
    p = tf11.paragraphs[0]
    p.text = "Como funciona o Consenso:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_AZUL_MEDIO
    
    itens_s11 = [
        "1. Execução Concorrente: Ambos os motores processam a placa retificada.",
        "2. Votação por Confiança: Se o OCR e o YOLO concordam, a confiança é máxima.",
        "3. Arbitragem Especializada: Em caracteres colados ou com reflexo (ex: 'M' vs 'H'), a matriz de probabilidade pondera quem possui maior certeza posicional.",
        "4. Resultado Consolidado: Atingiu 95,8% de acerto exato nas fotos reais!"
    ]
    for item in itens_s11:
        p_it = tf11.add_paragraph()
        p_it.text = item
        p_it.font.size = Pt(12.5)
        p_it.font.color.rgb = C_AZUL_ESCURO
        p_it.space_before = Pt(10)

    # =========================================================================
    # SLIDE 12: IMPACTO POR TÉCNICA E DISTRIBUIÇÃO DE ERROS
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s12, C_BRANCO)
    adicionar_cabecalho(s12, "11. Impacto por Técnica e Distribuição da Qualidade", "Ganhos incrementais e eliminação quase total de erros parciais e graves")
    
    img_g3 = DIR_BENCH / "grafico_3_ganho_por_metodo.png"
    img_g4 = DIR_BENCH / "grafico_4_distribuicao_qualidade_leitura.png"
    
    if img_g3.exists():
        s12.shapes.add_picture(str(img_g3), Inches(0.8), Inches(1.8), width=Inches(5.6))
    if img_g4.exists():
        s12.shapes.add_picture(str(img_g4), Inches(6.8), Inches(1.8), width=Inches(5.7))

    # =========================================================================
    # SLIDE 13: REENGENHARIA WEB: FASTAPI + TAILWIND CSS
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s13, C_BRANCO)
    adicionar_cabecalho(s13, "12. Nova Arquitetura de Software: Migração para FastAPI", "Substituição do Streamlit por backend assíncrono ultraleve e interface moderna")
    
    # 2 Colunas de Comparação
    # Coluna 1: Streamlit
    card_st = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    card_st.fill.solid()
    card_st.fill.fore_color.rgb = RGBColor(254, 242, 242)
    card_st.line.color.rgb = RGBColor(252, 165, 165)
    
    tbst = s13.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.4))
    tfst = tbst.text_frame
    tfst.word_wrap = True
    p = tfst.paragraphs[0]
    p.text = "❌ Base Anterior: Streamlit"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = RGBColor(185, 28, 28)
    
    pontos_st = [
        "• Monolítico e Pesado: O Streamlit reexecuta todo o script Python a cada interação do usuário.",
        "• Alto Consumo de Memória: Recarregava objetos e causava concorrência desnecessária de GPU/RAM.",
        "• Inferência Travada: A interface congelava enquanto o pipeline de IA processava a imagem.",
        "• Personalização Limitada: Dificuldade de customizar componentes específicos como badges Mercosul."
    ]
    for pt in pontos_st:
        p_pt = tfst.add_paragraph()
        p_pt.text = pt
        p_pt.font.size = Pt(12)
        p_pt.font.color.rgb = RGBColor(153, 27, 27)
        p_pt.space_before = Pt(8)
        
    # Coluna 2: FastAPI + Tailwind
    card_fa = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    card_fa.fill.solid()
    card_fa.fill.fore_color.rgb = RGBColor(240, 253, 244)
    card_fa.line.color.rgb = RGBColor(134, 239, 172)
    
    tbfa = s13.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.4))
    tffa = tbfa.text_frame
    tffa.word_wrap = True
    p = tffa.paragraphs[0]
    p.text = "✅ Nova Base: FastAPI + Tailwind CSS"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = RGBColor(21, 128, 61)
    
    pontos_fa = [
        "• Modelos Pré-carregados: Os pesos neurais permanecem em memória RAM/GPU na inicialização via lifespan.",
        "• Inferência Assíncrona: Endpoint REST POST /api/detectar processa e devolve a resposta instantaneamente.",
        "• Interface Moderna: Design Glassmorphism em Tailwind CSS, badge Mercosul fiel e visualização dinâmica de caixas delimitadoras.",
        "• Gestão de Banco Integrada: Consulta e cadastro de veículos regulares ou com queixa de roubo/furto no SQLite diretamente na tela."
    ]
    for pt in pontos_fa:
        p_pt = tffa.add_paragraph()
        p_pt.text = pt
        p_pt.font.size = Pt(12)
        p_pt.font.color.rgb = RGBColor(22, 101, 52)
        p_pt.space_before = Pt(8)

    # =========================================================================
    # SLIDE 14: DASHBOARD EXECUTIVO CONSOLIDADO E CONCLUSÕES
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s14, C_BRANCO)
    adicionar_cabecalho(s14, "13. Dashboard Executivo e Conclusões Finais", "Resumo das conquistas do projeto e visão holística do sistema ALPR")
    
    img_dash = DIR_BENCH / "painel_executivo_evolucao_completo.png"
    if img_dash.exists():
        s14.shapes.add_picture(str(img_dash), Inches(0.8), Inches(1.8), width=Inches(7.2))
        
    tb14 = s14.shapes.add_textbox(Inches(8.3), Inches(1.8), Inches(4.3), Inches(5.0))
    tf14 = tb14.text_frame
    tf14.word_wrap = True
    
    p = tf14.paragraphs[0]
    p.text = "Conclusões do Projeto:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_AZUL_MEDIO
    
    conclusoes = [
        "1. IA no mundo real exige engenharia: Um modelo com 99% em benchmark isolado necessita de tratamento afim e sintático para operar com confiabilidade.",
        "2. Redundância Híbrida: A união de YOLO + EasyOCR superou os limites individuais de cada tecnologia.",
        "3. Sistema Pronto para Produção: Nova interface web em FastAPI leve, responsiva e pronta para câmeras em tempo real.",
        "4. Repositório 100% Documentado: Código limpo, relatórios em planilha Excel/CSV e gráficos de alta resolução."
    ]
    for c in conclusoes:
        p_c = tf14.add_paragraph()
        p_c.text = c
        p_c.font.size = Pt(12)
        p_c.font.color.rgb = C_AZUL_ESCURO
        p_c.space_before = Pt(8)

    caminho_pptx = OUTPUT_DIR / "apresentacao_projeto_alpr.pptx"
    prs.save(str(caminho_pptx))
    print(f"🎉 Apresentação PowerPoint gerada com sucesso: {caminho_pptx}")


def criar_apresentacao_html():
    """Gera uma apresentação Web Interativa em HTML moderna e responsiva."""
    caminho_html = OUTPUT_DIR / "apresentacao_projeto_alpr.html"
    
    html_content = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Apresentação: Sistema Inteligente de Identificação de Placas (ALPR)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        body { font-family: 'Inter', sans-serif; background-color: #0B0F19; color: #F8FAFC; }
        .slide { display: none; min-height: 85vh; }
        .slide.active { display: flex; }
        .glass-card { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .mercosul-border { border-top: 14px solid #003399; }
    </style>
</head>
<body class="flex flex-col justify-between min-h-screen select-none">

    <!-- BARRA SUPERIOR -->
    <header class="w-full bg-slate-900/80 border-b border-slate-800 px-8 py-4 flex justify-between items-center z-50">
        <div class="flex items-center space-x-3">
            <span class="bg-blue-600 text-white font-black text-xs px-2.5 py-1 rounded tracking-widest uppercase">ALPR PRO</span>
            <span class="text-sm font-semibold text-slate-300">Sistema de Identificação de Placas e Consulta Policial</span>
        </div>
        <div class="flex items-center space-x-4">
            <span id="slideIndicator" class="text-xs font-mono font-bold bg-slate-800 text-blue-400 px-3 py-1 rounded-full border border-slate-700">Slide 1 / 14</span>
            <button onclick="toggleFullScreen()" class="text-slate-400 hover:text-white text-sm transition" title="Tela Cheia"><i class="fa-solid fa-expand"></i></button>
        </div>
    </header>

    <!-- CONTEÚDO DOS SLIDES -->
    <main class="flex-1 flex items-center justify-center p-6 max-w-7xl mx-auto w-full">

        <!-- SLIDE 1: CAPA -->
        <div class="slide active flex-col justify-center items-start space-y-6 w-full py-8">
            <div class="inline-flex items-center space-x-2 bg-blue-500/10 border border-blue-500/30 px-3 py-1.5 rounded-full text-blue-400 text-xs font-semibold tracking-wide">
                <i class="fa-solid fa-shield-halved"></i>
                <span>VISÃO COMPUTACIONAL & INTELIGÊNCIA ARTIFICIAL</span>
            </div>
            <h1 class="text-4xl md:text-5xl font-extrabold text-white leading-tight tracking-tight">
                SISTEMA INTELIGENTE DE <br><span class="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-400 to-emerald-400">IDENTIFICAÇÃO DE PLACAS (ALPR)</span>
            </h1>
            <p class="text-lg text-slate-400 max-w-3xl">
                Do Treinamento em Deep Learning ao Pipeline Híbrido de Alto Desempenho (YOLO + OCR), Tratamento Morfológico e Consulta em Tempo Real de Veículos Roubados.
            </p>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-4xl pt-4">
                <div class="glass-card p-4 rounded-xl border border-slate-700/50">
                    <span class="text-xs font-bold text-blue-400 tracking-wider uppercase">Equipe do Projeto</span>
                    <p class="text-sm font-medium text-slate-200 mt-1">Caio Felipe G. Lopes • Davi Alves Mares • Carlos Eduardo Pena Fiel Menon de Freitas</p>
                </div>
                <div class="glass-card p-4 rounded-xl border border-slate-700/50">
                    <span class="text-xs font-bold text-emerald-400 tracking-wider uppercase">Stack Tecnológico</span>
                    <p class="text-sm font-medium text-slate-200 mt-1">YOLOv8/11 • Roboflow Universe • EasyOCR • OpenCV • FastAPI • SQLite</p>
                </div>
            </div>
        </div>

        <!-- SLIDE 2: O DESAFIO REAL -->
        <div class="slide flex-col justify-center space-y-6 w-full">
            <div>
                <span class="text-xs font-bold text-blue-400 uppercase tracking-widest">Capítulo 1</span>
                <h2 class="text-3xl font-extrabold text-white mt-1">O Desafio Real da Leitura de Placas</h2>
                <p class="text-sm text-slate-400">Por que modelos convencionais de visão computacional falham nas ruas?</p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="glass-card p-5 rounded-xl border-l-4 border-amber-500">
                    <h3 class="text-base font-bold text-white flex items-center gap-2"><i class="fa-solid fa-compass-drafting text-amber-400"></i> Rotação & Distorção Afim</h3>
                    <p class="text-sm text-slate-300 mt-2">Câmeras públicas capturam placas com rotações de -35° a +35°. O desalinhamento destrói a ordem horizontal de leitura.</p>
                </div>
                <div class="glass-card p-5 rounded-xl border-l-4 border-blue-500">
                    <h3 class="text-base font-bold text-white flex items-center gap-2"><i class="fa-solid fa-sun text-blue-400"></i> Iluminação Adversária</h3>
                    <p class="text-sm text-slate-300 mt-2">Faróis de xênon, reflexos no metal da placa, sombras duras e penumbra estouram o histograma e cegam o detector.</p>
                </div>
                <div class="glass-card p-5 rounded-xl border-l-4 border-purple-500">
                    <h3 class="text-base font-bold text-white flex items-center gap-2"><i class="fa-solid fa-font text-purple-400"></i> Tipografia Antifraude FE-Schrift</h3>
                    <p class="text-sm text-slate-300 mt-2">A fonte do Mercosul possui cortes deliberados que confundem OCRs genéricos entre 'Q/O', '8/B', '5/S' e '1/I'.</p>
                </div>
                <div class="glass-card p-5 rounded-xl border-l-4 border-red-500">
                    <h3 class="text-base font-bold text-white flex items-center gap-2"><i class="fa-solid fa-flag text-red-400"></i> Poluição da Faixa 'BRASIL'</h3>
                    <p class="text-sm text-slate-300 mt-2">A tarja azul superior gera falsos caracteres iniciais (ex: leitura 'BRA5I30' ao invés da placa real 'QCU5C70').</p>
                </div>
            </div>
        </div>

        <!-- SLIDE 3: ARQUITETURA GLOBAL -->
        <div class="slide flex-col justify-center space-y-6 w-full">
            <div>
                <span class="text-xs font-bold text-blue-400 uppercase tracking-widest">Capítulo 2</span>
                <h2 class="text-3xl font-extrabold text-white mt-1">Arquitetura Multicamadas do Pipeline</h2>
                <p class="text-sm text-slate-400">Fluxo completo de processamento em 6 estágios integrados</p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="glass-card p-4 rounded-xl border border-slate-700">
                    <span class="text-xs font-bold text-blue-400">ESTÁGIO 1</span>
                    <h4 class="text-sm font-bold text-white mt-1">Detecção YOLO de Placas</h4>
                    <p class="text-xs text-slate-400 mt-1">Localização e recorte preciso da bounding box da placa veicular.</p>
                </div>
                <div class="glass-card p-4 rounded-xl border border-slate-700">
                    <span class="text-xs font-bold text-emerald-400">ESTÁGIO 2</span>
                    <h4 class="text-sm font-bold text-white mt-1">Deskew Hough (Rotação)</h4>
                    <p class="text-xs text-slate-400 mt-1">Correção trigonométrica do ângulo de inclinação para 0.0° perfeito.</p>
                </div>
                <div class="glass-card p-4 rounded-xl border border-slate-700">
                    <span class="text-xs font-bold text-purple-400">ESTÁGIO 3</span>
                    <h4 class="text-sm font-bold text-white mt-1">Multicontraste (4 Variantes)</h4>
                    <p class="text-xs text-slate-400 mt-1">Corte da tarja azul e geração de imagens em Normal, CLAHE, Otsu e Adaptativo.</p>
                </div>
                <div class="glass-card p-4 rounded-xl border border-slate-700">
                    <span class="text-xs font-bold text-amber-400">ESTÁGIO 4</span>
                    <h4 class="text-sm font-bold text-white mt-1">Consenso Híbrido (YOLO+OCR)</h4>
                    <p class="text-xs text-slate-400 mt-1">Votação por confiança caractere a caractere com redundância mútua.</p>
                </div>
                <div class="glass-card p-4 rounded-xl border border-slate-700">
                    <span class="text-xs font-bold text-indigo-400">ESTÁGIO 5</span>
                    <h4 class="text-sm font-bold text-white mt-1">Desambiguação FE-Schrift</h4>
                    <p class="text-xs text-slate-400 mt-1">Validação sintática posicional das 7 classes (Mercosul e Cinza).</p>
                </div>
                <div class="glass-card p-4 rounded-xl border border-slate-700">
                    <span class="text-xs font-bold text-red-400">ESTÁGIO 6</span>
                    <h4 class="text-sm font-bold text-white mt-1">Consulta Policial & Web</h4>
                    <p class="text-xs text-slate-400 mt-1">Busca instantânea no SQLite de veículos roubados e exibição no FastAPI.</p>
                </div>
            </div>
        </div>

        <!-- SLIDE 4: YOLO 1 TREINAMENTO -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-blue-400 uppercase tracking-widest">Capítulo 3</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">YOLO 1: Detector de Placas Veiculares</h2>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <div class="space-y-3 text-sm text-slate-300">
                    <div class="glass-card p-4 rounded-xl">
                        <span class="text-xs font-bold text-blue-400 uppercase">Dataset & Configuração</span>
                        <p class="mt-1">Treinado com o dataset Open Images V7 + placas reais brasileiras. Modelo YOLOv8 Nano com resolução de 640px e batch size 16.</p>
                    </div>
                    <div class="glass-card p-4 rounded-xl">
                        <span class="text-xs font-bold text-emerald-400 uppercase">Métricas Alcançadas</span>
                        <p class="mt-1">Box mAP@50 superior a 95%, garantindo recorte milimétrico da placa em ângulos frontais e traseiros.</p>
                    </div>
                </div>
                <div class="glass-card p-2 rounded-xl flex items-center justify-center">
                    <img src="../data/resultados treinamento modelo de identificação de placa/results.png" onerror="this.src='../relatorios_benchmark/benchmark_grafico_precisao.png'" class="max-h-72 rounded-lg object-contain shadow-lg">
                </div>
            </div>
        </div>

        <!-- SLIDE 5: YOLO 2 TREINAMENTO -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-blue-400 uppercase tracking-widest">Capítulo 4</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">YOLO 2: Detector de Caracteres (Roboflow)</h2>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <div class="glass-card p-2 rounded-xl flex items-center justify-center">
                    <img src="../data/resultados treinamento modelo de identificacao de caracteres yolo 2/metricas_roboflow_yolo2.png" class="max-h-72 rounded-lg object-contain shadow-lg">
                </div>
                <div class="glass-card p-2 rounded-xl flex items-center justify-center">
                    <img src="../data/resultados treinamento modelo de identificacao de caracteres yolo 2/results.png" class="max-h-72 rounded-lg object-contain shadow-lg">
                </div>
            </div>
        </div>

        <!-- SLIDE 6: O CHOQUE DE REALIDADE -->
        <div class="slide flex-col justify-center space-y-6 w-full">
            <div>
                <span class="text-xs font-bold text-red-400 uppercase tracking-widest">Capítulo 5</span>
                <h2 class="text-3xl font-extrabold text-white mt-1">O Choque de Realidade: A Falha em Cascata</h2>
                <p class="text-sm text-slate-400">Por que 99,3% de mAP no Roboflow se tornaram apenas 7,0% em fotos reais de rua?</p>
            </div>
            <div class="glass-card p-6 rounded-2xl border-l-4 border-red-500 space-y-4">
                <h3 class="text-xl font-bold text-white">A Matemática da Exigência Perfeita</h3>
                <p class="text-sm text-slate-300 leading-relaxed">
                    Em tarefas normais de detecção de objetos (ex: identificar carros ou pessoas), errar 1 objeto não invalida os outros. <br>
                    <strong>Em leitura de placas, o sistema é estritamente sequencial:</strong> se a IA acertar 6 caracteres e errar 1 único (ex: trocar 'B' por '8' ou 'Q' por 'O'), <strong>a placa inteira torna-se 100% inútil para a polícia</strong>.
                </p>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
                    <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-700">
                        <span class="text-xs text-slate-400">Acurácia por Caractere</span>
                        <div class="text-2xl font-black text-amber-400 mt-1">68,0%</div>
                        <span class="text-[11px] text-slate-500">Sob ângulo e reflexo real</span>
                    </div>
                    <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-700">
                        <span class="text-xs text-slate-400">Probabilidade Conjunta (7 chars)</span>
                        <div class="text-2xl font-black text-red-500 mt-1">(0,68)⁷ = 6,7%</div>
                        <span class="text-[11px] text-slate-500">Probabilidade da placa correta</span>
                    </div>
                    <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-700">
                        <span class="text-xs text-slate-400">Acerto Real Constatado</span>
                        <div class="text-2xl font-black text-red-400 mt-1">7,0%</div>
                        <span class="text-[11px] text-slate-500">YOLO isolado nas 71 fotos</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- SLIDE 7: BATALHA DIRETA -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-blue-400 uppercase tracking-widest">Capítulo 6</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">Batalha Direta: YOLO vs EasyOCR vs Híbrido</h2>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <div class="glass-card p-2 rounded-xl flex items-center justify-center">
                    <img src="../relatorios_benchmark/grafico_1_comparativo_yolo_ocr_hibrido.png" class="max-h-72 rounded-lg object-contain shadow-lg">
                </div>
                <div class="space-y-3">
                    <div class="glass-card p-4 rounded-xl border-l-4 border-red-500">
                        <span class="text-xs font-bold text-red-400">YOLO ISOLADO (7,0%)</span>
                        <p class="text-xs text-slate-300 mt-1">Sofria muito com a inclinação e ruído de fundo, perdendo letras intermediárias.</p>
                    </div>
                    <div class="glass-card p-4 rounded-xl border-l-4 border-amber-500">
                        <span class="text-xs font-bold text-amber-400">EASYOCR ISOLADO (43,7%)</span>
                        <p class="text-xs text-slate-300 mt-1">Superior ao YOLO em texto plano, porém lia a tarja 'BRASIL' e confundia caracteres semelhantes.</p>
                    </div>
                    <div class="glass-card p-4 rounded-xl border-l-4 border-emerald-500">
                        <span class="text-xs font-bold text-emerald-400">PIPELINE HÍBRIDO (95,8%)</span>
                        <p class="text-xs text-slate-300 mt-1">Salto dramático de +52,1% combinando retificação, multicontraste, desambiguação e votação consensual.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- SLIDE 8: SOLUÇÃO 1 - DESKEW -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-emerald-400 uppercase tracking-widest">Melhoria Técnica 1</span>
                <h2 class="text-3xl font-extrabold text-white">Retificação Geométrica: Deskew Hough</h2>
                <p class="text-sm text-slate-400">Ganho Absoluto: <strong class="text-emerald-400">+23,0% de Acurácia</strong></p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="glass-card p-5 rounded-xl border border-slate-700 space-y-3">
                    <h3 class="text-base font-bold text-blue-400">Como funciona o Algoritmo:</h3>
                    <ul class="text-xs text-slate-300 space-y-2">
                        <li>• Detecção de bordas via filtro Canny (50, 150).</li>
                        <li>• Detecção probabilística de linhas horizontais com HoughLinesP.</li>
                        <li>• Cálculo trigonométrico do ângulo angular médio de -35° a +35°.</li>
                        <li>• Rotação por interpolação bicúbica via getRotationMatrix2D e warpAffine.</li>
                    </ul>
                </div>
                <div class="glass-card p-5 rounded-xl border border-slate-700 flex flex-col justify-center items-center text-center">
                    <div class="text-4xl font-extrabold text-emerald-400">15% ➔ 38%</div>
                    <p class="text-xs text-slate-400 mt-2">Salto de precisão ao forçar todas as placas a entrarem perfeitamente horizontais no motor de reconhecimento.</p>
                </div>
            </div>
        </div>

        <!-- SLIDE 9: SOLUÇÃO 2 - MULTICONTRASTE -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-emerald-400 uppercase tracking-widest">Melhoria Técnica 2</span>
                <h2 class="text-3xl font-extrabold text-white">Multicontraste & Eliminação da Tarja 'BRASIL'</h2>
                <p class="text-sm text-slate-400">Ganho Absoluto: <strong class="text-emerald-400">+24,0% de Acurácia</strong></p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div class="glass-card p-3 rounded-xl border border-slate-700">
                    <span class="text-xs font-bold text-blue-400">VARIANTE 1</span>
                    <h4 class="text-xs font-bold text-white mt-1">Normal HD</h4>
                    <p class="text-[11px] text-slate-400 mt-1">Miolo recortado sem a tarja azul e upscaling Lanczos4.</p>
                </div>
                <div class="glass-card p-3 rounded-xl border border-slate-700">
                    <span class="text-xs font-bold text-emerald-400">VARIANTE 2</span>
                    <h4 class="text-xs font-bold text-white mt-1">Realce CLAHE</h4>
                    <p class="text-[11px] text-slate-400 mt-1">Equalização de histograma + sharpening para bordas suaves.</p>
                </div>
                <div class="glass-card p-3 rounded-xl border border-slate-700">
                    <span class="text-xs font-bold text-purple-400">VARIANTE 3</span>
                    <h4 class="text-xs font-bold text-white mt-1">Binarização Otsu</h4>
                    <p class="text-[11px] text-slate-400 mt-1">Segmentação P&B ideal para placas com sombra forte.</p>
                </div>
                <div class="glass-card p-3 rounded-xl border border-slate-700">
                    <span class="text-xs font-bold text-amber-400">VARIANTE 4</span>
                    <h4 class="text-xs font-bold text-white mt-1">Adaptativo Local</h4>
                    <p class="text-[11px] text-slate-400 mt-1">Limiarização adaptativa para iluminação não uniforme e farol.</p>
                </div>
            </div>
            <div class="glass-card p-3 rounded-xl text-center text-xs text-slate-300">
                Acurácia subiu de <strong>38,0% para 62,0%</strong> em virtude da imunidade contra variações de luz.
            </div>
        </div>

        <!-- SLIDE 10: SOLUÇÃO 3 - DESAMBIGUAÇÃO -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-emerald-400 uppercase tracking-widest">Melhoria Técnica 3</span>
                <h2 class="text-3xl font-extrabold text-white">Desambiguação Morfológica FE-Schrift</h2>
                <p class="text-sm text-slate-400">Ganho Absoluto: <strong class="text-emerald-400">+16,0% de Acurácia</strong></p>
            </div>
            <div class="glass-card p-5 rounded-xl border border-slate-700 space-y-3">
                <h3 class="text-sm font-bold text-white">Regras Posicionais Estritas Aplicadas:</h3>
                <div class="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
                    <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                        <strong class="text-blue-400">Posição 1, 2 e 3</strong>
                        <p class="text-slate-300 mt-1">LETRAS obrigatórias.<br>Converte '0'➔'O', '8'➔'B', '5'➔'S'.</p>
                    </div>
                    <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                        <strong class="text-purple-400">Posição 4</strong>
                        <p class="text-slate-300 mt-1">DÍGITO obrigatório.<br>Converte 'O'➔'0', 'B'➔'8', 'S'➔'5'.</p>
                    </div>
                    <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                        <strong class="text-emerald-400">Posição 5</strong>
                        <p class="text-slate-300 mt-1">Mercosul = LETRA.<br>Antiga Cinza = DÍGITO.</p>
                    </div>
                    <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                        <strong class="text-amber-400">Posição 6 e 7</strong>
                        <p class="text-slate-300 mt-1">DÍGITOS obrigatórios.<br>Converte 'I'➔'1', 'O'➔'0', 'Z'➔'2'.</p>
                    </div>
                </div>
            </div>
            <div class="glass-card p-3 rounded-xl text-center text-xs text-slate-300">
                Acurácia subiu de <strong>62,0% para 78,0%</strong> corrigindo toda a matriz de confusão da tipografia Mercosul.
            </div>
        </div>

        <!-- SLIDE 11: A VIRADA DE CHAVE -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-emerald-400 uppercase tracking-widest">Capítulo 7</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">A Virada de Chave: O Pipeline Híbrido</h2>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <div class="glass-card p-2 rounded-xl flex items-center justify-center">
                    <img src="../relatorios_benchmark/grafico_2_evolucao_etapas.png" class="max-h-72 rounded-lg object-contain shadow-lg">
                </div>
                <div class="space-y-3">
                    <div class="glass-card p-4 rounded-xl border border-slate-700">
                        <h4 class="text-sm font-bold text-emerald-400">Consenso Votado por Caractere</h4>
                        <p class="text-xs text-slate-300 mt-1">Onde o OCR hesita, o YOLO decide a classe pela caixa delimitadora; onde o YOLO hesita por colagem de letras ('M' e 'H'), o OCR refina a morfologia.</p>
                    </div>
                    <div class="glass-card p-4 rounded-xl border border-slate-700">
                        <h4 class="text-sm font-bold text-blue-400">Resultado Oficial Consolidado</h4>
                        <p class="text-xs text-slate-300 mt-1"><strong>95,8% de acerto exato</strong> nas placas principais e <strong>89,41% de acerto geral</strong> em todas as 85 placas do dataset.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- SLIDE 12: IMPACTO POR TÉCNICA -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-blue-400 uppercase tracking-widest">Capítulo 8</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">Ganhos Incrementais & Qualidade de Leitura</h2>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <div class="glass-card p-2 rounded-xl flex items-center justify-center">
                    <img src="../relatorios_benchmark/grafico_3_ganho_por_metodo.png" class="max-h-72 rounded-lg object-contain shadow-lg">
                </div>
                <div class="glass-card p-2 rounded-xl flex items-center justify-center">
                    <img src="../relatorios_benchmark/grafico_4_distribuicao_qualidade_leitura.png" class="max-h-72 rounded-lg object-contain shadow-lg">
                </div>
            </div>
        </div>

        <!-- SLIDE 13: REENGENHARIA FASTAPI -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-blue-400 uppercase tracking-widest">Capítulo 9</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">Migração de Streamlit para FastAPI + Tailwind</h2>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="glass-card p-5 rounded-xl border-l-4 border-red-500 space-y-2">
                    <h3 class="text-base font-bold text-red-400"><i class="fa-solid fa-xmark mr-1"></i> Base Anterior: Streamlit</h3>
                    <p class="text-xs text-slate-300">Monolítico e pesado. Reexecutava o script Python inteiro a cada clique, congestionando CPU/GPU e travando a interface gráfica.</p>
                </div>
                <div class="glass-card p-5 rounded-xl border-l-4 border-emerald-500 space-y-2">
                    <h3 class="text-base font-bold text-emerald-400"><i class="fa-solid fa-check mr-1"></i> Nova Base: FastAPI + Uvicorn</h3>
                    <p class="text-xs text-slate-300">Backend assíncrono ultraleve. Modelos de IA pré-carregados na inicialização, API REST instantânea e frontend responsivo com Glassmorphism.</p>
                </div>
            </div>
            <div class="glass-card p-4 rounded-xl text-xs text-slate-300">
                🚀 A aplicação agora inicia instantaneamente via <strong>iniciar_sistema.bat</strong> na porta <strong>http://localhost:8000</strong>.
            </div>
        </div>

        <!-- SLIDE 14: DASHBOARD EXECUTIVO -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-emerald-400 uppercase tracking-widest">Conclusão</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">Dashboard Executivo Consolidado</h2>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <div class="glass-card p-2 rounded-xl flex items-center justify-center">
                    <img src="../relatorios_benchmark/painel_executivo_evolucao_completo.png" class="max-h-72 rounded-lg object-contain shadow-lg">
                </div>
                <div class="space-y-3 text-xs text-slate-300">
                    <div class="glass-card p-3 rounded-lg border border-slate-700">
                        <strong class="text-white">1. Visão Computacional de Alto Nível:</strong> A solução combinou IA moderna com retificação trigonométrica e heurísticas morfológicas.
                    </div>
                    <div class="glass-card p-3 rounded-lg border border-slate-700">
                        <strong class="text-white">2. Confiabilidade Policial:</strong> Erros de leitura caíram drasticamente, garantindo busca precisa no banco de dados SQLite.
                    </div>
                    <div class="glass-card p-3 rounded-lg border border-slate-700">
                        <strong class="text-white">3. Repositório Completo:</strong> Todo o projeto foi comitado e sincronizado no GitHub com relatórios e scripts executáveis.
                    </div>
                </div>
            </div>
        </div>

    </main>

    <!-- BARRA INFERIOR DE NAVEGAÇÃO -->
    <footer class="w-full bg-slate-900/80 border-t border-slate-800 px-8 py-3 flex justify-between items-center z-50">
        <span class="text-xs text-slate-500">Dica: Use as setas do teclado &larr; &rarr; para navegar</span>
        <div class="flex space-x-3">
            <button onclick="prevSlide()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs rounded-lg transition border border-slate-700 flex items-center gap-2">
                <i class="fa-solid fa-arrow-left"></i> Anterior
            </button>
            <button onclick="nextSlide()" class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs rounded-lg transition shadow-lg shadow-blue-500/20 flex items-center gap-2">
                Próximo <i class="fa-solid fa-arrow-right"></i>
            </button>
        </div>
    </footer>

    <script>
        let currentSlide = 0;
        const slides = document.querySelectorAll('.slide');
        const indicator = document.getElementById('slideIndicator');

        function updateSlide() {
            slides.forEach((s, idx) => {
                s.classList.toggle('active', idx === currentSlide);
            });
            indicator.textContent = `Slide ${currentSlide + 1} / ${slides.length}`;
        }

        function nextSlide() {
            if (currentSlide < slides.length - 1) {
                currentSlide++;
                updateSlide();
            }
        }

        function prevSlide() {
            if (currentSlide > 0) {
                currentSlide--;
                updateSlide();
            }
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight' || e.key === ' ') nextSlide();
            if (e.key === 'ArrowLeft') prevSlide();
        });

        function toggleFullScreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
            } else if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        }
    </script>
</body>
</html>
"""
    with open(caminho_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"🎉 Apresentação Web HTML gerada com sucesso: {caminho_html}")


if __name__ == "__main__":
    print("Iniciando geração das apresentações de slides (PPTX e HTML)...")
    criar_apresentacao_pptx()
    criar_apresentacao_html()
    print("\n✅ Apresentações concluídas na pasta 'apresentacao/'!")
