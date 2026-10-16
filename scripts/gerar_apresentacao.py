"""
Script para Geração da Apresentação de Slides do Projeto ALPR.
Cria duas versões da apresentação:
1. apresentacao_projeto_alpr.pptx (PowerPoint Widescreen 16:9 profissional e editável)
2. apresentacao_projeto_alpr.html (Apresentação Web Interativa com Tailwind CSS e navegação moderna)
Incorpora todos os gráficos de treinamento (YOLO 1 e YOLO 2), benchmarks e a visualização
da imagem em cada passo do pipeline acompanhada de suas legendas explicativas.
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

# Pastas de Recursos Visuais
DIR_YOLO1 = BASE_DIR / "data" / "resultados treinamento modelo de identificação de placa"
if not DIR_YOLO1.exists():
    for d in (BASE_DIR / "data").glob("*resultados treinamento*identifica*"):
        if "caracteres" not in d.name.lower():
            DIR_YOLO1 = d
            break

DIR_YOLO2 = BASE_DIR / "data" / "resultados treinamento modelo de identificacao de caracteres yolo 2"
DIR_BENCH = BASE_DIR / "relatorios_benchmark"
DIR_ETAPAS = BASE_DIR / "apresentacao" / "etapas_pipeline"

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
    p.text = "Demonstração Visual: Transformação da Imagem em Cada Passo do Pipeline e Benchmark Oficial"
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
    
    desafios = [
        ("📐 Rotação & Ângulo", "Câmeras de tráfego capturam veículos em ângulos inclinados (-35° a +35°). A deformação quebra a ordem horizontal de leitura dos caracteres."),
        ("🌗 Iluminação Extrema", "Faróis acesos, reflexos solares no alumínio, sombras e penumbra geram estouro de branco ou perda crítica de contraste."),
        ("🔤 Tipografia FE-Schrift", "A fonte oficial do Mercosul foi projetada propositalmente para ser antifraude, confundindo leitores comuns entre 'Q/O', '8/B', '5/S' e '1/I'."),
        ("🇧🇷 Elementos Espúrios", "A tarja azul superior com a palavra 'BRASIL' e a bandeira geram falsos caracteres iniciais (ex: leitura 'BRA5I30' ao invés da placa real).")
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
    # SLIDE 3: O PIPELINE EM AÇÃO - PANORAMA VISUAL COMPLETO
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s3, C_BRANCO)
    adicionar_cabecalho(s3, "2. O Pipeline em Ação: Visão Panorâmica de Cada Etapa", "Como a imagem é capturada, recortada, nivelada, filtrada e reconhecida até o resultado final")
    
    img_painel = DIR_ETAPAS / "painel_infografico_pipeline_completo.jpg"
    if img_painel.exists():
        s3.shapes.add_picture(str(img_painel), Inches(0.8), Inches(1.8), width=Inches(11.7))

    # =========================================================================
    # SLIDE 4: PASSO A PASSO (1): CAPTURA, DETECÇÃO E CROP BRUTO
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s4, C_BRANCO)
    adicionar_cabecalho(s4, "3. Etapas 1, 2 e 3: Captura, Detecção YOLO e Recorte Bruto", "Da cena urbana completa ao isolamento da região da placa veicular")
    
    cards_s4 = [
        ("etapa_1_imagem_original.jpg", "1. Imagem Original", "Foto bruta capturada em ambiente real de tráfego."),
        ("etapa_2_deteccao_yolo1.jpg", "2. Detecção YOLO 1", "Localização da bounding box da placa veicular (>95% conf)."),
        ("etapa_3_crop_bruto_inclinado.jpg", "3. Recorte Bruto (Crop)", "Placa isolada. Note a inclinação angular e a tarja 'BRASIL'.")
    ]
    for i, (arq, tit, desc) in enumerate(cards_s4):
        x = Inches(0.8 + i * 3.95)
        p_img = DIR_ETAPAS / arq
        if p_img.exists():
            s4.shapes.add_picture(str(p_img), x, Inches(1.8), width=Inches(3.7))
        # Card com legenda explicativa
        card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(5.1), Inches(3.7), Inches(1.8))
        card.fill.solid()
        card.fill.fore_color.rgb = C_FUNDO_CARD
        card.line.color.rgb = RGBColor(203, 213, 225)
        
        tb = s4.shapes.add_textbox(x + Inches(0.15), Inches(5.2), Inches(3.4), Inches(1.5))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = tit
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = C_AZUL_MEDIO
        
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = C_CINZA_TEXTO
        p2.space_before = Pt(4)

    # =========================================================================
    # SLIDE 5: PASSO A PASSO (2): RETIFICAÇÃO GEOMÉTRICA (DESKEW)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s5, C_BRANCO)
    adicionar_cabecalho(s5, "4. Etapa 4: Retificação Geométrica (Deskew com Hough)", "Alinhamento automático do ângulo de rotação da placa para 0.0° perfeito")
    
    # Imagem antes e depois
    img_antes = DIR_ETAPAS / "etapa_3_crop_bruto_inclinado.jpg"
    img_depois = DIR_ETAPAS / "etapa_4_retificacao_deskew.jpg"
    
    if img_antes.exists():
        s5.shapes.add_picture(str(img_antes), Inches(0.8), Inches(1.8), width=Inches(5.6))
    if img_depois.exists():
        s5.shapes.add_picture(str(img_depois), Inches(6.9), Inches(1.8), width=Inches(5.6))
        
    tb5 = s5.shapes.add_textbox(Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.8))
    tf5 = tb5.text_frame
    tf5.word_wrap = True
    p = tf5.paragraphs[0]
    p.text = "💡 O que aconteceu neste passo:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_AZUL_MEDIO
    
    p2 = tf5.add_paragraph()
    p2.text = "• Bordas Canny + cv2.HoughLinesP detectaram segmentos horizontais na placa inclinada (+11.6° de desvio).\n• A mediana angular calculou a rotação exata e cv2.warpAffine aplicou a matriz afim corretiva.\n• Resultado: A placa agora entra perfeitamente nivelada no OCR, gerando um ganho imediato de +23,0% na taxa de leitura."
    p2.font.size = Pt(12)
    p2.font.color.rgb = C_AZUL_ESCURO
    p2.space_before = Pt(4)

    # =========================================================================
    # SLIDE 6: PASSO A PASSO (3): ISOLAMENTO DO MIOLO & MULTICONTRASTE
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s6, C_BRANCO)
    adicionar_cabecalho(s6, "5. Etapas 5 e 6: Corte da Tarja 'BRASIL' e Multicontraste Dinâmico", "Eliminação do falso prefixo 'BRA' e geração simultânea das 4 variantes de contraste")
    
    img_miolo = DIR_ETAPAS / "etapa_5_corte_tarja_brasil.jpg"
    img_multi = DIR_ETAPAS / "etapa_6_multicontraste_4filtros.jpg"
    
    if img_miolo.exists():
        s6.shapes.add_picture(str(img_miolo), Inches(0.8), Inches(1.8), width=Inches(5.6))
    if img_multi.exists():
        s6.shapes.add_picture(str(img_multi), Inches(6.9), Inches(1.8), width=Inches(5.6))
        
    tb6 = s6.shapes.add_textbox(Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.8))
    tf6 = tb6.text_frame
    tf6.word_wrap = True
    p = tf6.paragraphs[0]
    p.text = "💡 O que aconteceu nestes passos:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_AZUL_MEDIO
    
    p2 = tf6.add_paragraph()
    p2.text = "• Remoção da Tarja Azul: Descarte dos 20% do topo da placa, impedindo que 'BRASIL' e a bandeira gerem falsos caracteres.\n• 4 Filtros Concorrentes: Avaliação em Normal HD, Realce CLAHE + Sharpen (sombras suaves), Binarização Otsu (sombras densas) e Limiar Adaptativo Local (reflexos de farol).\n• Resultado: Ganho de +24,0% de acurácia, tornando o sistema imune a variações severas de luz."
    p2.font.size = Pt(12)
    p2.font.color.rgb = C_AZUL_ESCURO
    p2.space_before = Pt(4)

    # =========================================================================
    # SLIDE 7: PASSO A PASSO (4): RECONHECIMENTO YOLO 2 E RESULTADO FINAL
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s7, C_BRANCO)
    adicionar_cabecalho(s7, "6. Etapas 7 e 8: YOLO de Caracteres, Fusão Híbrida e Consulta", "Detecção individual dos 7 caracteres, votação por confiança e status policial")
    
    img_chars = DIR_ETAPAS / "etapa_7_yolo2_deteccao_caracteres.jpg"
    img_final = DIR_ETAPAS / "etapa_8_consenso_e_consulta_policial.jpg"
    
    if img_chars.exists():
        s7.shapes.add_picture(str(img_chars), Inches(0.8), Inches(1.8), width=Inches(5.6))
    if img_final.exists():
        s7.shapes.add_picture(str(img_final), Inches(6.9), Inches(1.8), width=Inches(5.6))
        
    tb7 = s7.shapes.add_textbox(Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.8))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    p = tf7.paragraphs[0]
    p.text = "💡 O que aconteceu nestes passos:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_AZUL_MEDIO
    
    p2 = tf7.add_paragraph()
    p2.text = "• Detecção pelo YOLO 2: Cada caractere ('Q', 'C', 'U', '5', 'C', '7', '0') tem sua caixa delimitadora identificada e ordenada.\n• Votação Ponderada: O EasyOCR valida a morfologia da fonte FE-Schrift e o YOLO arbitra a certeza posicional.\n• Consulta Policial Instantânea: Placa formatada validada e consultada no banco SQLite com resultado REGULAR."
    p2.font.size = Pt(12)
    p2.font.color.rgb = C_AZUL_ESCURO
    p2.space_before = Pt(4)

    # =========================================================================
    # SLIDE 8: TREINAMENTO DO YOLO 1 (DETECTOR DE PLACAS)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s8, C_BRANCO)
    adicionar_cabecalho(s8, "7. Treinamento do YOLO 1: Detecção de Placas", "Localização precisa da bounding box da placa veicular na imagem completa")
    
    tb8 = s8.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf8 = tb8.text_frame
    tf8.word_wrap = True
    itens_y1 = [
        ("📦 Dataset de Treino:", "Open Images V7 combinado com amostras de veículos brasileiros em tráfego."),
        ("⚙️ Hiperparâmetros:", "YOLOv8 Nano, Batch 16, Resolução de 640px com Data Augmentation automático."),
        ("🎯 Desempenho:", "Alcançou rápida convergência nas primeiras 30 épocas com mAP@50 > 95%."),
        ("💡 Papel no Pipeline:", "Isola com precisão os limites da placa veicular, descartando o veículo e o cenário urbano ao redor.")
    ]
    for i, (t, d) in enumerate(itens_y1):
        p = tf8.paragraphs[0] if i == 0 else tf8.add_paragraph()
        p.text = f"{t} {d}"
        p.font.size = Pt(13)
        p.font.color.rgb = C_AZUL_ESCURO
        p.space_before = Pt(12)
        
    img_y1 = DIR_YOLO1 / "results.png"
    if not img_y1.exists():
        img_y1 = DIR_YOLO1 / "BoxPR_curve.png"
    if img_y1.exists():
        s8.shapes.add_picture(str(img_y1), Inches(6.6), Inches(1.8), width=Inches(5.9))

    # =========================================================================
    # SLIDE 9: TREINAMENTO DO YOLO 2 (CARACTERES - ROBOFLOW)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s9, C_BRANCO)
    adicionar_cabecalho(s9, "8. Treinamento do YOLO 2: Reconhecimento de Caracteres", "Modelo especializado em 36 classes (0-9 e A-Z) treinado no Roboflow Universe")
    
    img_y2_card = DIR_YOLO2 / "metricas_roboflow_yolo2.png"
    if img_y2_card.exists():
        s9.shapes.add_picture(str(img_y2_card), Inches(0.8), Inches(1.8), width=Inches(6.2))
        
    img_y2_res = DIR_YOLO2 / "BoxPR_curve.png"
    if img_y2_res.exists():
        s9.shapes.add_picture(str(img_y2_res), Inches(7.3), Inches(1.8), width=Inches(5.2))

    # =========================================================================
    # SLIDE 10: O CHOQUE DE REALIDADE (EFEITO CASCATA DE ERRO)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s10, C_BRANCO)
    adicionar_cabecalho(s10, "9. O Choque de Realidade: O Efeito Cascata de Erro", "Por que uma rede neural com 99,3% de mAP alcançou apenas 7,0% de acerto nas placas reais?")
    
    tb10 = s10.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf10 = tb10.text_frame
    tf10.word_wrap = True
    
    p = tf10.paragraphs[0]
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
        p_pt = tf10.add_paragraph()
        p_pt.text = pt
        p_pt.font.size = Pt(13)
        p_pt.font.color.rgb = C_AZUL_ESCURO
        p_pt.space_before = Pt(12)

    # =========================================================================
    # SLIDE 11: BATALHA DIRETA: YOLO VS OCR VS HÍBRIDO
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s11, C_BRANCO)
    adicionar_cabecalho(s11, "10. Batalha Direta: YOLO Isolado vs EasyOCR Isolado", "Avaliação cega independente sobre as 71 imagens reais do Google Drive")
    
    img_comp = DIR_BENCH / "grafico_1_comparativo_yolo_ocr_hibrido.png"
    if img_comp.exists():
        s11.shapes.add_picture(str(img_comp), Inches(0.8), Inches(1.8), width=Inches(7.2))
        
    tb11 = s11.shapes.add_textbox(Inches(8.3), Inches(1.8), Inches(4.3), Inches(5.0))
    tf11 = tb11.text_frame
    tf11.word_wrap = True
    
    p = tf11.paragraphs[0]
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
        p_r = tf11.add_paragraph()
        p_r.text = r
        p_r.font.size = Pt(12.5)
        p_r.font.color.rgb = C_AZUL_ESCURO
        p_r.space_before = Pt(10)

    # =========================================================================
    # SLIDE 12: EVOLUÇÃO E GANHO INCREMENTAL POR TÉCNICA
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s12, C_BRANCO)
    adicionar_cabecalho(s12, "11. Evolução Incremental e Ganhos Marginais", "O que cada técnica técnica melhorou e a evolução acumulada do pipeline")
    
    img_g2 = DIR_BENCH / "grafico_2_evolucao_etapas.png"
    img_g3 = DIR_BENCH / "grafico_3_ganho_por_metodo.png"
    
    if img_g2.exists():
        s12.shapes.add_picture(str(img_g2), Inches(0.8), Inches(1.8), width=Inches(5.7))
    if img_g3.exists():
        s12.shapes.add_picture(str(img_g3), Inches(6.8), Inches(1.8), width=Inches(5.7))

    # =========================================================================
    # SLIDE 13: QUALIDADE DE LEITURA E SEVERIDADE DE ERROS
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s13, C_BRANCO)
    adicionar_cabecalho(s13, "12. Qualidade da Leitura e Eliminação de Erros Graves", "Distribuição da distância de edição (Levenshtein) no conjunto de teste")
    
    img_g4 = DIR_BENCH / "grafico_4_distribuicao_qualidade_leitura.png"
    if img_g4.exists():
        s13.shapes.add_picture(str(img_g4), Inches(0.8), Inches(1.8), width=Inches(7.2))
        
    tb13 = s13.shapes.add_textbox(Inches(8.3), Inches(1.8), Inches(4.3), Inches(5.0))
    tf13 = tb13.text_frame
    tf13.word_wrap = True
    p = tf13.paragraphs[0]
    p.text = "Análise da Qualidade:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_AZUL_MEDIO
    
    pontos_q = [
        "• Erros parciais (1 ou 2 caracteres) caíram de 32,4% no YOLO/OCR isolados para 0,0% no Pipeline Híbrido.",
        "• 95,8% das placas são lidas com precisão cirúrgica de 100%.",
        "• As únicas 3 divergências remanescentes ocorreram em fotos noturnas extremas ou placas desfocadas no fundo distante."
    ]
    for pt in pontos_q:
        p_pt = tf13.add_paragraph()
        p_pt.text = pt
        p_pt.font.size = Pt(12)
        p_pt.font.color.rgb = C_AZUL_ESCURO
        p_pt.space_before = Pt(10)

    # =========================================================================
    # SLIDE 14: REENGENHARIA WEB: FASTAPI + TAILWIND CSS
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s14, C_BRANCO)
    adicionar_cabecalho(s14, "13. Nova Arquitetura de Software: Migração para FastAPI", "Substituição do Streamlit por backend assíncrono ultraleve e interface moderna")
    
    card_st = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    card_st.fill.solid()
    card_st.fill.fore_color.rgb = RGBColor(254, 242, 242)
    card_st.line.color.rgb = RGBColor(252, 165, 165)
    
    tbst = s14.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.4))
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
        
    card_fa = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    card_fa.fill.solid()
    card_fa.fill.fore_color.rgb = RGBColor(240, 253, 244)
    card_fa.line.color.rgb = RGBColor(134, 239, 172)
    
    tbfa = s14.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.4))
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
        "• Interface Moderna: Design Glassmorphism em Tailwind CSS, badge Mercosul fiel e caixas delimitadoras dinâmicas.",
        "• Gestão de Banco Integrada: Consulta e cadastro de veículos regulares ou roubados no SQLite diretamente na tela."
    ]
    for pt in pontos_fa:
        p_pt = tffa.add_paragraph()
        p_pt.text = pt
        p_pt.font.size = Pt(12)
        p_pt.font.color.rgb = RGBColor(22, 101, 52)
        p_pt.space_before = Pt(8)

    # =========================================================================
    # SLIDE 15: DASHBOARD EXECUTIVO CONSOLIDADO E CONCLUSÕES
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    aplicar_fundo(s15, C_BRANCO)
    adicionar_cabecalho(s15, "14. Dashboard Executivo e Conclusões Finais", "Resumo das conquistas do projeto e visão holística do sistema ALPR")
    
    img_dash = DIR_BENCH / "painel_executivo_evolucao_completo.png"
    if img_dash.exists():
        s15.shapes.add_picture(str(img_dash), Inches(0.8), Inches(1.8), width=Inches(7.2))
        
    tb15 = s15.shapes.add_textbox(Inches(8.3), Inches(1.8), Inches(4.3), Inches(5.0))
    tf15 = tb15.text_frame
    tf15.word_wrap = True
    p = tf15.paragraphs[0]
    p.text = "Conclusões do Projeto:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_AZUL_MEDIO
    
    conclusoes = [
        "1. IA no mundo real exige engenharia: Um modelo com 99% em laboratório necessita de tratamento afim e sintático para operar com confiabilidade.",
        "2. Redundância Híbrida: A união de YOLO + EasyOCR superou os limites individuais de cada tecnologia.",
        "3. Sistema Pronto para Produção: Nova interface web em FastAPI leve, responsiva e pronta para câmeras em tempo real.",
        "4. Repositório 100% Documentado: Código limpo, planilhas Excel/CSV e gráficos de alta resolução salvos no GitHub."
    ]
    for c in conclusoes:
        p_c = tf15.add_paragraph()
        p_c.text = c
        p_c.font.size = Pt(12)
        p_c.font.color.rgb = C_AZUL_ESCURO
        p_c.space_before = Pt(8)

    caminho_pptx = OUTPUT_DIR / "apresentacao_projeto_alpr.pptx"
    prs.save(str(caminho_pptx))
    print(f"🎉 Apresentação PowerPoint gerada com sucesso: {caminho_pptx}")


def criar_apresentacao_html():
    """Gera uma apresentação Web Interativa em HTML moderna e responsiva com todas as etapas visuais."""
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
    </style>
</head>
<body class="flex flex-col justify-between min-h-screen select-none">

    <!-- BARRA SUPERIOR -->
    <header class="w-full bg-slate-900/80 border-b border-slate-800 px-8 py-3.5 flex justify-between items-center z-50">
        <div class="flex items-center space-x-3">
            <span class="bg-blue-600 text-white font-black text-xs px-2.5 py-1 rounded tracking-widest uppercase">ALPR PRO</span>
            <span class="text-sm font-semibold text-slate-300">Sistema de Identificação de Placas e Consulta Policial</span>
        </div>
        <div class="flex items-center space-x-4">
            <span id="slideIndicator" class="text-xs font-mono font-bold bg-slate-800 text-blue-400 px-3 py-1 rounded-full border border-slate-700">Slide 1 / 15</span>
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
                    <span class="text-xs font-bold text-emerald-400 tracking-wider uppercase">Visualização & Demonstração</span>
                    <p class="text-sm font-medium text-slate-200 mt-1">Transformação visual da imagem em cada estágio com legendas explicativas</p>
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

        <!-- SLIDE 3: O PIPELINE EM AÇÃO - INFOGRÁFICO COMPLETO -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-emerald-400 uppercase tracking-widest">Capítulo 2</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">O Pipeline em Ação: Visão Panorâmica</h2>
                <p class="text-sm text-slate-400">Como a imagem se transforma visualmente do início ao fim do processo</p>
            </div>
            <div class="glass-card p-2 rounded-xl flex items-center justify-center overflow-hidden">
                <img src="etapas_pipeline/painel_infografico_pipeline_completo.jpg" class="max-h-[72vh] w-full rounded-lg object-contain shadow-2xl">
            </div>
        </div>

        <!-- SLIDE 4: ETAPAS 1, 2 E 3 VISUAL -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-blue-400 uppercase tracking-widest">Capítulo 3</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">Etapas 1, 2 e 3: Captura, Detecção e Recorte</h2>
                <p class="text-sm text-slate-400">Da imagem bruta na via pública ao recorte da bounding box</p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="glass-card p-2 rounded-xl flex flex-col items-center">
                    <img src="etapas_pipeline/etapa_1_imagem_original.jpg" class="h-44 w-full rounded-lg object-contain">
                    <div class="p-2 text-center">
                        <span class="text-xs font-bold text-blue-400">1. IMAGEM ORIGINAL</span>
                        <p class="text-[11px] text-slate-400 mt-1">Foto bruta em ângulo real de tráfego.</p>
                    </div>
                </div>
                <div class="glass-card p-2 rounded-xl flex flex-col items-center">
                    <img src="etapas_pipeline/etapa_2_deteccao_yolo1.jpg" class="h-44 w-full rounded-lg object-contain">
                    <div class="p-2 text-center">
                        <span class="text-xs font-bold text-emerald-400">2. DETECÇÃO YOLO 1</span>
                        <p class="text-[11px] text-slate-400 mt-1">Bounding Box delimitada com >95% de confiança.</p>
                    </div>
                </div>
                <div class="glass-card p-2 rounded-xl flex flex-col items-center">
                    <img src="etapas_pipeline/etapa_3_crop_bruto_inclinado.jpg" class="h-44 w-full rounded-lg object-contain">
                    <div class="p-2 text-center">
                        <span class="text-xs font-bold text-amber-400">3. CROP BRUTO</span>
                        <p class="text-[11px] text-slate-400 mt-1">Placa isolada ainda inclinada e com a tarja azul.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- SLIDE 5: ETAPA 4 DESKEW HOUGH -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-emerald-400 uppercase tracking-widest">Capítulo 4</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">Etapa 4: Retificação Geométrica (Deskew)</h2>
                <p class="text-sm text-slate-400">Correção trigonométrica automática de rotação (Ganho: <strong class="text-emerald-400">+23,0%</strong>)</p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <div class="glass-card p-3 rounded-xl">
                    <span class="text-xs font-bold text-amber-400">ANTES: Placa Inclinada (+11.6°)</span>
                    <img src="etapas_pipeline/etapa_3_crop_bruto_inclinado.jpg" class="h-40 w-full rounded-lg object-contain mt-2">
                </div>
                <div class="glass-card p-3 rounded-xl">
                    <span class="text-xs font-bold text-emerald-400">DEPOIS: Placa Nivelada (0.0° Horizontal)</span>
                    <img src="etapas_pipeline/etapa_4_retificacao_deskew.jpg" class="h-40 w-full rounded-lg object-contain mt-2">
                </div>
            </div>
            <div class="glass-card p-3 rounded-xl text-xs text-slate-300">
                A transformada de Hough identifica as bordas horizontais da placa e a matriz afim desentorta a imagem antes da entrada no OCR.
            </div>
        </div>

        <!-- SLIDE 6: ETAPAS 5 E 6 MULTICONTRASTE -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-purple-400 uppercase tracking-widest">Capítulo 5</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">Etapas 5 e 6: Corte da Tarja e 4 Filtros Concorrentes</h2>
                <p class="text-sm text-slate-400">Eliminação de interferência do 'BRASIL' e robustez contra sombras e faróis (Ganho: <strong class="text-emerald-400">+24,0%</strong>)</p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <div class="glass-card p-3 rounded-xl">
                    <span class="text-xs font-bold text-purple-400">5. ISOLAMENTO DO MIOLO (SEM TARJA)</span>
                    <img src="etapas_pipeline/etapa_5_corte_tarja_brasil.jpg" class="h-44 w-full rounded-lg object-contain mt-2">
                </div>
                <div class="glass-card p-3 rounded-xl">
                    <span class="text-xs font-bold text-cyan-400">6. MULTICONTRASTE (4 FILTROS)</span>
                    <img src="etapas_pipeline/etapa_6_multicontraste_4filtros.jpg" class="h-44 w-full rounded-lg object-contain mt-2">
                </div>
            </div>
        </div>

        <!-- SLIDE 7: ETAPAS 7 E 8 CARACTERES E RESULTADO -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-emerald-400 uppercase tracking-widest">Capítulo 6</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">Etapas 7 e 8: YOLO de Caracteres e Consenso Final</h2>
                <p class="text-sm text-slate-400">Localização das caixas delimitadoras de cada letra e consulta na base de veículos roubados</p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <div class="glass-card p-3 rounded-xl">
                    <span class="text-xs font-bold text-purple-400">7. YOLO 2: CAIXAS DOS CARACTERES</span>
                    <img src="etapas_pipeline/etapa_7_yolo2_deteccao_caracteres.jpg" class="h-44 w-full rounded-lg object-contain mt-2">
                </div>
                <div class="glass-card p-3 rounded-xl">
                    <span class="text-xs font-bold text-emerald-400">8. RESULTADO ANOTADO & STATUS POLICIAL</span>
                    <img src="etapas_pipeline/etapa_8_consenso_e_consulta_policial.jpg" class="h-44 w-full rounded-lg object-contain mt-2">
                </div>
            </div>
        </div>

        <!-- SLIDE 8: YOLO 1 TREINAMENTO -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-blue-400 uppercase tracking-widest">Capítulo 7</span>
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

        <!-- SLIDE 9: YOLO 2 TREINAMENTO -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-blue-400 uppercase tracking-widest">Capítulo 8</span>
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

        <!-- SLIDE 10: O CHOQUE DE REALIDADE -->
        <div class="slide flex-col justify-center space-y-6 w-full">
            <div>
                <span class="text-xs font-bold text-red-400 uppercase tracking-widest">Capítulo 9</span>
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

        <!-- SLIDE 11: BATALHA DIRETA -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-blue-400 uppercase tracking-widest">Capítulo 10</span>
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

        <!-- SLIDE 12: EVOLUÇÃO E GANHO INCREMENTAL -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-emerald-400 uppercase tracking-widest">Capítulo 11</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">Evolução Incremental e Ganhos Marginais</h2>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <div class="glass-card p-2 rounded-xl flex items-center justify-center">
                    <img src="../relatorios_benchmark/grafico_2_evolucao_etapas.png" class="max-h-72 rounded-lg object-contain shadow-lg">
                </div>
                <div class="glass-card p-2 rounded-xl flex items-center justify-center">
                    <img src="../relatorios_benchmark/grafico_3_ganho_por_metodo.png" class="max-h-72 rounded-lg object-contain shadow-lg">
                </div>
            </div>
        </div>

        <!-- SLIDE 13: QUALIDADE DE LEITURA -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-blue-400 uppercase tracking-widest">Capítulo 12</span>
                <h2 class="text-2xl md:text-3xl font-extrabold text-white">Distribuição da Qualidade de Leitura</h2>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <div class="glass-card p-2 rounded-xl flex items-center justify-center">
                    <img src="../relatorios_benchmark/grafico_4_distribuicao_qualidade_leitura.png" class="max-h-72 rounded-lg object-contain shadow-lg">
                </div>
                <div class="space-y-3">
                    <div class="glass-card p-4 rounded-xl border border-slate-700">
                        <h4 class="text-sm font-bold text-emerald-400">Zero Erros Parciais</h4>
                        <p class="text-xs text-slate-300 mt-1">A taxa de placas com apenas 1 ou 2 caracteres divergentes foi zerada graças à desambiguação sintática FE-Schrift.</p>
                    </div>
                    <div class="glass-card p-4 rounded-xl border border-slate-700">
                        <h4 class="text-sm font-bold text-blue-400">95,8% de Confiabilidade</h4>
                        <p class="text-xs text-slate-300 mt-1">Precisão exata garantida em situações diurnas, noturnas e sob ângulos diagonais reais.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- SLIDE 14: REENGENHARIA FASTAPI -->
        <div class="slide flex-col justify-center space-y-4 w-full">
            <div>
                <span class="text-xs font-bold text-blue-400 uppercase tracking-widest">Capítulo 13</span>
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

        <!-- SLIDE 15: DASHBOARD EXECUTIVO -->
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
    print("Iniciando geração das apresentações de slides (PPTX e HTML) com imagens de todas as etapas...")
    criar_apresentacao_pptx()
    criar_apresentacao_html()
    print("\n✅ Apresentações concluídas na pasta 'apresentacao/'!")
