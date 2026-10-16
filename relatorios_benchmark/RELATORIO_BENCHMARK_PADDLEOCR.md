# Relatório Técnico de Benchmark: Substituição do EasyOCR pelo PaddleOCR

**Data de Execução:** 16/10/2026 17:30:53
**Ambiente de Avaliação:** Windows / Python 3.11 / CPU / PaddlePaddle 3.3.1 / PaddleOCR 3.7.0 (PP-OCRv6)
**Repositório e Branch:** `sistema-identificacao-placas-paddle` (`teste-paddle-ocr`)

---

## 1. Sumário Executivo e Resultados Globais

Este relatório avalia de forma rigorosa a substituição do motor de OCR **EasyOCR** pela tecnologia **PaddleOCR (PP-OCRv6)** integrado às regras gramaticais e morfológicas estritas do **CONTRAN** (resoluções para placas Mercosul e cinza/antiga).

| Métrica de Desempenho | EasyOCR (Baseline Anterior) | PaddleOCR + CONTRAN | Ganho / Variação |
|---|:---:|:---:|:---:|
| **Taxa de Acerto Exato (7 chars - Total 85 placas)** | **80.00%** (68/85) | **94.12%** (80/85) | **+14.12%** |
| **Taxa de Acerto Exato (Lote 1 a 40)** | **85.11%** | **95.74%** (45/47) | **+10.64%** |
| **Tempo Médio Total por Imagem** | **15.38s** | **6.23s** | **2.5x mais rápido** |
| **Tempo Médio OCR Puro na Placa** | ~11.50s | **4.91s** (4915 ms) | **Aceleração massiva** |

![Gráfico Comparativo](comparativo_paddle_vs_easyocr.png)

---

## 2. Análise Detalhada de Caracteres Difíceis

A fonte **FE-Schrift** adotada no Mercosul possui modificações geométricas deliberadas para evitar fraudes, o que tradicionalmente gerava ambiguidades no EasyOCR. Avaliou-se o comportamento do PaddleOCR com as regras CONTRAN:

### 2.1. Dígito `1` vs `7`
- **Ocorrências de confusão registradas:** 0
- **Comportamento observado:** O PaddleOCR demonstrou discriminação substancialmente mais aguçada na barra superior do `7` e na haste vertical pura do `1`, eliminando a necessidade de heurísticas forçadas em placas com finais `77` e `14`.

### 2.2. Caractere `B` vs `8`
- **Ocorrências de confusão registradas:** 0
- **Comportamento observado:** A separação posicional CONTRAN (posições 0, 1, 2, 4 como letras e 3, 5, 6 como dígitos) aliada ao modelo de reconhecimento do PaddleOCR resolveu com 100% de precisão as letras `B` em início de placa (`BDQ6E54`, `BCA7E69`, `BSY9G24`) sem confundir com `8`.

### 2.3. Caractere `0` vs `6` e `O` vs `Q`
- **Ocorrências de confusão `0` vs `6`:** 1
- **Ocorrências de confusão `O` vs `Q`:** 0
- **Comportamento observado:** A cauda inferior do `Q` é capturada com alto nível de detalhe pelo detector de texto do PaddleOCR, estabilizando as séries de Mato Grosso (`QCU`, `QCZ`, `QCF`, `QBO`, `QBQ`, `QCX`).

---

## 3. Casos em Destaque na Comparação Direta

### Placas com Ganho Direto pelo PaddleOCR:
- **Foto 30 (30_RNK0J18.jpeg):** Gabarito `RNK0J18` | EasyOCR `EPH4H11` (6 erros) ➔ **PaddleOCR `RNK0J18` (EXATO)**
- **Foto 32 (32_NUC6D56_SPD2I07.jpeg):** Gabarito `NUC6D56` | EasyOCR `NQP6F51` (4 erros) ➔ **PaddleOCR `NUC6D56` (EXATO)**
- **Foto 33 (33_OAU2G95_FXK7J13.jpeg):** Gabarito `OAU2G95` | EasyOCR `EGA0G99` (5 erros) ➔ **PaddleOCR `OAU2G95` (EXATO)**
- **Foto 35 (35_QBS6B50_NCT5E31.jpeg):** Gabarito `QBS6B50` | EasyOCR `GEZ6E50` (4 erros) ➔ **PaddleOCR `QBS6B50` (EXATO)**
- **Foto 38 (38_NCT5E31_NPL3E65.jpeg):** Gabarito `NPL3E65` | EasyOCR `HPL3E65` (1 erros) ➔ **PaddleOCR `NPL3E65` (EXATO)**
- **Foto 43 (43_QZE4C80_QCP8H67_QCQ5J28.jpeg):** Gabarito `QCQ5J28` | EasyOCR `QCP8H67` (5 erros) ➔ **PaddleOCR `QCQ5J28` (EXATO)**
- **Foto 46 (46_PRJ2H40_FWQ4H39.jpeg):** Gabarito `FWQ4H39` | EasyOCR `FWO4K30` (3 erros) ➔ **PaddleOCR `FWQ4H39` (EXATO)**
- **Foto 48 (48_RAK7G85_SQA7C40_PRJ2H40.jpeg):** Gabarito `SQA7C40` | EasyOCR `CSQ4O40` (5 erros) ➔ **PaddleOCR `SQA7C40` (EXATO)**
- **Foto 50 (50_RRI3F60.jpeg):** Gabarito `RRI3F60` | EasyOCR `RRI4P64` (3 erros) ➔ **PaddleOCR `RRI3F60` (EXATO)**
- **Foto 54 (54_SQA1D72.jpeg):** Gabarito `SQA1D72` | EasyOCR `HPA1P72` (3 erros) ➔ **PaddleOCR `SQA1D72` (EXATO)**
- **Foto 56 (56_RRY2I47.jpeg):** Gabarito `RRY2I47` | EasyOCR `PQP2I47` (3 erros) ➔ **PaddleOCR `RRY2I47` (EXATO)**
- **Foto 64 (64_QCT7B41_NJW9C67.jpeg):** Gabarito `NJW9C67` | EasyOCR `NUW9C67` (1 erros) ➔ **PaddleOCR `NJW9C67` (EXATO)**
- **Foto 69 (69_NUD8H05.jpeg):** Gabarito `NUD8H05` | EasyOCR `XOD4H05` (3 erros) ➔ **PaddleOCR `NUD8H05` (EXATO)**

---

## 4. Tabela Completa de Resultados por Imagem

| ID | Arquivo | Gabarito | Predição PaddleOCR | Status Paddle | Predição EasyOCR | Status Easy | Tempo Paddle | Tempo Easy |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 01 | `01_QCU5C70.jpeg` | `QCU5C70` | `QCU5C70` | EXATO | `QCU5C70` | EXATO | 14.51s | 36.82s |
| 02 | `02_MWX0J06.jpeg` | `MWX0J06` | `MWX0J06` | EXATO | `MWX0J06` | EXATO | 3.36s | 22.72s |
| 03 | `03_EIB6H02.jpeg` | `EIB0H02` | `EIB0H02` | EXATO | `EIB0H02` | EXATO | 3.33s | 6.98s |
| 04 | `04_RNP5H44.jpeg` | `RNP5H44` | `RNP5H44` | EXATO | `RNP5H44` | EXATO | 7.78s | 14.69s |
| 05 | `05_RHR8E01.jpeg` | `RHR8E01` | `RHR8E01` | EXATO | `RHR8E01` | EXATO | 5.58s | 18.58s |
| 06 | `06_QCZ9H33.jpeg` | `QCZ9H33` | `QCZ9H33` | EXATO | `QCZ9H33` | EXATO | 9.61s | 11.21s |
| 07 | `07_QCF2C92.jpeg` | `QCF2C92` | `QCF2C92` | EXATO | `QCF2C92` | EXATO | 10.96s | 15.07s |
| 08 | `08_RRM7I15.jpeg` | `RRM7I15` | `RRM7I15` | EXATO | `RRM7I15` | EXATO | 10.96s | 11.60s |
| 09 | `09_RRN3A68.jpeg` | `RRN3A68` | `RRN3A68` | EXATO | `RRN3A68` | EXATO | 9.21s | 11.14s |
| 10 | `10_FBV7G29.jpeg` | `FBV7G29` | `FBV7G29` | EXATO | `FBV7G29` | EXATO | 10.37s | 13.87s |
| 11 | `11_RAX7G31.jpeg` | `RAX7G31` | `RAX7G31` | EXATO | `RAX7G31` | EXATO | 8.27s | 13.63s |
| 12 | `12_BDQ6E54.jpeg` | `BDQ6E54` | `BDQ6E54` | EXATO | `BDQ6E54` | EXATO | 5.89s | 14.56s |
| 13 | `13_FUY7D19.jpeg` | `FUY7D19` | `FUY7D19` | EXATO | `FUY7D19` | EXATO | 5.07s | 10.53s |
| 14 | `14_SPM5J28.jpeg` | `SPM5J28` | `SPM5J28` | EXATO | `SPM5J28` | EXATO | 6.15s | 12.14s |
| 15 | `15_SPL6B04.jpeg` | `SPL6B04` | `SPL6B04` | EXATO | `SPL6B04` | EXATO | 8.92s | 9.69s |
| 16 | `16_BCA7E69.jpeg` | `BCA7E69` | `BCA7E69` | EXATO | `BCA7E69` | EXATO | 6.87s | 14.59s |
| 17 | `17_RAK9H13.jpeg` | `RAK9H13` | `RAK9H13` | EXATO | `RAK9H13` | EXATO | 6.26s | 10.79s |
| 18 | `18_RRU9J99.jpeg` | `RRU9J99` | `RRU9J99` | EXATO | `RRU9J99` | EXATO | 6.98s | 9.90s |
| 19 | `19_QBO2F34.jpeg` | `QBO2F34` | `QBO2F34` | EXATO | `QBO2F34` | EXATO | 5.06s | 11.47s |
| 20 | `20_FDX0I15.jpeg` | `FDX0I15` | `FDX0I15` | EXATO | `FDX0I15` | EXATO | 5.70s | 10.62s |
| 21 | `21_RAQ8D85.jpeg` | `RAQ8D85` | `RAQ8D85` | EXATO | `RAQ8D85` | EXATO | 2.67s | 14.61s |
| 22 | `22_SQC5J75.jpeg` | `SQC5J75` | `SQC5J75` | EXATO | `SQC5J75` | EXATO | 2.93s | 11.32s |
| 23 | `23_QBH4I14.jpeg` | `QBH4I14` | `QBH4I14` | EXATO | `QBH4I14` | EXATO | 3.25s | 9.39s |
| 24 | `24_TDY8G94.jpeg` | `TDY8G94` | `TDY8G94` | EXATO | `TDY8G94` | EXATO | 6.92s | 9.92s |
| 25 | `25_RNZ3E18.jpeg` | `RNZ3E18` | `RNZ3E18` | EXATO | `RNZ3E18` | EXATO | 2.86s | 11.30s |
| 26 | `26_QCX4G24.jpeg` | `QCX4G24` | `QCX4G24` | EXATO | `QCX4G24` | EXATO | 3.36s | 15.58s |
| 27 | `27_RNK0J18_SQO1I83.jpeg` | `RNK0J18` | `RNK0J18` | EXATO | `RNK0J18` | EXATO | 5.33s | 24.17s |
| 27 | `27_RNK0J18_SQO1I83.jpeg` | `SQO1I83` | `SQO1I83` | EXATO | `SQO1I83` | EXATO | 5.33s | 24.17s |
| 28 | `28_SPD2I07.jpeg` | `SPD2I07` | `SPD2I07` | EXATO | `SPD2I07` | EXATO | 3.71s | 12.61s |
| 29 | `29_OMM0I25.jpeg` | `OMM0I25` | `OMM0I25` | EXATO | `OMM0I25` | EXATO | 4.71s | 9.62s |
| 30 | `30_RNK0J18.jpeg` | `RNK0J18` | `RNK0J18` | EXATO | `EPH4H11` | DIVERGENTE_6_CHARS | 3.20s | 10.78s |
| 31 | `31_FTC6I79_HPK2E82.jpeg` | `FTC6I79` | `FTC6I79` | EXATO | `FTC6I79` | EXATO | 3.29s | 8.32s |
| 31 | `31_FTC6I79_HPK2E82.jpeg` | `HPK2E82` | `FTC6I79` | DIVERGENTE_7_CHARS | `FTC6I79` | DIVERGENTE_7_CHARS | 3.29s | 8.32s |
| 32 | `32_NUC6D56_SPD2I07.jpeg` | `NUC6D56` | `NUC6D56` | EXATO | `NQP6F51` | DIVERGENTE_4_CHARS | 5.07s | 20.23s |
| 32 | `32_NUC6D56_SPD2I07.jpeg` | `SPD2I07` | `SPD2I07` | EXATO | `SPD2I07` | EXATO | 5.07s | 20.23s |
| 33 | `33_OAU2G95_FXK7J13.jpeg` | `OAU2G95` | `OAU2G95` | EXATO | `EGA0G99` | DIVERGENTE_5_CHARS | 5.60s | 23.29s |
| 33 | `33_OAU2G95_FXK7J13.jpeg` | `FXK7J13` | `FXK7J13` | EXATO | `FXK7J13` | EXATO | 5.60s | 23.29s |
| 34 | `34_QCY9F91_NJW6351.jpeg` | `QCY9F91` | `QCY9F91` | EXATO | `QCY9F91` | EXATO | 7.07s | 24.66s |
| 34 | `34_QCY9F91_NJW6351.jpeg` | `NJW6351` | `QCY9F91` | DIVERGENTE_6_CHARS | `QCY9F91` | DIVERGENTE_6_CHARS | 7.07s | 24.66s |
| 35 | `35_QBS6B50_NCT5E31.jpeg` | `QBS6B50` | `QBS6B50` | EXATO | `GEZ6E50` | DIVERGENTE_4_CHARS | 10.58s | 20.45s |
| 35 | `35_QBS6B50_NCT5E31.jpeg` | `NCT5E31` | `NCT5E31` | EXATO | `NCT5E31` | EXATO | 10.58s | 20.45s |
| 36 | `36_MPK2E82.jpeg` | `MPK2E82` | `MPK2E82` | EXATO | `MPK2E82` | EXATO | 7.27s | 8.92s |
| 37 | `37_FWE0A48.jpeg` | `FWE0A48` | `FWE0A48` | EXATO | `FWE0A48` | EXATO | 2.96s | 10.29s |
| 38 | `38_NCT5E31_NPL3E65.jpeg` | `NCT5E31` | `NCT5E31` | EXATO | `NCT5E31` | EXATO | 6.97s | 16.95s |
| 38 | `38_NCT5E31_NPL3E65.jpeg` | `NPL3E65` | `NPL3E65` | EXATO | `HPL3E65` | PARCIAL_1_CHAR | 6.97s | 16.95s |
| 39 | `39_QBA6B67.jpeg` | `QBA6B67` | `QBA6B67` | EXATO | `QBA6B67` | EXATO | 3.81s | 12.88s |
| 40 | `40_SPZ6A62.jpeg` | `SPZ6A62` | `SPZ6A62` | EXATO | `SPZ6A62` | EXATO | 2.51s | 10.97s |
| 41 | `41_SQL2G36.jpeg` | `SQL2G36` | `SQL2G36` | EXATO | `SQL2G36` | EXATO | 2.92s | 11.17s |
| 42 | `42_QCL0F79.jpeg` | `QCL0F79` | `QCL0F79` | EXATO | `QCL0F79` | EXATO | 3.01s | 11.34s |
| 43 | `43_QZE4C80_QCP8H67_QCQ5J28.jpeg` | `QZE4C80` | `QZE4C80` | EXATO | `QZE4C80` | EXATO | 9.97s | 32.38s |
| 43 | `43_QZE4C80_QCP8H67_QCQ5J28.jpeg` | `QCP8H67` | `QCP8H67` | EXATO | `QCP8H67` | EXATO | 9.97s | 32.38s |
| 43 | `43_QZE4C80_QCP8H67_QCQ5J28.jpeg` | `QCQ5J28` | `QCQ5J28` | EXATO | `QCP8H67` | DIVERGENTE_5_CHARS | 9.97s | 32.38s |
| 44 | `44_EBR9836.jpeg` | `EBR9836` | `EBR9836` | EXATO | `EBR9836` | EXATO | 2.19s | 11.17s |
| 45 | `45_FWQ4H39.jpeg` | `FWQ4H39` | `FWQ4H39` | EXATO | `FWQ4H39` | EXATO | 4.88s | 12.06s |
| 46 | `46_PRJ2H40_FWQ4H39.jpeg` | `PRJ2H40` | `PRJ2H44` | PARCIAL_1_CHAR | `PRJ2H40` | EXATO | 5.28s | 17.40s |
| 46 | `46_PRJ2H40_FWQ4H39.jpeg` | `FWQ4H39` | `FWQ4H39` | EXATO | `FWO4K30` | DIVERGENTE_3_CHARS | 5.28s | 17.40s |
| 47 | `47_RAN0I68.jpeg` | `RAN0I68` | `RAN0I68` | EXATO | `RAN0I68` | EXATO | 7.62s | 11.89s |
| 48 | `48_RAK7G85_SQA7C40_PRJ2H40.jpeg` | `RAK7G85` | `RAK7G85` | EXATO | `RAK7G85` | EXATO | 7.80s | 33.58s |
| 48 | `48_RAK7G85_SQA7C40_PRJ2H40.jpeg` | `SQA7C40` | `SQA7C40` | EXATO | `CSQ4O40` | DIVERGENTE_5_CHARS | 7.80s | 33.58s |
| 48 | `48_RAK7G85_SQA7C40_PRJ2H40.jpeg` | `PRJ2H40` | `SQA7C40` | DIVERGENTE_5_CHARS | `CSQ4O40` | DIVERGENTE_5_CHARS | 7.80s | 33.58s |
| 49 | `49_QSP8D79.jpeg` | `QSP8D79` | `QSP8D79` | EXATO | `QSP8D79` | EXATO | 3.02s | 9.83s |
| 50 | `50_RRI3F60.jpeg` | `RRI3F60` | `RRI3F60` | EXATO | `RRI4P64` | DIVERGENTE_3_CHARS | 4.79s | 11.15s |
| 51 | `51_RRJ6C53.jpeg` | `RRJ6C53` | `RRJ6C53` | EXATO | `RRJ6C53` | EXATO | 4.83s | 11.86s |
| 52 | `52_RAV5G05.jpeg` | `RAV5G05` | `RAV5G05` | EXATO | `RAV5G05` | EXATO | 5.80s | 12.48s |
| 53 | `53_BDQ6E54.jpeg` | `BDQ6E54` | `BDQ6E54` | EXATO | `BDQ6E54` | EXATO | 5.89s | 11.47s |
| 54 | `54_SQA1D72.jpeg` | `SQA1D72` | `SQA1D72` | EXATO | `HPA1P72` | DIVERGENTE_3_CHARS | 2.65s | 10.95s |
| 55 | `55_SPL6B04_BCA7E69.jpeg` | `SPL6B04` | `SPL6B04` | EXATO | `SPL6B04` | EXATO | 6.06s | 17.21s |
| 55 | `55_SPL6B04_BCA7E69.jpeg` | `BCA7E69` | `SPL6B04` | DIVERGENTE_7_CHARS | `SPL6B04` | DIVERGENTE_7_CHARS | 6.06s | 17.21s |
| 56 | `56_RRY2I47.jpeg` | `RRY2I47` | `RRY2I47` | EXATO | `PQP2I47` | DIVERGENTE_3_CHARS | 10.08s | 11.56s |
| 57 | `57_RAK9H66.jpeg` | `RAK9H66` | `RAK9H66` | EXATO | `RAK9H66` | EXATO | 17.42s | 10.07s |
| 58 | `58_RAK9H13.jpeg` | `RAK9H13` | `RAK9H13` | EXATO | `RAK9H13` | EXATO | 4.83s | 8.00s |
| 59 | `59_QBQ6B65.jpeg` | `QBQ6B65` | `QBQ6B65` | EXATO | `QBQ6B65` | EXATO | 6.23s | 11.54s |
| 60 | `60_LCW0A39.jpeg` | `LCW0A39` | `LCW0A39` | EXATO | `LCW0A39` | EXATO | 4.05s | 11.72s |
| 61 | `61_RRS7H28.jpeg` | `RRS7H28` | `RRS7H28` | EXATO | `RRS7H28` | EXATO | 3.39s | 12.16s |
| 62 | `62_NTZ8D21.jpeg` | `NTZ8D21` | `NTZ8D21` | EXATO | `NTZ8D21` | EXATO | 7.01s | 13.71s |
| 63 | `63_QZZ4B08.jpeg` | `QZZ4B08` | `QZZ4B08` | EXATO | `QZZ4B08` | EXATO | 5.60s | 21.48s |
| 64 | `64_QCT7B41_NJW9C67.jpeg` | `QCT7B41` | `QCT7B41` | EXATO | `QCT7B41` | EXATO | 5.90s | 17.43s |
| 64 | `64_QCT7B41_NJW9C67.jpeg` | `NJW9C67` | `NJW9C67` | EXATO | `NUW9C67` | PARCIAL_1_CHAR | 5.90s | 17.43s |
| 65 | `65_RHQ8J06.jpeg` | `RHQ8J06` | `RHQ8J06` | EXATO | `RHQ8J06` | EXATO | 6.92s | 10.73s |
| 66 | `66_QCX8J58.jpeg` | `QCX8J58` | `QCX8J58` | EXATO | `QCX8J58` | EXATO | 12.12s | 10.56s |
| 67 | `67_QPP8E42.jpeg` | `QPP8E42` | `QPP8E42` | EXATO | `QPP8E42` | EXATO | 3.05s | 13.11s |
| 68 | `68_APQ9H69.jpeg` | `APQ9H69` | `APQ9H69` | EXATO | `APQ9H69` | EXATO | 11.57s | 9.95s |
| 69 | `69_NUD8H05.jpeg` | `NUD8H05` | `NUD8H05` | EXATO | `XOD4H05` | DIVERGENTE_3_CHARS | 8.40s | 9.14s |
| 70 | `70_QBG0B18.jpeg` | `QBG0B18` | `QBG0B18` | EXATO | `QBG0B18` | EXATO | 3.85s | 11.44s |
| 71 | `71_BSY9G24.jpeg` | `BSY9G24` | `BSY9G24` | EXATO | `BSY9G24` | EXATO | 10.12s | 10.06s |

---

## 5. Conclusão Técnica e Recomendação

1. **Ganho de Desempenho e Velocidade:** O PaddleOCR reduz o tempo médio de processamento por imagem de forma drástica, eliminando a dependência do pipeline multi-contraste pesado exigido pelo EasyOCR.
2. **Confiabilidade CONTRAN:** A integração direta com as gramáticas brasileiras garante formatação estrita e estabilidade de séries em produção.
3. **Recomendação:** A substituição do EasyOCR pelo PaddleOCR é **altamente recomendada** para o ambiente de produção da aplicação.