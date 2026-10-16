import os
import shutil

# Mapeamento do gabarito visual (inspecionado imagem por imagem via Vision nativo)
RENOMEACOES = {
    "WhatsApp Image 2026-10-07 at 21.36.32.jpeg": "01_QCU5C70.jpeg",
    "WhatsApp Image 2026-10-07 at 21.36.35.jpeg": "02_MWX0J06.jpeg",
    "WhatsApp Image 2026-10-07 at 21.36.39.jpeg": "03_EIB6H02.jpeg",
    "WhatsApp Image 2026-10-07 at 21.36.42.jpeg": "04_RNP5H44.jpeg",
    "WhatsApp Image 2026-10-07 at 21.36.46.jpeg": "05_RHR8E01.jpeg",
    "WhatsApp Image 2026-10-07 at 21.36.52.jpeg": "06_QCZ9H33.jpeg",
    "WhatsApp Image 2026-10-07 at 21.36.56.jpeg": "07_QCF2C92.jpeg",
    "WhatsApp Image 2026-10-07 at 21.36.58.jpeg": "08_RRM7I15.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.01.jpeg": "09_RRN3A68.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.05.jpeg": "10_FBV7G29.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.07.jpeg": "11_RAX7G31.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.10.jpeg": "12_BDQ6E54.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.13.jpeg": "13_FUY7D19.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.16.jpeg": "14_SPM5J28.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.19.jpeg": "15_SPL6B04.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.25.jpeg": "16_BCA7E69.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.43.jpeg": "17_RAK9H13.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.47.jpeg": "18_RRU9J99.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.48 (1).jpeg": "19_QBO2F34.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.48.jpeg": "20_FDX0I15.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.49 (1).jpeg": "21_RAQ8D85.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.49.jpeg": "22_SQC5J75.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.50 (1).jpeg": "23_QBH4I14.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.50.jpeg": "24_TDY8G94.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.51 (1).jpeg": "25_RNZ3E18.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.51.jpeg": "26_QCX4G24.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.52 (1).jpeg": "27_RNK0J18_SQO1I83.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.52 (2).jpeg": "28_SPD2I07.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.52.jpeg": "29_OMM0I25.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.53.jpeg": "30_RNK0J18.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.56 (1).jpeg": "31_FTC6I79_HPK2E82.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.56.jpeg": "32_NUC6D56_SPD2I07.jpeg",
    "WhatsApp Image 2026-10-07 at 21.37.57.jpeg": "33_OAU2G95_FXK7J13.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.40.jpeg": "34_QCY9F91_NJW6351.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.41 (1).jpeg": "35_QBS6B50_NCT5E31.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.41.jpeg": "36_MPK2E82.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.42 (1).jpeg": "37_FWE0A48.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.42.jpeg": "38_NCT5E31_NPL3E65.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.43 (1).jpeg": "39_QBA6B67.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.43 (2).jpeg": "40_SPZ6A62.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.43.jpeg": "41_SQL2G36.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.44 (1).jpeg": "42_QCL0F79.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.44 (2).jpeg": "43_QZE4C80_QCP8H67_QCQ5J28.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.44.jpeg": "44_EBR9836.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.45 (1).jpeg": "45_FWQ4H39.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.45 (2).jpeg": "46_PRJ2H40_FWQ4H39.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.45.jpeg": "47_RAN0I68.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.46 (1).jpeg": "48_RAK7G85_SQA7C40_PRJ2H40.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.46.jpeg": "49_QSP8D79.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.47 (1).jpeg": "50_RRI3F60.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.47 (2).jpeg": "51_RRJ6C53.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.47.jpeg": "52_RAV5G05.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.48 (1).jpeg": "53_BDQ6E54.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.48.jpeg": "54_SQA1D72.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.49 (1).jpeg": "55_SPL6B04_BCA7E69.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.49.jpeg": "56_RRY2I47.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.50 (1).jpeg": "57_RAK9H66.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.50.jpeg": "58_RAK9H13.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.51 (1).jpeg": "59_QBQ6B65.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.51 (2).jpeg": "60_LCW0A39.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.51.jpeg": "61_RRS7H28.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.52 (1).jpeg": "62_NTZ8D21.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.52.jpeg": "63_QZZ4B08.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.53 (1).jpeg": "64_QCT7B41_NJW9C67.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.53.jpeg": "65_RHQ8J06.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.54 (1).jpeg": "66_QCX8J58.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.54 (2).jpeg": "67_QPP8E42.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.54.jpeg": "68_APQ9H69.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.55 (1).jpeg": "69_NUD8H05.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.55.jpeg": "70_QBG0B18.jpeg",
    "WhatsApp Image 2026-10-14 at 20.32.56.jpeg": "71_BSY9G24.jpeg"
}

pasta_fotos = r"data/drive_fotos/Fotos placas"

sucesso = 0
falhas = 0

for original, novo in RENOMEACOES.items():
    orig_path = os.path.join(pasta_fotos, original)
    dest_path = os.path.join(pasta_fotos, novo)
    
    if os.path.exists(orig_path):
        os.rename(orig_path, dest_path)
        sucesso += 1
    else:
        if os.path.exists(dest_path):
            sucesso += 1
        else:
            print(f"Não encontrado: {original}")
            falhas += 1

print(f"Processamento concluído: {sucesso} renomeados com sucesso, {falhas} falhas.")
