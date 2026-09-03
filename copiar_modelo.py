import os
import glob
import shutil
from ultralytics import YOLO

downloads = "C:/Users/Admin/Downloads"
arquivos = glob.glob(os.path.join(downloads, "best*.pt"))
print(f"Arquivos encontrados em Downloads ({len(arquivos)}):")
for a in arquivos:
    print(f" - {os.path.basename(a)} ({os.path.getsize(a)/1024/1024:.2f} MB) - Modificado: {os.path.getmtime(a)}")

if arquivos:
    mais_recente = max(arquivos, key=os.path.getmtime)
    print(f"\nSelecionado mais recente: {mais_recente}")
    destino = os.path.join("models", "best.pt")
    shutil.copy2(mais_recente, destino)
    print(f"✅ Copiado com sucesso para '{destino}'!")
    
    # Valida carregamento
    print("Validando carregamento do novo modelo YOLO...")
    modelo = YOLO(destino)
    print(f"✅ Modelo carregado com sucesso! Classes: {modelo.names}")
else:
    print("❌ Nenhum arquivo best*.pt encontrado na pasta Downloads.")
