"""
Script de Treinamento Local: YOLO de Reconhecimento de Caracteres de Placas Brasileiras.
Suporta datasets do Roboflow Universe e datasets locais em formato YOLO.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# Diretórios base do projeto
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "data" / "datasets_caracteres"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def baixar_dataset_roboflow(api_key: str, workspace: str, project_name: str, version: int = 1) -> str:
    """Baixa um dataset do Roboflow Universe no formato YOLOv8."""
    print(f"📦 Conectando ao Roboflow ({workspace}/{project_name} v{version})...")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        dataset = project.version(version).download("yolov8", location=str(DATASETS_DIR / project_name))
        print(f"✅ Dataset baixado em: {dataset.location}")
        return os.path.join(dataset.location, "data.yaml")
    except ImportError:
        print("❌ Biblioteca 'roboflow' não instalada. Execute: pip install roboflow")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro ao baixar dataset do Roboflow: {e}")
        sys.exit(1)


def treinar_yolo_caracteres(
    data_yaml_path: str,
    epochs: int = 50,
    img_size: int = 320,
    batch_size: int = 16,
    model_base: str = "yolov8n.pt",
    device: str = ""
):
    """Executa o treinamento do YOLO para reconhecimento de caracteres."""
    from ultralytics import YOLO

    if not os.path.exists(data_yaml_path):
        print(f"❌ Arquivo de configuração não encontrado: {data_yaml_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🚀 INICIANDO TREINAMENTO DO YOLO DE CARACTERES")
    print(f"📄 Configuração: {data_yaml_path}")
    print(f"⚙️ Épocas: {epochs} | Batch: {batch_size} | Tamanho Imagem: {img_size}px")
    print(f"🧠 Modelo Base: {model_base}")
    print("=" * 60 + "\n")

    modelo = YOLO(model_base)

    # Treinamento com Data Augmentation automático do YOLO (cores, escala, rotação leve)
    resultados = modelo.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=15,
        save=True,
        device=device if device else None,
        project=str(BASE_DIR / "runs" / "caracteres"),
        name="treino_caracteres",
        exist_ok=True,
        # Hiperparâmetros de Data Augmentation para robustez a reflexos e sombras
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=5.0,
        scale=0.2,
        perspective=0.0005,
        fliplr=0.0,  # Não espelhar horizontalmente letras/números!
        mosaic=0.5
    )

    # Copia o melhor modelo para models/detector_caracteres.pt
    best_pt = BASE_DIR / "runs" / "caracteres" / "treino_caracteres" / "weights" / "best.pt"
    destino_final = MODELS_DIR / "detector_caracteres.pt"

    if best_pt.exists():
        shutil.copy(best_pt, destino_final)
        print("\n" + "=" * 60)
        print(f"🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print(f"🏆 Modelo salvo em: {destino_final}")
        print("=" * 60)
    else:
        print("⚠️ Não foi possível encontrar os pesos em 'best.pt'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinador de YOLO para Caracteres de Placas")
    parser.add_argument("--data", type=str, default="", help="Caminho para o data.yaml local")
    parser.add_argument("--epochs", type=int, default=40, help="Número de épocas de treinamento")
    parser.add_argument("--batch", type=int, default=16, help="Tamanho do batch")
    parser.add_argument("--imgsz", type=int, default=320, help="Resolução das imagens de entrada")
    parser.add_argument("--roboflow_key", type=str, default="", help="Chave de API do Roboflow")
    parser.add_argument("--roboflow_workspace", type=str, default="", help="Workspace do Roboflow")
    parser.add_argument("--roboflow_project", type=str, default="", help="Projeto do Roboflow")
    parser.add_argument("--roboflow_version", type=int, default=1, help="Versão do dataset Roboflow")

    args = parser.parse_args()

    data_path = args.data
    if args.roboflow_key and args.roboflow_workspace and args.roboflow_project:
        data_path = baixar_dataset_roboflow(
            api_key=args.roboflow_key,
            workspace=args.roboflow_workspace,
            project_name=args.roboflow_project,
            version=args.roboflow_version
        )

    if not data_path:
        yaml_encontrados = list(DATASETS_DIR.glob("**/data.yaml"))
        if yaml_encontrados:
            data_path = str(yaml_encontrados[0])
            print(f"🔍 Usando dataset encontrado automaticamente: {data_path}")
        else:
            print("ℹ️ Dica: Forneça o caminho do data.yaml com '--data caminho/data.yaml' ou informe as credenciais do Roboflow.")
            sys.exit(0)

    treinar_yolo_caracteres(
        data_yaml_path=data_path,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch_size=args.batch
    )
