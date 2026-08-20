"""
Módulo de Banco de Dados para Consulta e Cadastro de Veículos.
Utiliza SQLite para armazenar o registro de veículos e status de roubo/furto.
"""

import sqlite3
import os
from typing import Optional, Dict, Any, List

# Caminho padrão para o banco de dados
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "veiculos.db")


def obter_conexao(caminho_db: str = DB_PATH) -> sqlite3.Connection:
    """Retorna uma conexão com o banco de dados SQLite."""
    os.makedirs(os.path.dirname(caminho_db), exist_ok=True)
    conn = sqlite3.connect(caminho_db)
    conn.row_factory = sqlite3.Row  # Permite acessar colunas pelo nome
    return conn


def inicializar_banco(caminho_db: str = DB_PATH) -> None:
    """Cria a tabela de veículos se não existir e popula com dados de teste."""
    conn = obter_conexao(caminho_db)
    cursor = conn.cursor()

    # Criação da tabela de veículos
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS veiculos (
        placa TEXT PRIMARY KEY,
        marca TEXT NOT NULL,
        modelo TEXT NOT NULL,
        cor TEXT NOT NULL,
        ano INTEGER NOT NULL,
        status_roubo INTEGER NOT NULL DEFAULT 0, -- 0: Regular, 1: Roubado/Furtado
        data_ocorrencia TEXT,
        boletim_ocorrencia TEXT,
        cidade TEXT,
        estado TEXT
    )
    """)

    # Verifica se já existem dados cadastrados
    cursor.execute("SELECT COUNT(*) FROM veiculos")
    if cursor.fetchone()[0] == 0:
        # Inserção de dados simulados para teste
        dados_iniciais = [
            ("BRA2E19", "Toyota", "Corolla", "Prata", 2021, 1, "2024-05-10", "BO-2024-10293", "São Paulo", "SP"),
            ("ABC1234", "Volkswagen", "Gol", "Branco", 2018, 1, "2024-04-22", "BO-2024-08472", "Campinas", "SP"),
            ("RIO2A18", "Hyundai", "HB20", "Preto", 2022, 0, None, None, "Rio de Janeiro", "RJ"),
            ("KGB4567", "Chevrolet", "Onix", "Vermelho", 2020, 1, "2024-06-01", "BO-2024-11409", "Belo Horizonte", "MG"),
            ("XYZ9876", "Fiat", "Uno", "Azul", 2015, 0, None, None, "Curitiba", "PR"),
            ("MER0C20", "Jeep", "Compass", "Cinza", 2023, 0, None, None, "Brasília", "DF"),
            ("SPX9I99", "Honda", "Civic", "Preto", 2019, 1, "2024-06-15", "BO-2024-12550", "Santos", "SP"),
        ]

        cursor.executemany("""
        INSERT INTO veiculos (placa, marca, modelo, cor, ano, status_roubo, data_ocorrencia, boletim_ocorrencia, cidade, estado)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, dados_iniciais)

        conn.commit()

    conn.close()


def normalizar_placa(placa: str) -> str:
    """Remove caracteres especiais, espaços e converte para maiúsculo."""
    return "".join(c for c in placa if c.isalnum()).upper()


def consultar_placa(placa: str, caminho_db: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Consulta uma placa no banco de dados.
    Retorna um dicionário com os dados do veículo ou None se não encontrado.
    """
    placa_limpa = normalizar_placa(placa)
    conn = obter_conexao(caminho_db)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM veiculos WHERE placa = ?", (placa_limpa,))
    linha = cursor.fetchone()
    conn.close()

    if linha:
        return dict(linha)
    return None


def cadastrar_veiculo(dados: Dict[str, Any], caminho_db: str = DB_PATH) -> bool:
    """Cadastra ou atualiza as informações de um veículo no banco."""
    conn = obter_conexao(caminho_db)
    cursor = conn.cursor()

    placa_limpa = normalizar_placa(dados["placa"])

    cursor.execute("""
    INSERT OR REPLACE INTO veiculos (
        placa, marca, modelo, cor, ano, status_roubo, data_ocorrencia, boletim_ocorrencia, cidade, estado
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        placa_limpa,
        dados.get("marca", "Desconhecida"),
        dados.get("modelo", "Desconhecido"),
        dados.get("cor", "Desconhecida"),
        dados.get("ano", 2020),
        dados.get("status_roubo", 0),
        dados.get("data_ocorrencia"),
        dados.get("boletim_ocorrencia"),
        dados.get("cidade", "Não informada"),
        dados.get("estado", "UF")
    ))

    conn.commit()
    conn.close()
    return True


def listar_veiculos(caminho_db: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retorna todos os veículos cadastrados no banco."""
    conn = obter_conexao(caminho_db)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM veiculos ORDER BY status_roubo DESC, placa ASC")
    linhas = cursor.fetchall()
    conn.close()
    return [dict(linha) for linha in linhas]
