"""
Testes unitários para validação dos módulos:
- utils.py: validação de placas (Mercosul / Antiga) e heurísticas de correção
- database.py: inicialização, inserção e consulta de veículos roubados
"""

import os
import sys
import unittest

# Adiciona a raiz do projeto ao path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.utils import (
    limpar_texto,
    classificar_padrao_placa,
    tentar_corrigir_placa,
    formatar_placa_exibicao
)
from src.database import (
    inicializar_banco,
    consultar_placa,
    cadastrar_veiculo,
    listar_veiculos,
    normalizar_placa
)


class TesteSistemaPlacas(unittest.TestCase):

    def setUp(self):
        # Cria banco de teste em memória ou arquivo temporário
        self.db_teste = os.path.join(BASE_DIR, "data", "test_veiculos.db")
        inicializar_banco(self.db_teste)

    def tearDown(self):
        if os.path.exists(self.db_teste):
            os.remove(self.db_teste)

    def test_validacao_padrao_mercosul(self):
        self.assertEqual(classificar_padrao_placa("BRA2E19"), "Mercosul")
        self.assertEqual(classificar_padrao_placa("RIO2A18"), "Mercosul")
        self.assertEqual(classificar_padrao_placa("MER0C20"), "Mercosul")

    def test_validacao_padrao_antigo(self):
        self.assertEqual(classificar_padrao_placa("ABC1234"), "Antigo")
        self.assertEqual(classificar_padrao_placa("KGB4567"), "Antigo")
        self.assertEqual(classificar_padrao_placa("XYZ9876"), "Antigo")

    def test_correcao_heuristica_ocr(self):
        # 'O' na posição numérica virando '0'
        corrigido, padrao = tentar_corrigir_placa("ABC123O")
        self.assertEqual(corrigido, "ABC1230")
        self.assertEqual(padrao, "Antigo")

        # 'O' na última posição de placa Mercosul virando '0'
        corrigido_merc, padrao_merc = tentar_corrigir_placa("BRA2E1O")
        self.assertEqual(corrigido_merc, "BRA2E10")
        self.assertEqual(padrao_merc, "Mercosul")

    def test_formatacao_exibicao(self):
        self.assertEqual(formatar_placa_exibicao("ABC1234"), "ABC-1234")
        self.assertEqual(formatar_placa_exibicao("BRA2E19"), "BRA-2E19")

    def test_banco_consulta_veiculo_roubado(self):
        resultado = consultar_placa("BRA2E19", self.db_teste)
        self.assertIsNotNone(resultado)
        self.assertEqual(resultado["status_roubo"], 1)
        self.assertEqual(resultado["modelo"], "Corolla")

    def test_banco_consulta_veiculo_regular(self):
        resultado = consultar_placa("RIO2A18", self.db_teste)
        self.assertIsNotNone(resultado)
        self.assertEqual(resultado["status_roubo"], 0)

    def test_banco_cadastro_novo(self):
        novo = {
            "placa": "TEST123",
            "marca": "Ford",
            "modelo": "Ka",
            "cor": "Preto",
            "ano": 2019,
            "status_roubo": 1,
            "boletim_ocorrencia": "BO-9999",
            "cidade": "Santos",
            "estado": "SP"
        }
        cadastrar_veiculo(novo, self.db_teste)
        resultado = consultar_placa("TEST123", self.db_teste)
        self.assertIsNotNone(resultado)
        self.assertEqual(resultado["marca"], "Ford")


if __name__ == "__main__":
    unittest.main()
