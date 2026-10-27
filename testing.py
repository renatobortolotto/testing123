import sys
import unittest

nomes_testes = list(CONTRACT["test_dataprep"])

suite_diagnostico = unittest.TestSuite([
    unittest.FunctionTestCase(
        globals()[nome],
        description=nome,
    )
    for nome in nomes_testes
])

resultado_testes = unittest.TextTestRunner(
    stream=sys.stdout,
    verbosity=2,
).run(suite_diagnostico)