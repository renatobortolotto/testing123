import inspect

from kraken.contract.validations.pipeline import gcp_validator

linhas = inspect.getsource(gcp_validator).splitlines()
termos = (
    "campos_dvry",
    "campos_big",
    "_kraken",
    "saveAsTable",
)

encontrados = [
    indice
    for indice, linha in enumerate(linhas)
    if any(termo in linha for termo in termos)
]

for indice in encontrados:
    print(f"\nTrecho próximo à linha {indice + 1}:")
    print("\n".join(linhas[max(0, indice - 8):indice + 12]))

if not encontrados:
    print("Referências não encontradas neste módulo.")