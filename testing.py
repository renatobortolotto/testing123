from pathlib import Path

import kraken

raiz = Path(kraken.__file__).resolve().parent
total = 0

for arquivo in sorted(raiz.rglob("*.py")):
    linhas = arquivo.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    for indice, linha in enumerate(linhas):
        if "campos_dvry" not in linha:
            continue

        total += 1
        print(f"\nArquivo: {arquivo.relative_to(raiz)}")
        print(f"Linha: {indice + 1}")
        inicio = max(0, indice - 8)
        fim = min(len(linhas), indice + 20)
        print("\n".join(linhas[inicio:fim]))

print(f"\nTotal de referências: {total}")