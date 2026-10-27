safra_teste = "20261026"
inicio_teste, fim_teste = calcular_janela(safra_teste)

query_teste = montar_query_features(
    safra_teste,
    list(FEATURE_DESCRIPTIONS),
)

print("Início:", inicio_teste)
print("Fim:", fim_teste)

termos = (
    "entrada_referencia",
    "2026-10",
    "__inicio__",
    "__fim__",
    "current_date",
)

for linha in query_teste.splitlines():
    if any(termo in linha.lower() for termo in termos):
        print(repr(linha))