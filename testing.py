def validar_contrato():
    """Monta o contrato e executa a validação pelo Kraken."""
    global CONTRACT

    from kraken.contract import DeployContract

    bundle = carregar_bundle()
    CONTRACT = montar_contrato(bundle)

    dc = DeployContract(globals())
    return dc.create(CONTRACT)