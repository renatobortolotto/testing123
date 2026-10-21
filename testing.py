# Databricks notebook source
# Substitua INTEGRALMENTE a célula Python da seção operacional:
# "2. Leitura por ID e persistência operacional".
# Não adicione como patch avulso e não altere o bloco histórico.
#
# Depois, reexecute SOMENTE o bloco operacional, desde OP_CFG até a conclusão.
# A configuração gera um novo OP_ID e impede misturar saídas parciais.
# Preserve seu id_ms03_validado atual, a lista validada e o modo TODOS/AMOSTRA.
#
# Mantém as funções de persistência e saída originais. Não altera modelos,
# scores, retenção, fontes ou tabelas existentes.
# Requer a configuração/imports do operacional já executados.

def op_sql(nome: str) -> str:
    partes = nome.split(".")
    if not 1 <= len(partes) <= 3 or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) for p in partes
    ):
        raise ValueError(f"Identificador SQL inválido: {nome!r}")
    return ".".join(f"`{p}`" for p in partes)


def op_colunas(df: DataFrame, campos: set, origem: str) -> None:
    faltam = campos - set(df.columns)
    if faltam:
        raise ValueError(f"{origem}: faltam as colunas {sorted(faltam)}.")


def op_zero(rotulo: str, df: DataFrame) -> None:
    if df.limit(1).count():
        raise RuntimeError(f"Verificação falhou: {rotulo}. Nenhuma view será atualizada.")


def op_unico(df: DataFrame, rotulo: str) -> dict:
    rows = df.limit(2).collect()
    if len(rows) != 1:
        raise RuntimeError(f"{rotulo}: esperado um único registro; encontrados {len(rows)}.")
    return rows[0].asDict(recursive=True)


def op_pequeno(df: DataFrame, limite: int, rotulo: str) -> list:
    rows = df.limit(limite + 1).collect()
    if len(rows) > limite:
        raise RuntimeError(f"{rotulo}: excede a proteção do driver ({limite} linhas).")
    return rows


def op_ler(tabela: str) -> DataFrame:
    op_sql(tabela)
    if not spark.catalog.tableExists(tabela):
        raise RuntimeError(
            f"Fonte indisponível: {tabela}. Atualize a ingestão ou o nome em OP_CFG. "
            "Não é necessário repetir o treinamento. Uma view temporária de outro "
            "notebook pode não existir nesta sessão; prefira uma tabela permanente."
        )
    return spark.table(tabela)


# Registra apenas DataFrames abertos nesta execução operacional.
# Não use este registro para reaproveitar uma execução parcial anterior.
_OP_FONTES_FIXADAS = {}
OP_CFG["politica_leitura_fontes"] = "snapshot_delta_inicio_v1"


def _op_meta_delta(tabela: str) -> dict | None:
    """Retorna identidade e versão atual, sem ler as linhas da tabela."""
    info = spark.catalog.getTable(tabela)
    if info.isTemporary or str(info.tableType).upper() == "VIEW":
        return None

    detalhe = spark.sql(
        f"DESCRIBE DETAIL {op_sql(tabela)}"
    ).first().asDict()
    if str(detalhe.get("format", "")).lower() != "delta":
        return None
    if not detalhe.get("id") or not detalhe.get("location"):
        raise RuntimeError(f"Identidade Delta incompleta: {tabela}")

    historico = spark.sql(
        f"DESCRIBE HISTORY {op_sql(tabela)} LIMIT 1"
    ).first()
    if historico is None:
        raise RuntimeError(f"Histórico Delta indisponível: {tabela}")

    return {
        "id": str(detalhe["id"]),
        "location": str(detalhe["location"]),
        "version": int(historico["version"]),
    }


def op_versao_delta(tabela: str):
    """Compatibilidade com o restante do operacional."""
    meta = _op_meta_delta(tabela)
    return None if meta is None else meta["version"]


def _op_mesma_tabela(tabela: str, inicio: dict, fim: dict | None) -> None:
    """Novos commits são permitidos; troca da tabela/localização, não."""
    if fim is None or any(
        inicio[campo] != fim[campo] for campo in ("id", "location")
    ):
        raise RuntimeError(
            f"A identidade/localização da fonte {tabela} mudou. "
            "Não publicar esta execução; investigar a substituição da tabela."
        )
    if fim["version"] < inicio["version"]:
        raise RuntimeError(
            f"O histórico de {tabela} retrocedeu. "
            "Não será usada outra tabela silenciosamente."
        )


def op_fonte_operacional(tabela: str) -> DataFrame:
    """Fixa a versão Delta vigente na primeira leitura de cada fonte."""
    if tabela in OP_FONTES:
        info = OP_FONTES[tabela]
        if (
            info.get("id_execucao_leitura") != OP_ID
            or tabela not in _OP_FONTES_FIXADAS
        ):
            raise RuntimeError(
                "Há fontes de uma execução anterior/não fixada na sessão. "
                "Reexecute desde a configuração do operacional, com novo OP_ID."
            )
        return _OP_FONTES_FIXADAS[tabela]

    df = op_ler(tabela)
    meta = _op_meta_delta(tabela)
    if meta is None:
        if tabela == OP_CFG["fonte_eventos"]:
            raise RuntimeError(
                "A fonte de eventos deve ser uma tabela Delta permanente "
                "para fixar a versão durante o scoring."
            )
        if tabela != OP_CFG["fonte_publico"]:
            raise RuntimeError(f"Fonte não Delta não prevista: {tabela}")
        # O público de uma view continua sendo materializado em _publico_hml
        # pela seção 6. Isto não certifica as versões das bases dessa view.
        info = {
            "id_execucao_leitura": OP_ID,
            "modo_leitura": "PUBLICO_SEM_VERSAO_DELTA",
            "versao_delta_inicio": None,
            "versao_delta_fim": None,
        }
        print(
            f"PÚBLICO SEM VERSÃO DELTA: {tabela}. "
            "Será persistido pela seção 6; não é um snapshot das bases da view.",
            flush=True,
        )
    else:
        versao = meta["version"]
        print(
            f"FONTE FIXADA: {tabela} | versão Delta={versao}",
            flush=True,
        )
        # Versão atual capturada nesta execução, não versão antiga do treino.
        # Falha de leitura NÃO aciona fallback para a versão mais recente.
        df = (
            spark.read.option("versionAsOf", versao)
            .table(tabela)
        )
        # Uma leitura pequena detecta bloqueios antes do processamento pesado.
        # Não comprova a disponibilidade de todos os arquivos.
        df.limit(1).collect()
        depois = _op_meta_delta(tabela)
        _op_mesma_tabela(tabela, meta, depois)
        info = {
            "id_execucao_leitura": OP_ID,
            "modo_leitura": "DELTA_SNAPSHOT_INICIO",
            "id_tabela_delta": meta["id"],
            "localizacao_delta": meta["location"],
            "versao_delta_inicio": versao,
            "versao_delta_lida": versao,
            "versao_delta_fim": depois["version"],
        }

    OP_FONTES[tabela] = info
    _OP_FONTES_FIXADAS[tabela] = df
    return df


def op_verificar_fontes() -> None:
    """Verifica a identidade sem abortar por novos commits após a leitura."""
    for tabela, info in OP_FONTES.items():
        if (
            info.get("id_execucao_leitura") != OP_ID
            or tabela not in _OP_FONTES_FIXADAS
        ):
            raise RuntimeError(
                f"{tabela} não foi aberta pelo leitor fixo desta execução. "
                "Reinicie apenas o operacional; não pule esta verificação."
            )

        modo = info.get("modo_leitura")
        if modo == "PUBLICO_SEM_VERSAO_DELTA":
            if tabela != OP_CFG["fonte_publico"]:
                raise RuntimeError("Fonte não Delta fora do contrato.")
            if _op_meta_delta(tabela) is not None:
                raise RuntimeError("O tipo da fonte de público mudou.")
            continue

        if modo != "DELTA_SNAPSHOT_INICIO":
            raise RuntimeError(f"Modo de leitura não reconhecido: {modo!r}")

        inicio = {
            "id": info["id_tabela_delta"],
            "location": info["localizacao_delta"],
            "version": info["versao_delta_lida"],
        }
        fim = _op_meta_delta(tabela)
        _op_mesma_tabela(tabela, inicio, fim)
        info["versao_delta_fim"] = fim["version"]
        info["fonte_avancou"] = fim["version"] != inicio["version"]

        if info["fonte_avancou"]:
            print(
                f"FONTE AVANÇOU: {tabela} | "
                f"versão usada={inicio['version']} | "
                f"versão atual={fim['version']}. "
                "Mantida a versão fixada; novos dados entram na próxima execução.",
                flush=True,
            )


def op_assinatura(df: DataFrame) -> dict:
    # Mesma assinatura usada ao gravar o pacote histórico, com o mesmo fuso.
    j = F.to_json(F.struct(*[F.col(c) for c in sorted(df.columns)]), {
        "ignoreNullFields": "false",
        "timestampFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX",
        "timestampNTZFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
    })
    h = df.select(F.sha2(j, 256).alias("h"))
    a = [F.count("*").alias("n")]
    for i in range(4):
        a.append(F.sum(F.conv(F.substring("h", i * 16 + 1, 16), 16, 10)
                       .cast("decimal(38,0)")).alias(f"s{i}"))
    v = h.agg(*a).first().asDict()
    return {"n": int(v["n"]), "somas_sha256": [str(v[f"s{i}"] or 0) for i in range(4)]}


def op_gravar(df: DataFrame, nome: str) -> DataFrame:
    tabela = OP_SAIDAS[nome]
    proibidas = set(OP_FONTES) | set(OP_TABELAS_MODELO)
    if tabela in proibidas:
        raise RuntimeError("Saída operacional não pode coincidir com fonte.")
    saida = df.withColumn("id_execucao", F.lit(OP_ID))
    saida = saida.withColumn("id_ms03_validado", F.lit(OP_CFG["id_ms03_validado"]))
    saida = saida.withColumn("id_pacote_modelo", F.lit(OP_META["id_pacote"]))
    saida = saida.withColumn("versao_operacional", F.lit(OP_VERSAO))
    saida = saida.withColumn("modo_scoring", F.lit(OP_CFG["modo_scoring"]))
    if spark.catalog.tableExists(tabela):
        anterior = spark.table(tabela)
        op_colunas(anterior, {"id_execucao"}, tabela)
        if anterior.filter(F.col("id_execucao") == OP_ID).limit(1).count():
            raise RuntimeError(
                f"{nome} já foi escrito neste ID. Use Executar tudo para nova execução."
            )
    (saida.write.format("delta").mode("append")
     .partitionBy("id_execucao").saveAsTable(tabela))
    return spark.table(tabela).filter(F.col("id_execucao") == OP_ID)


def op_relatorio(nome: str, df: DataFrame, ordem=None) -> None:
    OP_RELATORIOS[nome] = df
    print("\n" + nome)
    if ordem:
        df = df.orderBy(*ordem)
    for campo in df.schema.fields:
        if campo.dataType.simpleString() in {"double", "float"}:
            df = df.withColumn(campo.name, F.round(F.col(campo.name), 6))
    df.show(OP_CFG["max_linhas_exibir"], truncate=False)


# Verifica disponibilidade das duas entradas antes dos cálculos do motor.
# Apenas checagem do contrato; não utiliza views como histórico congelado.
for _fonte in (OP_CFG["fonte_publico"], OP_CFG["fonte_eventos"]):
    op_ler(_fonte)

