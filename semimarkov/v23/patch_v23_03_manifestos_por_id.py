# Databricks notebook source
# Cole esta célula imediatamente ANTES da seção 3 da Parte 03.
# Pré-requisito: as seções 1 e 2 da Parte 03 já foram executadas.
# Apenas manifestos são lidos no estado atual por ID + hash.
# Base, catálogo, suporte, modelos e validação conservam o leitor histórico.

import hashlib
import json

from pyspark.sql import functions as F


necessarios = (
    "spark", "MS03_CFG", "MS03_FONTES", "MS03_CHECKS", "ms03_ler",
    "ms03_sql", "ms_hash",
)
ausentes = [nome for nome in necessarios if nome not in globals()]
if ausentes:
    raise RuntimeError(
        f"Execute as seções 1 e 2 da Parte 03 antes do patch: {ausentes}"
    )

# Usa a 02C executada nesta sessão. Não procura o maior/mais recente ID.
# Em outra sessão, usa o id_02c explicitamente configurado em MS03_CFG.
_id_02c = str(globals().get("C02_ID") or MS03_CFG["id_02c"]).strip()


def _ms03_manifesto_atual(tabela, filtros, rotulo):
    """Materializa uma única linha; detecta alteração concorrente da tabela."""
    nome_sql = ms03_sql(tabela)
    if not spark.catalog.tableExists(tabela):
        raise RuntimeError(f"Tabela não encontrada: {tabela}")

    def ultima_versao():
        return int(spark.sql(
            f"DESCRIBE HISTORY {nome_sql} LIMIT 1"
        ).first()["version"])

    versao_antes = ultima_versao()
    df = spark.table(tabela)
    faltantes = set(filtros) - set(df.columns)
    if faltantes:
        raise RuntimeError(f"{rotulo}: colunas ausentes: {sorted(faltantes)}")
    for coluna, valor in filtros.items():
        df = df.filter(F.col(coluna) == F.lit(valor))
    linhas = df.limit(2).collect()
    versao_depois = ultima_versao()
    if versao_antes != versao_depois:
        raise RuntimeError(
            f"{rotulo}: a tabela mudou durante a leitura. Reexecute só o patch."
        )
    if len(linhas) != 1:
        raise RuntimeError(
            f"{rotulo}: esperado um único registro concluído para {filtros}. "
            "Não será escolhido outro experimento automaticamente."
        )
    # Os dados deste registro ficam materializados, não dependem de time travel.
    return (
        linhas[0].asDict(recursive=True),
        spark.createDataFrame(linhas, df.schema),
        versao_depois,
    )


_tab_02c = MS03_CFG["tabela_execucoes_02c"]
_man_02c, _df_man_02c, _v_man_02c = _ms03_manifesto_atual(
    _tab_02c,
    {"id_02c": _id_02c, "status": "CONCLUIDO_COMPARACAO_02C"},
    "Manifesto 02C",
)
if _man_02c["versao_02c"] != "v2.3_02c_memoria_tempo_compartilhado_v1":
    raise RuntimeError("Contrato da 02C diferente do esperado pela Parte 03.")
_cfg_02c = json.loads(_man_02c["config_json"])
if ms_hash(_cfg_02c) != _man_02c["hash_02c"]:
    raise RuntimeError("Hash do conteúdo da configuração 02C divergente.")
_exp = str(_man_02c["id_experimento"])
_fit_id = str(_man_02c["id_ajuste_fonte"])
if (
    str(_cfg_02c["config"]["id_experimento"]) != _exp
    or str(_cfg_02c["config"]["id_ajuste"]) != _fit_id
):
    raise RuntimeError("IDs do manifesto 02C divergem da configuração registrada.")

_tab_fit = _cfg_02c["config"]["tabela_ajustes"]
if _tab_fit == _tab_02c:
    raise RuntimeError("As duas tabelas de manifesto deveriam ser distintas.")
_man_fit, _df_man_fit, _v_man_fit = _ms03_manifesto_atual(
    _tab_fit,
    {
        "id_experimento": _exp,
        "id_ajuste": _fit_id,
        "status": "CONCLUIDO_TREINO_VALIDACAO",
    },
    "Manifesto do ajuste",
)
if _man_fit["hash_ajuste"] != _cfg_02c["hash_ajuste_fonte"]:
    raise RuntimeError("Hash do ajuste diferente daquele registrado na 02C.")

# Reproduz exatamente a assinatura que a Parte 02 calculava.
# Essa assinatura valida a configuração assinada, NÃO é um hash de toda a base.
_cfg_fit = json.loads(_man_fit["config_json"])
_assinatura_fit = {
    "versao": _cfg_fit["versao_ajuste"],
    "numerico": _cfg_fit["numerico"],
    "politica_preparo": _cfg_fit["fonte_preparo"]["politica_sha256"],
    "idades": _cfg_fit["idades_seg"],
}
_hash_fit = hashlib.sha256(
    json.dumps(_assinatura_fit, sort_keys=True).encode("utf-8")
).hexdigest()
if _hash_fit != _man_fit["hash_ajuste"]:
    raise RuntimeError("A configuração assinada do treinamento não confere.")

# Encadeia os IDs do MESMO manifesto, não de variáveis de outro treinamento.
MS03_CFG.update({
    "id_experimento": _exp,
    "id_ajuste": _fit_id,
    "id_02c": _id_02c,
})
MS03_CFG["patch_manifestos_por_id"] = {
    "versao": "v1",
    "tabela_02c": _tab_02c,
    "versao_02c_lida": _v_man_02c,
    "tabela_ajuste": _tab_fit,
    "versao_ajuste_referenciada_02c": _cfg_02c["fontes_delta"][_tab_fit],
    "versao_ajuste_lida": _v_man_fit,
    "hash_config_02c": _man_02c["hash_02c"],
    "hash_config_ajuste": _man_fit["hash_ajuste"],
}
_MS03_MANIFESTOS_FIXOS = {
    _tab_02c: (_df_man_02c, _v_man_02c),
    _tab_fit: (_df_man_fit, _v_man_fit),
}

# Mantém o leitor original mesmo quando este patch é executado novamente.
_MS03_LEITOR_ESTRITO = getattr(
    ms03_ler, "_leitor_estrito_original", ms03_ler
)
# A execução anterior parou na seção 3. A seção 3 será reiniciada integralmente.
MS03_FONTES.clear()
MS03_CHECKS.clear()


def ms03_ler(tabela, versao=None):
    if tabela in _MS03_MANIFESTOS_FIXOS:
        df, versao_lida = _MS03_MANIFESTOS_FIXOS[tabela]
        MS03_FONTES[tabela] = versao_lida
        print(
            f"MANIFESTO por ID validado: {tabela} | "
            f"versão solicitada={versao} | versão lida={versao_lida}",
            flush=True,
        )
        return df

    print(
        f"FONTE HISTÓRICA: {tabela} | "
        f"versão={versao if versao is not None else 'atual a fixar'}",
        flush=True,
    )
    # Nenhum fallback para dados atuais nas outras fontes.
    df = _MS03_LEITOR_ESTRITO(tabela, versao)
    df.limit(1).collect()
    return df


ms03_ler._leitor_estrito_original = _MS03_LEITOR_ESTRITO
print("Patch aplicado. IDs utilizados pela Parte 03:")
for _chave in ("id_experimento", "id_ajuste", "id_02c"):
    print(f"  {_chave}: {MS03_CFG[_chave]}")
print("Execute agora a seção 3 inteira e, se concluir, continue a partir da 4.")
