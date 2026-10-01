### Exemplo intuitivo do efeito temporal

Suponha que, para um determinado estado atual \(i\), a distribuição histórica dos próximos destinos seja:

| Destino | \(P(J=j \mid I=i)\) |
|---|---:|
| PIX | 0.50 |
| Boleto | 0.30 |
| Extrato | 0.20 |

Ou seja, **sem considerar o tempo**:

```text
PIX       50%
Boleto    30%
Extrato   20%
```

Nesse momento, PIX parece claramente o próximo destino mais provável.

Agora suponha que **10 minutos já tenham se passado sem que uma nova ação aconteça**.

Para cada possível destino, temos uma função de sobrevivência:

\[
S_{ij}(t)=P(T>t\mid I=i,J=j)
\]

Suponha que, em \(t=10\) minutos, tenhamos:

| Destino | \(S_{ij}(10)\) |
|---|---:|
| PIX | 0.10 |
| Boleto | 0.60 |
| Extrato | 0.80 |

A interpretação é:

- apenas **10%** das transições que terminam em PIX costumam demorar mais de 10 minutos;
- **60%** das transições que terminam em Boleto costumam demorar mais de 10 minutos;
- **80%** das transições que terminam em Extrato costumam demorar mais de 10 minutos.

Ou seja, o fato de já terem passado 10 minutos fornece informação sobre qual destino continua sendo mais compatível com o comportamento observado.

Calculamos então, para cada destino:

\[
\text{peso}_j = p_{ij}S_{ij}(a)
\]

#### PIX

\[
0.50 \times 0.10 = 0.05
\]

#### Boleto

\[
0.30 \times 0.60 = 0.18
\]

#### Extrato

\[
0.20 \times 0.80 = 0.16
\]

Temos:

| Destino | Probabilidade base \(p_{ij}\) | Sobrevivência \(S_{ij}(10)\) | Peso |
|---|---:|---:|---:|
| PIX | 0.50 | 0.10 | 0.05 |
| Boleto | 0.30 | 0.60 | 0.18 |
| Extrato | 0.20 | 0.80 | 0.16 |

Antes de considerar o tempo:

\[
PIX > Boleto > Extrato
\]

Depois de observar que **10 minutos já se passaram**:

\[
Boleto > Extrato > PIX
\]

Esse é o principal efeito introduzido pelo componente Semi-Markov.

O modelo não considera apenas:

> **"Qual destino costuma acontecer depois deste estado?"**

Ele também considera:

> **"Dado que já passou esse tempo sem uma nova transição, quais destinos continuam sendo mais compatíveis com esse tempo de espera?"**

Assim, um destino que inicialmente possuía alta probabilidade pode perder importância caso normalmente aconteça rapidamente e o cliente já esteja há muito tempo sem realizar uma nova ação.

Da mesma forma, destinos que historicamente demoram mais para acontecer podem ganhar importância conforme o tempo transcorrido aumenta.

Após esse cálculo, os pesos ainda precisam ser normalizados:

\[
q_j(a)=
\frac{
p_{ij}S_{ij}(a)
}{
\sum_k p_{ik}S_{ik}(a)
}
\]

No exemplo:

\[
0.05 + 0.18 + 0.16 = 0.39
\]

Portanto:

\[
q_{PIX}(10)
=
\frac{0.05}{0.39}
\approx 0.128
\]

\[
q_{Boleto}(10)
=
\frac{0.18}{0.39}
\approx 0.462
\]

\[
q_{Extrato}(10)
=
\frac{0.16}{0.39}
\approx 0.410
\]

A distribuição final passa a ser aproximadamente:

| Destino | Probabilidade após considerar o tempo |
|---|---:|
| PIX | 12.8% |
| Boleto | 46.2% |
| Extrato | 41.0% |

Ou seja:

```text
Antes do componente temporal:

PIX       50%
Boleto    30%
Extrato   20%


Após 10 minutos sem nova ação:

PIX       12.8%
Boleto    46.2%
Extrato   41.0%
```

A ideia central pode ser resumida como:

> **O Semi-Markov parte da probabilidade histórica de cada destino e repondera essa probabilidade de acordo com quão compatível cada destino é com o tempo que já passou.**