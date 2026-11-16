# Relatório de qualidade dos dados

## Perfil geral
- linhas: 10,675,502
- itens: 4,342
- skins: 521
- armas: 34
- inicio: 2013-08-14 00:00:00
- fim: 2026-06-15 00:00:00
- volume_total: 1291463671.0

## Colunas constantes (descartadas): ['source', 'currency', 'souvenir', 'weapon_category']

## Valores nulos / inválidos / duplicados
- preco_nulo: 0
- preco_nao_positivo: 0
- volume_nulo_ou_zero: 0
- duplicatas: 0

## Ruído (preço diário com pouca informação)
- linhas com source_point_count = 1: 10,556,472 (98.9%)
- linhas com volume <= 3: 911,919 (8.5%)

## Dados faltantes (lacunas nas séries diárias)
- itens com cobertura < 90% da própria vida: 309 de 4342 (7.1%)
- pior cobertura: 4.8%
- cobertura mediana: 100.0%
- itens rastreados: 85 em 2013 -> 4,341 em 2026

## Painel mensal bruto: 367,611 linhas (item x mês)
- meses faltantes dentro da vida dos itens: 427 (0.12%)
- outliers mensais removidos (Hampel, >3x da mediana local): 121
- preços imputados por interpolação (lacunas <= 2 meses): 354
- preços que continuam faltantes (lacunas longas, descartados): 194
- meses com menos de 7/30 dos dias negociados (baixa confiabilidade): 4,268

## Painel mensal limpo: 367,844 linhas, 4,342 itens, 521 skins