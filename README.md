# WAE VSI optimisation

Projekt dla zadania 25 z przedmiotu Wstęp do algorytmów ewolucyjnych 2026L.

Autorzy:
- Jarosław Bieniek, 316018
- Krzysztof Gryboś, 348556

## Cel projektu

Celem projektu jest optymalizacja dostarczonego symulatora układu VSI. Problem traktujemy jako czarnoskrzynkową minimalizację funkcji celu:

```text
x in [-4, 4]^4 -> J(x)
```

## Uruchamianie
Program uruchamiany jest za pomocą [uv](https://docs.astral.sh/uv/). Do działania potrzebuje uruchomionego serwera [vsi-simulator-server](https://codeberg.org/ewarchul/vsi-simulator-server).

```bash
uv run src/main.py --cases <wybrany przykład> --out <folder na wyniki> --address <adres serwera> --port <port serwera>
```

Na przykład
```bash
uv run src/main.py --cases cases/cases_demo.csv --out out --address 127.0.0.1 --port 8080
```
