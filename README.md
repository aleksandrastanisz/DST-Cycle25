# DST-Cycle25

Czy prognozy indeksu Dst z modeli ML wnoszą coś ponad persystencję w czasie silnych burz 25. cyklu Słońca i czy metody przedziałów niepewności, które starają się utrzymać pokrycie w burzach, zachowują je na cyklu, którego nie widziały?

Modele są trenowane na latach 1995–2009 (głównie 23. cykl), przedziały konformalne kalibrowane na latach 2010–2019 (24. cykl), a ocena odbywa się raz, według zamrożonego protokołu, na 25. cyklu (styczeń 2020 – czerwiec 2025).

**Status:** protokół w wersji 0.4 (szkic, niezamrożony); żaden model nie był jeszcze trenowany ani oceniany na danych testowych.

## Protokół

Pełny plan badania, z hipotezami, regułami decyzyjnymi i dziennikiem odstępstw: [PROTOCOL_pl.md](PROTOCOL_pl.md) · [PROTOCOL_en.md](PROTOCOL_en.md).

W skrócie:

- **Modele:** persystencja, model empiryczny O'Briena i McPherrona, XGBoost na cechach fizycznych i LSTM; horyzonty 1 h i 6 h.
- **Niepewność:** predykcja konformalna zwykła, Mondrian i adaptacyjna (DtACI), poziom nominalny 90%.
- **Hipotezy:** H1 – przewaga ML nad persystencją maleje w silnych burzach; H2 – pokrycie Mondriana spada w burzach nowego cyklu; H3 – metoda adaptacyjna je odzyskuje.
- **Statystyka:** jednostką jest burza, nie godzina; bootstrap po burzach, korekta Holma, progi praktyczne ustalone z góry.

## Uruchomienie

```bash
pip install -r requirements.txt
python -m pytest              # testy logiki pobierania, bez sieci
python scripts/download_data.py
```

Skrypt pobiera dane raz i zapisuje `data/MANIFEST.json` z datą pobrania, zakresami wersji Dst z Kyoto i sumami SHA-256 wszystkich plików. Dane testowe trafiają do `data/sealed/` i nie są otwierane przed zamrożeniem protokołu.

## Źródła danych

- [NASA OMNI2, dane godzinowe](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/)
- [Kyoto WDC, indeks Dst](https://wdc.kugi.kyoto-u.ac.jp/dstdir/): wersja ostateczna do 2020 roku, wstępna od 2021 roku
