# PROTOCOL: prognozy Dst i pokrycie przedziałów niepewności w burzach 25. cyklu Słońca

**Wersja:** 0.4 (szkic, niezamrożony), 2026-11-24

**Status:** szkic. Zamrożenie następuje po zamknięciu punktów otwartych z §15, przed pierwszym treningiem. Od zamrożenia każda zmiana wymaga wpisu w §13 (data, powód, wpływ).

**Zakres:** dane historyczne (NASA OMNI, Kyoto WDC). Wnioski nie dotyczą prognozy operacyjnej na danych z L1 w czasie rzeczywistym (§12).

---

## 1. Pytanie badawcze

> Czy prognozy indeksu Dst z modeli ML wnoszą coś ponad persystencję w czasie silnych burz 25. cyklu Słońca i czy metody przedziałów niepewności, które starają się utrzymać pokrycie w burzach, zachowują je po przejściu na cykl, którego nie widziały?

Nie pytamy, który model ML jest najlepszy, ani o prognozy z wyprzedzeniem dłuższym niż 6 h. Wszystkie modele są budowane w tym projekcie od zera; z pracami innych autorów porównujemy się dopiero w dyskusji, po ocenie końcowej (§12). Nie testujemy też, czy zwykła predykcja konformalna ma pokrycie warunkowe w burzach: z teorii wiadomo, że go nie gwarantuje, więc raportujemy ją wyłącznie jako punkt odniesienia.

## 2. Pojęcia

**Dst** to godzinowy indeks prądu pierścieniowego [nT]; im bardziej ujemny, tym silniejsza burza. Wartość godzinowa opisuje średnią z przedziału [t_start, t_start + 1 h) (§5.2).

**Modele punktowe** (jeden zestaw danych, jedna procedura oceny):

| ID | Model | Rola |
|---|---|---|
| P | persystencja: `Dst(t+h) = Dst(t)` | punkt odniesienia w H1 |
| E | model empiryczny O'Briena i McPherrona (AK2) | drugi punkt odniesienia, eksploracyjnie; dopasowany dla Dst > −150 nT, więc w silnych burzach ekstrapoluje |
| X | XGBoost na cechach fizycznych zdefiniowanych w §7.1 | model ML, konfirmacyjnie |
| L | LSTM | model ML, konfirmacyjnie |

**Metody przedziałów** (nakładane na każdy model punktowy, poziom nominalny 90%):

| ID | Metoda | Rola |
|---|---|---|
| C-split | zwykła (split) predykcja konformalna | punkt odniesienia, opisowo |
| C-Mond | Mondrian: kwantyle osobno dla klas Dst znanego w chwili wydania prognozy | H2 |
| C-DtACI | adaptacyjna predykcja konformalna z dynamicznym doborem kroku (Gibbs i Candès 2024) | H3 |
| C-ACI | adaptacyjna predykcja konformalna ze stałym krokiem γ (Gibbs i Candès 2021) | czułość (S1) |

**Okno burzy** i **kategorie** zdefiniowano w §6. **Pokrycie** to udział godzin, w których obserwowany Dst leży w przedziale. **Horyzont** h ∈ {1 h, 6 h}; oba są równorzędne.

## 3. Co jest rozróżnialne

Przed interpretacją wyników przyjmujemy, że nie da się rozdzielić:

1. **Zmiany fizycznej i zmiany wersji etykiet.** Trening i kalibracja mają Dst ostateczny, test głównie wstępny (§5.1). Wersje liczą linię bazową i zmienność Sq inaczej, więc część przesunięcia rozkładu w teście może pochodzić z wersji indeksu, a nie z fizyki.
2. **Wartości modelu i jakości danych wejściowych.** OMNI to dane przetworzone po fakcie, z przesunięciem do czoła fali uderzeniowej. Wynik jest górną granicą dla prognozy operacyjnej.
3. **Pokrycia w burzy i warunkowania na wyniku.** Okna burz wyznacza obserwowany Dst (§6), więc pokrycie „w burzy” jest warunkowe względem wyniku. Okna obejmują wszystkie fazy burzy, co osłabia ten efekt, ale go nie usuwa.
4. **Przesunięcia między cyklami i trudności samych burz.** Dlatego pokrycie metod C-Mond i C-DtACI w burzach raportujemy także na 24. cyklu, poza próbą kalibracyjną (§7.3, M9). Spadek względem 24. cyklu jest miarą przesunięcia; samo pokrycie poniżej nominalnego już nie.

## 4. Hipotezy

Hipotezy rozstrzygane tylko regułami z §10. Każdy wynik, także zerowy, jest raportowany. Każdą hipotezę testujemy dla dwóch modeli ML (X, L) i dwóch horyzontów (1 h, 6 h), czyli 4 testy na hipotezę.

- **H1 (przewaga ML maleje z siłą burzy).** Skill score modelu ML względem persystencji jest niższy w oknach burz silnych niż w godzinach spokojnych.
- **H2 (pokrycie warunkowe nie przenosi się na nowy cykl).** Przedziały C-Mond skalibrowane na 24. cyklu mają w oknach burz silnych 25. cyklu pokrycie poniżej nominalnego 90%.
- **H3 (adaptacja odzyskuje pokrycie).** Przedziały C-DtACI mają w oknach burz silnych wyższe pokrycie niż C-Mond i nie niższe niż 85%.

Przewidywania kierunkowe, **eksploracyjne** (bez korekty wielokrotnych porównań, raportowane jako obserwacje):

- Przewaga ML nad persystencją jest większa przy h = 6 h niż przy h = 1 h, bo persystencja szybko traci wartość z horyzontem.
- C-Mond traci pokrycie głównie na początku fazy głównej, gdy Dst w chwili wydania prognozy jest jeszcze w klasie spokojnej.
- C-DtACI płaci za pokrycie szerokością przedziałów głównie w fazie głównej, a mniej w fazie odnowy.
- Model E zaniża minimum silnych burz mniej niż modele ML, mimo pracy poza zakresem dopasowania.

## 5. Dane

### 5.1 Źródła i wersje

| Zmienna | Źródło | Rozdzielczość | Uwagi |
|---|---|---|---|
| \|B\|, By i Bz w GSM, gęstość, temperatura, V, ciśnienie dynamiczne, pole E | [NASA OMNI2](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/), pliki roczne | 1 h | przesunięte do czoła fali uderzeniowej; wartości braku według `omni2.text` |
| Dst | [Kyoto WDC](https://wdc.kugi.kyoto-u.ac.jp/dstdir/) przez HAPI, dataset `hour_dst` | 1 h | wersja zapisana dla każdej godziny |
| Dst w OMNI | OMNI2, kolumna 41 | 1 h | tylko do sprawdzenia wyrównania godzin (§5.2) |

**Wersje Dst** według strony Kyoto (stan na 2026-11-24): ostateczny 1957–2020, wstępny 2021/01–2025/06, później tylko czas rzeczywisty. Skrypt odczytuje te zakresy ze strony w chwili pobrania i zapisuje jej kopię.

### 5.2 Konwencja czasu

- `t_start` to początek godziny, której średnią opisuje rekord.
- Wartość za przedział [t − 1 h, t) jest znana w chwili t. Prognoza wydana w chwili t używa tylko takich wartości.
- Cel prognozy z horyzontem h to Dst za przedział [t + h − 1 h, t + h).
- Kyoto i OMNI mogą różnie znakować godziny. Przesunięcie ustalamy na danych 1995–2014, porównując Dst z Kyoto z Dst w OMNI. Akceptujemy przesunięcie o zgodności co najmniej 99%; jeśli żadne go nie osiąga, prace stają do wyjaśnienia przyczyny. Przy h = 1 h przesunięcie o godzinę oznaczałoby wyciek celu do wejścia.

### 5.3 Podziały (skończone, zamrożone, na dysku)

Dane pobiera raz `scripts/download_data.py`. Data pobrania, zakresy wersji Dst, przesunięcie godzin i sumy SHA-256 wszystkich plików są zapisane w `data/MANIFEST.json`.

| Zbiór | Okres (UTC) | Cykl (w przybliżeniu) | Wersja Dst | Plik | Użycie |
|---|---|---|---|---|---|
| train | 1995-01-01 – 2009-12-31 | 23 | ostateczna | `data/processed/train.parquet` | trening i strojenie (§7.2) |
| calib | 2010-01-01 – 2019-12-31 | 24 | ostateczna | `data/processed/calib.parquet` | kwantyle konformalne, rozgrzanie metod adaptacyjnych, odniesienie M9, analiza mocy; bez wyboru modeli |
| test | 2020-01-01 – 2025-06-30 | 25 | ostateczna (2020), wstępna (2021 – VI 2025) | `data/sealed/test.parquet` | jednorazowa ocena końcowa |

Granice zbiorów są datami kalendarzowymi, a nie dokładnymi granicami cykli: train zawiera też koniec 22. cyklu (1995–1996), a 24. cykl zaczął się pod koniec 2008 roku. Każdy etap obejmuje jednak głównie jeden cykl. Kalibracja na całym 24. cyklu, a nie na latach 2015–2019, ma zapewnić burze w różnych fazach cyklu.

**Zbiór testowy** jest dotykany **tylko** w ocenie końcowej. Nie służy do strojenia, wyboru modelu, debugowania ani wykresów. Wyjątek, liczony wyłącznie z Dst przed uruchomieniem jakiegokolwiek modelu: liczba okien burz w teście (§6), potrzebna też do analizy mocy (§10.4).

## 6. Burze

**Definicja** (z góry, wyłącznie z szeregu Dst):

1. Minimum burzy: godzina t, w której `Dst(t) ≤ −50 nT` i `Dst(t)` jest najmniejsze w przedziale [t − 24 h, t + 24 h]. Przy remisie bierzemy najwcześniejszą godzinę.
2. Okno burzy: [t_min − 24 h, t_min + 48 h]. Nakładające się okna łączymy; kategorię połączonego okna wyznacza najniższe minimum.
3. Okno należy do zbioru, w którym leży jego minimum, i jest przycinane do granic tego zbioru.
4. Godzina prognozy należy do okna, jeśli jej **godzina celu** leży w oknie.
5. **Bloki spokojne:** godziny celu poza wszystkimi oknami dzielimy na kolejne bloki kalendarzowe 27-dniowe (jeden obrót Słońca), zaczynając od początku zbioru. Blok z mniej niż 72 godzinami spokojnymi dołączamy do poprzedniego.
6. **Fazy burzy:** faza główna to godziny celu od początku okna do najniższego minimum okna włącznie; faza odnowy to godziny celu po tym minimum. W oknach połączonych z kilku burz fazy liczymy względem najniższego minimum.

**Kategorie:**

| Kategoria | Minimum Dst [nT] | Rola |
|---|---|---|
| spokojne | bloki z pkt 5 | odniesienie w H1 |
| umiarkowane | −50 do −100 | raportowane zawsze |
| silne | ≤ −100 | H1–H3 |
| ekstremalne | ≤ −250 (podzbiór silnych) | wyłącznie opisowo |

**Minimalna liczba burz.** Jeśli w teście jest mniej niż **10** okien burz silnych, H1–H3 raportujemy opisowo, bez rozstrzygnięcia. Liczbę ustalamy przed uruchomieniem modeli i wpisujemy do §15.

## 7. Modele i trening

- **Wejście wszystkich modeli:** tylko wartości znane w chwili t (§5.2). Bieżący Dst jest na wejściu, bo jest dostępny operacyjnie (indeks czasu rzeczywistego). Dzięki temu persystencja jest trudnym punktem odniesienia.
- **P:** `Dst(t+h) = Dst(t)`.
- **E (AK2):** `dDst*/dt = Q − Dst*/τ`, `Q = −4,4·(VBs − 0,5)` nT/h dla VBs > 0,5 mV/m i 0 w przeciwnym razie, `τ = 2,4·exp(9,74 / (4,69 + VBs))` h, `Dst* = Dst − 7,26·√Pdyn + 11` (Dst w nT, Pdyn w nPa). VBs w mV/m: `V [km/s] · Bs [nT] · 10⁻³`, gdzie Bs = −Bz dla Bz < 0 i 0 w przeciwnym razie. Całkowanie krokiem 1 h od `Dst*(t)` przez h kroków. Wiatr słoneczny jest trzymany na wartości z chwili t, bo przyszłej nie znamy. Bez dopasowania stałych.
- **X:** XGBoost (regresja, strata kwadratowa) na zestawie cech z §7.1, jeden model na horyzont. Braki w cechach obsługuje natywnie. Każda cecha musi przejść test przyczynowości (§14). Stałe: `subsample = 0,8`, `colsample_bytree = 0,8`, żeby seed wpływał na model.
- **L:** okno 24 h zmiennych V, Bz i By w GSM, \|B\|, gęstość, ciśnienie dynamiczne, VBs i Dst; standaryzacja ze statystyk train; jedna sieć na horyzont; wyjście to jedna liczba.
- **Wszystkie dopasowywane przekształcenia** (standaryzacja, ewentualne imputery) uczą się wyłącznie na train.
- **Braki danych:** ostatnia znana wartość wypełnia lukę najwyżej 3 h wstecz. Dłuższa luka oznacza brak prognozy w tej godzinie. Ocena główna odbywa się tylko na wspólnym zbiorze godzin, w których **wszystkie** cztery modele mają prognozę i znany jest cel. Ponieważ luki w danych wiatru słonecznego mogą skupiać się w dużych burzach, raportujemy udział wyłączonych godzin w każdej kategorii (M10) i wykonujemy S5 (§9).
- **Seedy:** 5 seedów (42–46) dla X i L. Seed steruje inicjalizacją, kolejnością danych i podpróbkowaniem w XGBoost. Prognoza punktowa to średnia z 5 modeli; rozrzut między seedami raportujemy osobno.
- **Zespół jako jednostka:** hipotezy dotyczą prognozy uśrednionej po seedach, bo taki model wdrożono by w praktyce.

### 7.1 Cechy modelu X (ustalone z góry)

Wszystkie cechy liczone wyłącznie z wartości znanych w chwili t (§5.2); „opóźnienie k” oznacza wartość za przedział [t − k − 1 h, t − k).

| Grupa | Cechy | Uzasadnienie |
|---|---|---|
| Stan prądu pierścieniowego | Dst z opóźnieniem 0, 1, 2, 3 i 6 h; `Dst(t) − Dst(t − 1 h)`; `Dst(t) − Dst(t − 3 h)` | bieżący stan i tempo zmian (faza główna czy odnowa) |
| Wiatr słoneczny | V, Bz i By w GSM, \|B\|, gęstość, Pdyn z opóźnieniem 0, 1, 2 i 3 h | bezpośrednie wymuszanie |
| Sprzężenie | VBs [mV/m]; funkcja sprzężenia Newella `V^(4/3)·B_T^(2/3)·sin^(8/3)(θ/2)`, gdzie `B_T = √(By² + Bz²)`, `θ = atan2(By, Bz)`; `√Pdyn` | wstrzykiwanie energii (Burton, O'Brien i McPherron) i poprawka ciśnieniowa w Dst* |
| Historia wymuszania | średnie kroczące z 3 i 6 h dla VBs, Bz, Pdyn i funkcji Newella; sumy VBs z 6 i 12 h | energia zgromadzona przed chwilą t |

Razem 44 cechy: 7 + 24 + 3 + 10. Lista jest zamrożona razem z protokołem; nie dodajemy ani nie usuwamy cech na podstawie wyników, także walidacyjnych. Ważność cech (SHAP) raportujemy wyłącznie opisowo.

### 7.2 Strojenie (równy budżet)

Strojenie wyłącznie w obrębie train: dopasowanie na latach 1995–2005, walidacja na 2006–2009, kryterium MSE na walidacji, osobny seed 100. Każdy model ML dostaje 12 konfiguracji:

| Model | Siatka |
|---|---|
| X | max_depth ∈ {4, 6, 8} × learning_rate ∈ {0,03, 0,1} × min_child_weight ∈ {1, 10}; liczba drzew przez wczesne zatrzymanie (cierpliwość 50, maks. 5 000) |
| L | rozmiar ukryty ∈ {32, 64, 128} × lr ∈ {10⁻³, 3·10⁻³} × warstwy ∈ {1, 2}; Adam, batch 256, wczesne zatrzymanie (cierpliwość 5, maks. 100 epok) |

Model końcowy trenujemy na całym train (1995–2009) z wybraną konfiguracją i liczbą drzew lub epok z najlepszego przebiegu strojenia. Jeśli optimum leży na krawędzi siatki, siatki nie rozszerzamy i raportujemy to z wynikami. Czy konfiguracja jest wspólna dla obu horyzontów, czy wybierana osobno: decyzja w §15.

### 7.3 Przedziały

Miara niezgodności dla wszystkich metod: `|Dst − prognoza|`. Reszty kalibracyjne pochodzą z calib (modele trenowane wyłącznie na train).

- **C-split:** kwantyl rzędu ⌈(n + 1)·0,9⌉ / n z reszt na calib. Szerokość jest stała dla danego modelu i horyzontu.
- **C-Mond:** jak C-split, ale kwantyle osobno dla klas `Dst(t)` w chwili wydania prognozy: > −30, −30 do −50, −50 do −100, ≤ −100 nT. Klasa zależy od chwili wydania, nie od wyniku. Klasa z mniej niż 200 godzinami w calib jest łączona z sąsiednią, bliższą zeru; liczności klas raportujemy.
- **C-DtACI:** algorytm 1 z Gibbs i Candès (2024): k = 8 ekspertów ACI z krokami γ ∈ {0,001; 0,002; 0,004; 0,008; 0,016; 0,032; 0,064; 0,128}, łączonych wagami wykładniczymi; parametry η i σ według wzorów z pracy dla okna I = 72 h (długość okna burzy). Kwantyl rzędu 1 − α(t) z reszt kalibracyjnych.
- **C-ACI (S1):** jeden ekspert, `α(t+1) = α(t) + γ·(0,1 − err(t))`, γ ∈ {0,005; 0,05}.
- **Wspólne zasady metod adaptacyjnych:**
  - **Opóźnienie informacji zwrotnej:** prognoza wydana w chwili t daje `err` dopiero w t + h. Aktualizacja w chwili t używa tylko prognoz, których cel jest już znany.
  - Start: α = 0,1 (i równe wagi ekspertów) na 2010-01-01; przebieg przez cały calib i dalej przez test bez resetu.
  - Gdy α ≤ 0, przedział jest nieskończony i liczony jako pokryty; gdy α ≥ 1, pusty i liczony jako niepokryty. Raportujemy liczbę takich godzin, a szerokość podajemy jako medianę.
- **Odniesienie na 24. cyklu (M9):** C-Mond i C-DtACI liczone w dwóch połowach calib krzyżowo: kwantyle z lat 2010–2014 stosowane do 2015–2019 i odwrotnie. Daje to pokrycie w burzach silnych poza próbą kalibracyjną, ale w obrębie cyklu kalibracji.

## 8. Metryki

Wszystkie liczone na wspólnym zbiorze godzin (§7), osobno dla każdego modelu, horyzontu i kategorii z §6.

**Agregacja.** Statystyka główna to średnia po **jednostkach** (oknach burz, a w kategorii spokojnej po blokach), każda jednostka z równą wagą. Wersja ważona godzinami jest raportowana pomocniczo. Dla skill score agregujemy błędy, nie ilorazy:

`SS_kat = 1 − średnia_j MSE_model,j / średnia_j MSE_P,j`, gdzie j przebiega jednostki kategorii.

**Główne:**

| ID | Definicja |
|---|---|
| M1 | `SS_kat` jak wyżej |
| M2 | pokrycie: średnia po jednostkach z udziału godzin, w których cel leży w przedziale |

**Pomocnicze:**

| ID | Definicja |
|---|---|
| M1b | różnica `średnia_j MSE_P,j − średnia_j MSE_model,j` [nT²]; uzupełnia M1, bo w godzinach spokojnych MSE persystencji jest małe i iloraz bywa niestabilny |
| M3 | RMSE i MAE [nT] |
| M4 | mediana szerokości przedziału [nT] |
| M5 | interval score dla α = 0,1: `(u − l) + (2/α)·(l − y)·1[y < l] + (2/α)·(y − u)·1[y > u]` [nT] |
| M6 | SS względem modelu E: `1 − średnia_j MSE_model,j / średnia_j MSE_E,j` |
| M7 | kontrola „ukrytej persystencji”: przesunięcie k ∈ {0, …, h + 3} h, przy którym korelacja prognozy z obserwacją w oknie burzy jest najwyższa; mediana po oknach i udział okien z k ≥ h (por. Laperre i in. 2020) |
| M8 | dla pojedynczego okna: minimum prognozy minus minimum obserwacji [nT] oraz błąd momentu minimum [h] |
| M9 | pokrycie C-Mond i C-DtACI w burzach silnych 24. cyklu, krzyżowo (§7.3); różnica M2(test) − M9 |
| M10 | udział godzin wyłączonych z oceny przez braki danych, w każdej kategorii |
| M11 | krzywa dawka–odpowiedź: M1 i M2 liczone dla każdego okna osobno w funkcji minimum Dst okna; wykres punktowy i średnie w przedziałach minimum co 50 nT |
| M12 | przesunięcie prognoz punktowych: M3(test) / M3(calib) w każdej kategorii; calib jest dla modeli poza próbą treningową |
| M13 | krzywa pokrycia: M2 dla poziomów nominalnych 50, 60, 70, 80, 90 i 95% (C-split, C-Mond, C-DtACI), osobno dla godzin spokojnych i burz silnych |

M1–M5 raportujemy także osobno dla faz burzy (§6, pkt 6).

## 9. Eksperymenty

| ID | Opis | Rola |
|---|---|---|
| R0 | testy poprawności pipeline'u na latach 2006–2009 (walidacja ze strojenia), §9.1 | warunek wstępny; bez danych testowych |
| P0 | analiza mocy na 24. cyklu (§10.4) | warunek wstępny; bez danych testowych |
| E-main | P, E, X, L × h ∈ {1, 6} × {C-split, C-Mond, C-DtACI} na test | konfirmacyjny (H1, H2, H3) |
| S1 | C-ACI z γ ∈ {0,005; 0,05} | czułość na sposób adaptacji |
| S2 | statystyki ważone godzinami zamiast jednostkami | czułość na agregację |
| S3 | okna burz [t_min − 12 h, t_min + 24 h] zamiast §6 | czułość na definicję okna |
| S4 | M6 i analog H1 względem modelu E | czy wniosek zależy od wyboru punktu odniesienia |
| S5 | P i X na wszystkich godzinach, w których same mają prognozę (XGBoost obsługuje braki) | czułość na wyłączenia z powodu braków danych |
| S6 | „zostaw jedną burzę”: każda statystyka konfirmacyjna liczona bez każdego okna burzy silnej po kolei | stabilność rozstrzygnięć |
| X1 | burze z 10–11 maja i 10–11 października 2024: M2, M4 i M8 | eksploracyjny, wyłącznie opisowo |
| X2 | modele zamrożone po E-main, ocenione na Dst w czasie rzeczywistym, 2025-07-01 – 2025-12-31 | eksploracyjny; bliżej warunków operacyjnych |
| X3 | odpowiedniki H1–H3 dla kategorii ekstremalnej | wyłącznie opisowo |
| X4 | jedno powtórzenie treningu L (seed 42, oba horyzonty) z tą samą konfiguracją | skala szumu między przebiegami z niedeterminizmu GPU; wyłącznie opisowo |
| A1 | X i L bez żadnych wejść z Dst (X bez grupy „stan prądu pierścieniowego”, L bez kanału Dst) | ablacja: czy przewaga nad persystencją pochodzi z wiatru słonecznego, czy z przedłużania bieżącego Dst |
| A2 | X bez grupy „sprzężenie” | ablacja: wkład funkcji sprzężenia |
| A3 | X bez grupy „historia wymuszania” | ablacja: wkład energii zgromadzonej przed chwilą t |
| A4 | X wyłącznie na grupie „stan prądu pierścieniowego” | ablacja: model czysto autoregresyjny, uzupełnienie A1 |

**Ablacje A1–A4** używają seedów 42–46, konfiguracji wybranych w strojeniu modelu pełnego (bez ponownego strojenia), tych samych podziałów danych i wszystkich trzech metod przedziałów. Mierzą więc czułość przy ustalonych hiperparametrach. Dla każdej ablacji raportujemy M1, M2 i M7 w każdej kategorii oraz różnicę względem odpowiednika z E-main (ten sam model, horyzont i seedy), z 95% przedziałem bootstrapowym (§10.1).

### 9.1 Testy poprawności (R0)

Wykonywane na danych walidacyjnych ze strojenia (2006–2009), przed jakimkolwiek kontaktem z calib i test. Niepowodzenie któregokolwiek oznacza błąd do znalezienia, a nie wynik.

1. **Kontrola pozytywna:** X i L muszą mieć MSE niższe niż P dla obu horyzontów. Model, który nie bije persystencji na danych z tego samego cyklu, jest najpewniej źle podłączony.
2. **Kontrola wycieku przez cel:** X trenowany z celem dodanym jako cecha musi osiągnąć MAE bliskie zeru (< 1 nT). Potwierdza, że pipeline potrafi wykryć wyciek, gdy ten istnieje.
3. **Kontrola negatywna:** X trenowany na celu losowo przetasowanym w obrębie train nie może być lepszy od prognozy stałej (średniej z train) o więcej niż 1% MSE.
4. **Model E:** korelacja prognozy z obserwacją > 0,8 dla h = 1 h. Sprawdza jednostki (VBs w mV/m, Pdyn w nPa) i całkowanie.

**Analiza S1–S5, A1–A4 i X1–X4** jest opisowa, bez testów hipotez: estymata punktowa i 95% przedział bootstrapowy (§10).

**Kolejność przebiegów** (najpierw najważniejsze dla interpretacji): R0, P0, strojenie, E-main, S6, A1, A4, S5, S2, S4, A2, A3, S1, S3, X4, X1, X3, X2.

## 10. Statystyka i reguły decyzyjne

### 10.1 Jednostki i bootstrap

**Jednostka niezależna** to okno burzy lub blok spokojny (§6), nie godzina, bo godziny jednej burzy są silnie skorelowane.

**Bootstrap.** W każdej replikacji losujemy ze zwracaniem okna burz osobno w każdej kategorii oraz bloki spokojne. Jeden zestaw losowań jest wspólny dla wszystkich modeli, horyzontów i metod przedziałów, bo porównania są sparowane. 10 000 replikacji, seed 2026. Przedziały 95% percentylowe. Jednostronna wartość p to udział replikacji, w których statystyka leży na granicy hipotezy zerowej lub po jej stronie (granica: 0 dla H1 i H3, 0,90 dla H2).

**Korekta.** Metoda Holma osobno w każdej rodzinie: 4 testy na hipotezę (X, L × 1 h, 6 h), α = 0,05.

### 10.2 Reguły

**H1.** Dla modelu v i horyzontu h: `Δ = M1(spokojne) − M1(silne)`. H1 uznajemy za popartą dla (v, h), gdy jednocześnie:

- `Δ ≥ 0,05` (próg praktyczny),
- p skorygowane metodą Holma < 0,05.

**H2.** Dla C-Mond na modelu v i horyzoncie h: `cov = M2(silne)`. H2 uznajemy za popartą, gdy jednocześnie:

- `cov ≤ 0,85` (co najmniej 5 punktów procentowych poniżej nominalnego),
- p Holma < 0,05 dla `cov < 0,90`.

M9 i różnicę M2(test) − M9 raportujemy obok; nie wchodzą do reguły, ale bez nich wyniku H2 nie interpretujemy jako skutku przesunięcia między cyklami (§3, pkt 4).

**H3.** `D = M2(silne; C-DtACI) − M2(silne; C-Mond)`. H3 uznajemy za popartą, gdy jednocześnie:

- `D ≥ 0,03`,
- p Holma < 0,05 dla `D > 0`,
- `M2(silne; C-DtACI) ≥ 0,85`.

Raportujemy także stosunek median szerokości C-DtACI do C-Mond; nie wchodzi do reguły.

### 10.3 Odporność

- **Mało skupień.** Bootstrap po kilkunastu skupieniach bywa zbyt optymistyczny. Dla H3, gdzie porównanie jest sparowane w obrębie okna, dodatkowo raportujemy dokładny test permutacyjny ze zmianą znaku różnic `D_j` po oknach. Rozstrzyga reguła z §10.2; rozbieżność z testem permutacyjnym zaznaczamy w raporcie.
- **Wpływ pojedynczej burzy (S6).** Jeśli usunięcie jednego okna zmienia rozstrzygnięcie którejkolwiek hipotezy, rozstrzygnięcie pozostaje formalnie ważne, ale raportujemy je jako **niestabilne**, z nazwą tej burzy.
- **Braki danych.** Jeśli M10 w kategorii silnej przekracza 10%, wyniki H1–H3 raportujemy z zastrzeżeniem i obok wyników S5.

### 10.4 Analiza mocy (P0, przed zamrożeniem)

Na 24. cyklu, z modelami trenowanymi na train: losujemy ze zwracaniem tyle okien burz silnych, ile jest w teście (liczba z §6), i dla efektów równych progom z §10.2 szacujemy prawdopodobieństwo poparcia każdej hipotezy (1 000 symulacji). Raportujemy minimalny wykrywalny efekt przy mocy 80%. Jeśli progi są niewykrywalne, zmieniamy je **przed** zamrożeniem i zapisujemy powód w §15. Po zamrożeniu progów się nie zmienia.

**Dodatkowo (zawsze):** M1–M13 dla wszystkich modeli, horyzontów i kategorii, z 95% przedziałami bootstrapowymi; rozrzut M1 i M3 między seedami; C-split jako punkt odniesienia.

**Wynik zerowy jest wynikiem.** Jeśli hipoteza nie jest poparta, raport tak to stwierdza i pokazuje rozkłady bootstrapowe. Nie wolno zmieniać punktu końcowego, progu, kategorii ani definicji burzy po obejrzeniu wyników.

## 11. Zagrożenia dla trafności

| Zagrożenie | Postępowanie |
|---|---|
| Różne wersje Dst w treningu i teście | Nazwane w §3; X2 na danych czasu rzeczywistego jako kontrola; ponowna ocena po publikacji Dst ostatecznego (poza zakresem) |
| Dane OMNI przetworzone po fakcie | Zakres wniosków (§12) |
| Wyciek przez przesunięcie godzin | Sprawdzenie wyrównania na latach 1995–2014 (§5.2) i test przyczynowości cech (§14) |
| Cechy nieprzyczynowe lub przekształcenia dopasowane na wszystkich danych | Test przyczynowości każdej cechy (§14); przekształcenia uczone tylko na train; kontrole R0 (§9.1) |
| Dobór cech pod wynik | Lista cech zamrożona z protokołem (§7.1); brak selekcji cech |
| Hipotezy sformułowane po lekturze prac z wynikami na 25. cyklu (m.in. Yu i in. 2026: łączne pokrycie poniżej nominalnego) | Ujawnione w pracy; ślepa jest ocena naszych modeli, nie wiedza o literaturze |
| Autorka zna przebieg burz z 2024 roku | Ślepa jest ocena modeli, nie wiedza o zdarzeniach; definicje burz i reguły ustalone przed oceną |
| Okna burz wyznaczone z obserwowanego Dst | §3; S3 sprawdza czułość na definicję okna |
| Mało silnych burz w teście; bootstrap przy małej liczbie skupień | Próg 10 burz (§6); analiza mocy P0; test permutacyjny dla H3; kategoria ekstremalna tylko opisowo |
| Jedna burza (np. z maja 2024) dominuje statystyki | Agregacja z równą wagą okien (§8); S6 i oznaczenie rozstrzygnięć niestabilnych |
| Braki danych skupione w burzach | M10, próg 10% i S5 |
| Niestabilny skill score w godzinach spokojnych | M1b jako uzupełnienie; agregacja błędów zamiast ilorazów |
| Model E poza zakresem dopasowania w silnych burzach | Tylko eksploracyjnie; nazwane w §2 |
| Pokrycie poniżej nominalnego może wynikać z trudności burz, a nie z przesunięcia cykli | M9 jako odniesienie na 24. cyklu (§3, pkt 4) |
| Reszty godzinowe nie są wymienialne (autokorelacja) | Gwarancja C-split i C-Mond nie jest dokładna już na calib; raportujemy pokrycie na calib jako kontrolę |
| Hiperparametry strojone tylko na 23. cyklu | Ograniczenie; równy budżet strojenia dla X i L (§7.2) |
| Wybór kroku w metodzie adaptacyjnej | C-DtACI dobiera krok sam; siatka γ i okno I ustalone z góry; C-ACI ze stałym γ w S1 |
| Rewizje danych po pobraniu | Jednorazowe pobranie, sumy SHA-256 i data w `MANIFEST.json` |
| Przewaga ML wynika z przedłużania bieżącego Dst, a nie z fizyki wymuszania | A1 i A4; kontrola M7 |
| Błąd w implementacji statystyki | Kod z §10 sprawdzony na danych syntetycznych o znanym efekcie przed oceną końcową (§14) |
| Niedeterminizm treningu na GPU | X4 jako miara szumu; prognoza jako średnia z 5 seedów |

## 12. Zakres wniosków

Wnioski dotyczą: prognoz Dst z horyzontem 1 h i 6 h, danych OMNI i Kyoto z §5, modeli P, E, X i L zbudowanych w tym projekcie, wytrenowanych na latach 1995–2009 i ocenionych na latach 2020 – czerwiec 2025, metod przedziałów z §2 oraz definicji burz z §6. **Nie dotyczą** prognozy operacyjnej na danych z L1 w czasie rzeczywistym, innych indeksów (SYM-H, Kp), dłuższych horyzontów ani modeli niewymienionych w §2. X2 daje tylko wskazówkę co do warunków operacyjnych. Wnioski o burzach ekstremalnych (≤ −250 nT) są opisowe.

**Porównanie z literaturą** (m.in. Sandu i Beșliu-Ionescu 2026, Yu i in. 2026) jest wyłącznie opisowe i odbywa się w dyskusji, po ocenie końcowej. Różnice w okresach testowych, wersjach Dst i metrykach sprawiają, że liczby nie są bezpośrednio porównywalne; nie wpływa ono na żadne rozstrzygnięcie.

## 13. Dziennik odstępstw (zmiany po zamrożeniu)

| Data | Co zmieniono | Powód | Wpływ na analizę |
|---|---|---|---|
| | | | |

Zmiany przed zamrożeniem są widoczne w historii Git. Każde użycie danych testowych przed oceną końcową, także w testach kodu, jest ujawniane w tej tabeli, nawet gdy żaden model nie był na nich oceniany.

## 14. Reprodukowalność

- Dane: `scripts/download_data.py`, uruchamiany raz; `data/MANIFEST.json` z datą pobrania, adresami, zakresami wersji Dst, przesunięciem godzin, wersjami pakietów i sumami SHA-256.
- Wszystkie parametry modeli, przedziałów i bootstrapu w `configs/*.yaml`. Raport zawiera hash konfiguracji i commita.
- Testy automatyczne: parser OMNI i wartości braku, wyrównanie godzin, rozłączność podziałów, definicja okien burz na danych syntetycznych, przyczynowość każdej cechy (zmiana danych po chwili t nie zmienia cechy w chwili t), opóźnienie informacji zwrotnej w C-ACI i C-DtACI, zgodność implementacji C-DtACI z przykładem z pracy Gibbsa i Candèsa.
- Skrypt oceny końcowej odmawia działania, jeśli suma SHA-256 pliku `data/sealed/test.parquet` nie zgadza się z `MANIFEST.json` albo commit nie ma tagu `protocol-v1.0`.
- Implementacja bootstrapu, korekty Holma i reguł z §10 jest sprawdzona na danych syntetycznych o znanym efekcie (z efektem i bez) przed oceną końcową.
- Każdy przebieg zapisuje na starcie commit i flagę zmian niezacommitowanych. Przebiegi E-main i ablacji uruchamiamy wyłącznie z czystego repozytorium.
- XGBoost na CPU jest powtarzalny przy tym samym seedzie; trening LSTM na GPU nie jest bitowo powtarzalny (X4 mierzy skalę różnic).
- `scripts/make_tables.py` odtwarza każdą tabelę i rysunek z zapisanych prognoz, bez ręcznych kroków.
- Zamrożony protokół, z kodem i wynikami R0 oraz P0, jest archiwizowany w Zenodo przed jakąkolwiek oceną na teście.

## 15. Decyzje przed zamrożeniem

**Rozstrzygnięte:**

- **Horyzonty:** 1 h i 6 h równorzędnie, z korektą Holma na 4 testy w rodzinie.
- **Podział:** głównie cyklami 23 / 24 / 25 (§5.3); test kończy się na ostatnim miesiącu Dst wstępnego (czerwiec 2025).
- **Poziom nominalny:** 90%, bo daje więcej przypadków poza przedziałem w burzach niż 95%, a więc większą moc.
- **Jednostka analizy:** okno burzy lub blok spokojny; agregacja z równą wagą jednostek.
- **Metody konfirmacyjne:** C-Mond (H2) i C-DtACI (H3). C-split nie jest testowany, bo nie obiecuje pokrycia warunkowego (zmiana z wersji 0.1).
- **Modele:** wszystkie budowane w projekcie od zera; cechy modelu X zamrożone w §7.1 (zmiana z wersji 0.2, w której model X opierał się na kodzie Sandu i Beșliu-Ionescu).
- **Stałe modelu E:** b = 7,26 nT·nPa^−1/2 i c = 11 nT, zgodnie z O'Brienem i McPherronem (2000).

**Otwarte (do zamknięcia przed zamrożeniem):**

- [ ] Wynik R0 (testy poprawności, §9.1)
- [ ] Przesunięcie godzin Kyoto względem OMNI i zgodność (z `MANIFEST.json`)
- [ ] Numery kolumn i wartości braku OMNI sprawdzone z `omni2.text`
- [ ] Wzory na η i σ w C-DtACI oraz siatka γ przepisane z Gibbs i Candès (2024)
- [ ] Liczności klas C-Mond na calib i ewentualne łączenie klas
- [ ] Siatki strojenia z §7.2 oraz czy konfiguracja jest wspólna dla obu horyzontów
- [ ] Liczba okien burz silnych w teście (§6)
- [ ] Wynik P0 i ostateczne progi praktyczne z §10.2 (obecnie Δ ≥ 0,05; cov ≤ 0,85; D ≥ 0,03)
- [ ] Szacunek czasu obliczeń po teście dymnym (strojenie, E-main, ablacje) na GTX 1660
