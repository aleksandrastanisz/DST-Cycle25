# PROTOCOL: prognozy Dst i pokrycie przedziałów niepewności w burzach 25. cyklu Słońca

**Wersja:** 1.1 (zamrożona), 2026-11-24. Poprzednia: 1.0, 2026-11-24, [10.5281/zenodo.24511114](https://doi.org/10.5281/zenodo.24511114). Różnice opisuje §13.

**Status:** zamrożony przed pierwszym treningiem i przed jakąkolwiek oceną na danych testowych. Każda kolejna zmiana wymaga wpisu w §13 (data, powód, wpływ).

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
2. **Wartości modelu i jakości danych wejściowych.** OMNI to dane przetworzone po fakcie, z przesunięciem do czoła fali uderzeniowej. Wynik jest górną granicą dla prognozy operacyjnej. Dotyczy to także Dst na wejściu modeli: w chwili t operacyjnie dostępny jest tylko Dst czasu rzeczywistego, z szumem i skokami linii bazowej, a modele dostają wersję ostateczną lub wstępną. Zawyża to trafność modeli i persystencji, najbardziej przy h = 1 h.
3. **Pokrycia w burzy i warunkowania na wyniku.** Okna burz wyznacza obserwowany Dst (§6), więc pokrycie „w burzy” jest warunkowe względem wyniku. Okna obejmują wszystkie fazy burzy, co osłabia ten efekt, ale go nie usuwa.
4. **Przesunięcia między cyklami i trudności samych burz.** Dlatego pokrycie metod C-Mond i C-DtACI w burzach raportujemy także na 24. cyklu, poza próbą kalibracyjną (§7.3, M9). Spadek względem 24. cyklu jest miarą przesunięcia i to on jest testowany w H2; samo pokrycie poniżej nominalnego już nie.
5. **Przesunięcia między cyklami i słabości cyklu kalibracji.** 24. cykl był najsłabszy od stu lat (26 okien burz silnych, żadnej superintensywnej), a 25. jest wyraźnie silniejszy. Wynik H2 dotyczy więc przejścia ze słabego cyklu kalibracji na silniejszy cykl testowy i nie rozstrzyga, czy taki sam spadek wystąpiłby między cyklami o podobnej sile.

## 4. Hipotezy

Hipotezy rozstrzygane tylko regułami z §10. Każdy wynik, także zerowy, jest raportowany. Każdą hipotezę testujemy dla dwóch modeli ML (X, L) i dwóch horyzontów (1 h, 6 h), czyli 4 testy na hipotezę.

- **H1 (przewaga ML maleje z siłą burzy).** Skill score modelu ML względem persystencji jest niższy w oknach burz silnych niż w godzinach spokojnych.
- **H2 (pokrycie warunkowe spada po przejściu na nowy cykl).** Pokrycie przedziałów C-Mond w oknach burz silnych jest niższe na 25. cyklu niż na 24. cyklu, mierzonym poza próbą kalibracyjną (M9).
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

**Wersje Dst** pochodzą z kodu wersji (`versionCode`), który HAPI Kyoto zwraca dla każdej godziny: 2x ostateczna, 1x wstępna, 0x czas rzeczywisty. Zbiór `hour_dst` podaje dla każdego okresu najwyższą dostępną wersję. Stan w chwili pobrania (2026-11-24): ostateczna do końca 2020, wstępna przez cały okres 2021–2025. Skrypt przerywa działanie, jeśli train lub calib zawiera wersję inną niż ostateczna; liczności wersji w każdym zbiorze są w `MANIFEST.json`.

### 5.2 Konwencja czasu

- `t_start` to początek godziny, której średnią opisuje rekord.
- Wartość za przedział [t − 1 h, t) jest znana w chwili t. Prognoza wydana w chwili t używa tylko takich wartości.
- Cel prognozy z horyzontem h to Dst za przedział [t + h − 1 h, t + h).
- **Zapis:** `Dst(t)` oznacza wartość za przedział [t − 1 h, t), czyli rekord o `t_start = t − 1 h`. Dotyczy to wszystkich wzorów z argumentem t (§2, §7, §7.1). W §6 i w kodzie czasy oznaczają `t_start` rekordów godzinowych.
- Kyoto i OMNI mogą różnie znakować godziny. Przesunięcie ustalamy na danych 1995–2014, porównując Dst z Kyoto z Dst w OMNI. Akceptujemy przesunięcie o zgodności co najmniej 99%; jeśli żadne go nie osiąga, prace stają do wyjaśnienia przyczyny. Przy h = 1 h przesunięcie o godzinę oznaczałoby wyciek celu do wejścia.

### 5.3 Podziały (skończone, zamrożone, na dysku)

Dane pobiera raz `scripts/download_data.py`. Data pobrania, liczności wersji Dst w każdym zbiorze, przesunięcie godzin i sumy SHA-256 wszystkich plików są zapisane w `data/MANIFEST.json`.

| Zbiór | Okres (UTC) | Cykl (w przybliżeniu) | Wersja Dst | Plik | Użycie |
|---|---|---|---|---|---|
| train | 1995-01-01 – 2009-12-31 | 23 | ostateczna | `data/processed/train.parquet` | trening i strojenie (§7.2) |
| calib | 2010-01-01 – 2019-12-31 | 24 | ostateczna | `data/processed/calib.parquet` | kwantyle konformalne, rozgrzanie metod adaptacyjnych, odniesienie M9, analiza mocy; bez wyboru modeli |
| test | 2020-01-01 – 2025-12-31 | 25 | ostateczna (2020), wstępna (2021–2025) | `data/sealed/test.parquet` | jednorazowa ocena końcowa |

Granice zbiorów są datami kalendarzowymi, a nie dokładnymi granicami cykli: train zawiera też koniec 22. cyklu (1995–1996), a 24. cykl zaczął się pod koniec 2008 roku. Każdy etap obejmuje jednak głównie jeden cykl. Kalibracja na całym 24. cyklu, a nie na latach 2015–2019, ma zapewnić burze w różnych fazach cyklu.

**Zbiór testowy** jest dotykany **tylko** w ocenie końcowej. Nie służy do strojenia, wyboru modelu, debugowania ani wykresów. Wyjątek, liczony wyłącznie z Dst, przed oceną jakiegokolwiek modelu na teście: liczba okien burz i bloków spokojnych w teście (§6), potrzebna też do analizy mocy (§10.4).

## 6. Burze

**Definicja** (z góry, wyłącznie z szeregu Dst):

1. Minimum burzy: rekord godzinowy o `t_start` = s, którego wartość Dst wynosi ≤ −50 nT i jest najmniejsza wśród rekordów o `t_start` w [s − 24 h, s + 24 h]. Przy remisie bierzemy najwcześniejszy rekord. Na brzegach danych (1995-01-01 i 2025-12-31) przedział obejmuje tylko dostępne rekordy. Poniżej t_min oznacza `t_start` minimum.
2. Okno burzy: [t_min − 24 h, t_min + 48 h]. Nakładające się okna łączymy; kategorię połączonego okna wyznacza najniższe minimum.
3. Okno należy do zbioru, w którym leży jego minimum (w oknie połączonym: najniższe minimum), i jest przycinane do granic tego zbioru.
4. Godzina prognozy należy do okna, jeśli jej **godzina celu** leży w oknie.
5. **Bloki spokojne:** godziny celu poza wszystkimi oknami dzielimy na kolejne bloki kalendarzowe 27-dniowe (jeden obrót Słońca), zaczynając od początku zbioru. Blok z mniej niż 72 godzinami spokojnymi dołączamy do poprzedniego, a jeśli jest pierwszym blokiem zbioru, do następnego. Godziny okna burzy przypisanego do sąsiedniego zbioru, odcięte przy przycinaniu (pkt 3), nie są godzinami spokojnymi i nie wchodzą do żadnej kategorii.
6. **Fazy burzy:** faza główna to godziny celu od początku okna do najniższego minimum okna włącznie; faza odnowy to godziny celu po tym minimum. W oknach połączonych z kilku burz fazy liczymy względem najniższego minimum.

**Kategorie** (progi −50 i −100 nT według Gonzaleza i in. 1994; próg −250 nT dla burz superintensywnych według Echera i in. 2008):

| Kategoria | Minimum Dst [nT] | Rola |
|---|---|---|
| spokojne | bloki z pkt 5 | odniesienie w H1 |
| umiarkowane | −100 < min ≤ −50 | raportowane zawsze |
| silne | min ≤ −100 | H1–H3 |
| superintensywne | min ≤ −250 (podzbiór silnych) | wyłącznie opisowo |

**Minimalna liczba burz.** Jeśli w teście jest mniej niż **10** okien burz silnych, H1–H3 raportujemy opisowo, bez rozstrzygnięcia. Liczbę ustalamy przed uruchomieniem modeli i wpisujemy do §15.

## 7. Modele i trening

- **Wejście wszystkich modeli:** tylko wartości znane w chwili t (§5.2). Bieżący Dst jest na wejściu, bo operacyjnie dostępny jest jego odpowiednik w czasie rzeczywistym; modele dostają jednak wersję ostateczną lub wstępną (§3, pkt 2). Dzięki temu persystencja jest trudnym punktem odniesienia.
- **P:** `Dst(t+h) = Dst(t)`.
- **E (AK2):** `dDst*/dt = Q − Dst*/τ`, `Q = −4,4·(VBs − 0,5)` nT/h dla VBs > 0,5 mV/m i 0 w przeciwnym razie, `τ = 2,4·exp(9,74 / (4,69 + VBs))` h, `Dst* = Dst − 7,26·√Pdyn + 11` (Dst w nT, Pdyn w nPa). VBs w mV/m: `V [km/s] · Bs [nT] · 10⁻³`, gdzie Bs = −Bz dla Bz < 0 i 0 w przeciwnym razie. Całkowanie jawnym schematem Eulera, krokiem 1 h, od `Dst*(t)` przez h kroków; przy kroku 1 h schemat jest stabilny, bo τ > 2,4 h. Wiatr słoneczny jest trzymany na wartości z chwili t, bo przyszłej nie znamy. Bez dopasowania stałych.
- **X:** XGBoost (regresja, strata kwadratowa) na zestawie cech z §7.1, jeden model na horyzont. Braki w cechach obsługuje natywnie. Każda cecha musi przejść test przyczynowości (§14). Stałe: `subsample = 0,8`, `colsample_bytree = 0,8` (żeby seed wpływał na model), `tree_method = hist`, uczenie na CPU; pozostałe parametry domyślne dla zamrożonej wersji pakietu (§14).
- **L:** okno 24 h zmiennych V, Bz i By w GSM, \|B\|, gęstość, ciśnienie dynamiczne, VBs i Dst; standaryzacja ze statystyk train; jedna sieć na horyzont; warstwa liniowa na ostatnim stanie ukrytym daje jedną liczbę; strata MSE, bez dropoutu.
- **Wszystkie dopasowywane przekształcenia** (standaryzacja, ewentualne imputery) uczą się wyłącznie na train.
- **Braki danych:** wejścia wszystkich modeli uzupełniamy ostatnią znaną wartością, najwyżej 3 h wstecz i wyłącznie z przeszłości. Po tym kroku: P wymaga `Dst(t)`; E wymaga `Dst(t)` oraz V, Bz i Pdyn w chwili t; L wymaga kompletnego okna 24 h wszystkich kanałów; X przyjmuje pozostałe braki natywnie. Model bez wymaganych wejść nie wydaje prognozy w tej godzinie. Ocena główna odbywa się tylko na wspólnym zbiorze godzin, w których **wszystkie** cztery modele mają prognozę i znany jest cel. Ponieważ luki w danych wiatru słonecznego mogą skupiać się w dużych burzach, raportujemy udział wyłączonych godzin w każdej kategorii (M10) i wykonujemy S5 (§9).
- **Seedy:** 5 seedów (42–46) dla X i L. Seed steruje inicjalizacją i kolejnością danych w L oraz podpróbkowaniem w X. Prognoza punktowa to średnia z 5 modeli; rozrzut między seedami raportujemy osobno.
- **Zespół jako jednostka:** hipotezy dotyczą prognozy uśrednionej po seedach, bo taki model wdrożono by w praktyce.

### 7.1 Cechy modelu X (ustalone z góry)

Wszystkie cechy liczone wyłącznie z wartości znanych w chwili t (§5.2); „opóźnienie k” oznacza wartość za przedział [t − k − 1 h, t − k). Średnie i sumy kroczące liczymy tylko z pełnych okien (po uzupełnieniu luk z §7); okno z brakiem daje brak cechy.

| Grupa | Cechy | Uzasadnienie |
|---|---|---|
| Stan prądu pierścieniowego | Dst z opóźnieniem 0, 1, 2, 3 i 6 h; `Dst(t) − Dst(t − 1 h)`; `Dst(t) − Dst(t − 3 h)` | bieżący stan i tempo zmian (faza główna czy odnowa) |
| Wiatr słoneczny | V, Bz i By w GSM, \|B\|, gęstość, Pdyn z opóźnieniem 0, 1, 2 i 3 h | bezpośrednie wymuszanie |
| Sprzężenie | VBs [mV/m]; funkcja sprzężenia Newella `V^(4/3)·B_T^(2/3)·\|sin(θ/2)\|^(8/3)`, gdzie `B_T = √(By² + Bz²)`, `θ = atan2(By, Bz)` (kąt zegarowy IMF; wartość bezwzględna, bo θ bywa ujemne); `√Pdyn` | wstrzykiwanie energii (Burton, O'Brien i McPherron) i poprawka ciśnieniowa w Dst* |
| Historia wymuszania | średnie kroczące z 3 i 6 h dla VBs, Bz, Pdyn i funkcji Newella; sumy VBs z 6 i 12 h | energia zgromadzona przed chwilą t |

Razem 44 cechy: 7 + 24 + 3 + 10. Lista jest zamrożona razem z protokołem; nie dodajemy ani nie usuwamy cech na podstawie wyników, także walidacyjnych. Ważność cech (SHAP) raportujemy wyłącznie opisowo.

### 7.2 Strojenie (równy budżet)

Strojenie wyłącznie w obrębie train, walidacją krzyżową blokami lat. Train dzielimy na 5 kolejnych bloków po 3 lata: 1995–1997, 1998–2000, 2001–2003, 2004–2006 i 2007–2009. W każdym foldzie jeden blok jest walidacyjny, a model uczy się na pozostałych, z których usuwamy po 27 dni przylegające do bloku walidacyjnego, żeby autokorelacja nie przenosiła informacji między nimi. Kryterium: średnie MSE walidacyjne z 5 foldów; osobny seed 100. Walidacja krzyżowa zastępuje jeden okres walidacyjny, bo lata 2006–2009 to głębokie minimum słoneczne z pojedynczymi burzami silnymi, a hiperparametry dobrane na takim okresie faworyzowałyby warunki spokojne. Każdy model ML dostaje 12 konfiguracji na horyzont, strojonych osobno dla h = 1 h i h = 6 h:

| Model | Siatka |
|---|---|
| X | max_depth ∈ {4, 6, 8} × learning_rate ∈ {0,03, 0,1} × min_child_weight ∈ {1, 10}; liczba drzew przez wczesne zatrzymanie (cierpliwość 50, maks. 5 000) |
| L | rozmiar ukryty ∈ {32, 64, 128} × lr ∈ {10⁻³, 3·10⁻³} × warstwy ∈ {1, 2}; Adam, batch 256, wczesne zatrzymanie (cierpliwość 5, maks. 100 epok) |

Wczesne zatrzymanie w każdym foldzie odbywa się na jego bloku walidacyjnym. Model końcowy trenujemy na całym train (1995–2009) z wybraną konfiguracją; liczba drzew lub epok to mediana najlepszych iteracji z 5 foldów tej konfiguracji. Jeśli optimum leży na krawędzi siatki, siatki nie rozszerzamy i raportujemy to z wynikami.

### 7.3 Przedziały

Miara niezgodności dla wszystkich metod: `|Dst − prognoza|`. Reszty kalibracyjne pochodzą z calib (modele trenowane wyłącznie na train), od 2010-01-28: pierwsze 27 dni calib pomijamy, żeby autokorelacja z końcem train nie zaniżała reszt. Dotyczy to wszystkich metod i M9.

- **C-split:** ⌈(n + 1)·(1 − α)⌉-ta najmniejsza reszta z n reszt kalibracyjnych, dla α = 0,1 (w M13 także dla innych poziomów). Szerokość jest stała dla danego modelu i horyzontu.
- **C-Mond:** jak C-split, ale kwantyle osobno dla klas `Dst(t)` w chwili wydania prognozy: `Dst(t) > −30`, `−50 < Dst(t) ≤ −30`, `−100 < Dst(t) ≤ −50`, `Dst(t) ≤ −100` nT. Klasa zależy od chwili wydania, nie od wyniku. Klasa z mniej niż 200 godzinami w calib jest łączona z sąsiednią, bliższą zeru. Łączenie ustalamy raz, na całym calib, i stosujemy bez zmian w teście i w M9; liczności klas raportujemy.
- **C-DtACI:** Gibbs i Candès (2024), wariant deterministyczny (algorytm 2: α(t) to średnia ważona poziomów ekspertów, a nie losowanie jednego). k = 8 ekspertów ACI z krokami γ ∈ {0,001; 0,002; 0,004; 0,008; 0,016; 0,032; 0,064; 0,128}, jak w pracy. Wagi aktualizowane stratą pinball `ℓ(β, θ) = α(β − θ) − min{0, β − θ}`, gdzie β(t) to największy poziom, przy którym cel leży w przedziale, a α = 0,1. Mieszanie `σ = 1/(2I)`, krok wag `η = √(3/I) · √((ln(k·I) + 2) / ((1 − α)²α²))` (wzory z pracy, z ich przybliżeniem straty). Okno I = 72 h, czyli długość okna burzy, zamiast 500 w pracy; daje to σ ≈ 0,0069 i η ≈ 6,6. Kwantyl rzędu 1 − α(t) z reszt kalibracyjnych.
- **C-ACI (S1):** jeden ekspert, `α(t+1) = α(t) + γ·(0,1 − err(t))`, γ ∈ {0,005; 0,05}.
- **Wspólne zasady metod adaptacyjnych:**
  - **Opóźnienie informacji zwrotnej:** prognoza wydana w chwili t daje `err` dopiero w t + h. Aktualizacja w chwili t używa tylko prognoz, których cel jest już znany.
  - Start: α = 0,1 (i równe wagi ekspertów) na 2010-01-01; przebieg przez cały calib i dalej przez test bez resetu.
  - Godziny bez prognozy (§7, braki danych) są pomijane: nie dają przedziału ani aktualizacji.
  - Gdy α ≤ 0, przedział jest nieskończony i liczony jako pokryty; gdy α ≥ 1, pusty i liczony jako niepokryty. Raportujemy liczbę takich godzin, a szerokość podajemy jako medianę.
- **Odniesienie na 24. cyklu (M9):** C-Mond i C-DtACI liczone w dwóch połowach calib krzyżowo: kwantyle z lat 2010–2014 stosowane do 2015–2019 i odwrotnie. Reszty dzielimy według godziny, a okno burzy przypisujemy do połowy, w której leży jego najniższe minimum. Dla C-DtACI rozkład reszt pochodzi z jednej połowy, a przebieg online przez drugą połowę zaczyna się od stanu początkowego (α = 0,1, równe wagi). Daje to pokrycie w burzach silnych poza próbą kalibracyjną, ale w obrębie cyklu kalibracji.

## 8. Metryki

Wszystkie liczone na wspólnym zbiorze godzin (§7), osobno dla każdego modelu, horyzontu i kategorii z §6.

**Wspólny zbiór godzin** ustalamy osobno dla każdego horyzontu; jest taki sam dla wszystkich metod przedziałów. Ablacje (§9) oceniamy na zbiorze z E-main, a godziny, w których model z ablacji nie ma prognozy, dodatkowo wyłączamy i raportujemy ich liczbę.

**Jednostka wchodzi do analizy**, jeśli w zbiorze wspólnym ma co najmniej połowę swoich godzin celu; pozostałe jednostki wyłączamy i liczymy w M10.

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
| M7 | kontrola „ukrytej persystencji”: przesunięcie k ∈ {0, …, h + 3} h, przy którym korelacja prognozy z obserwacją w oknie burzy jest najwyższa; mediana po oknach i udział okien z k ≥ h (por. Laperre i in. 2020); okna ze stałą prognozą pomijamy |
| M8 | dla pojedynczego okna: minimum prognozy minus minimum obserwacji [nT] oraz błąd momentu minimum [h]; przy remisie bierzemy najwcześniejszą godzinę |
| M9 | pokrycie C-Mond i C-DtACI w burzach silnych 24. cyklu, krzyżowo (§7.3); spadek `δ = M9 − M2(test)`, dla C-Mond statystyka H2 |
| M10 | udział godzin wyłączonych z oceny przez braki danych oraz liczba jednostek wyłączonych regułą połowy godzin, w każdej kategorii |
| M11 | krzywa dawka–odpowiedź: M1 i M2 liczone dla każdego okna osobno w funkcji minimum Dst okna; wykres punktowy i średnie w przedziałach minimum co 50 nT |
| M12 | przesunięcie prognoz punktowych: M3(test) / M3(calib) w każdej kategorii; calib jest dla modeli poza próbą treningową |
| M13 | krzywa pokrycia: M2 dla poziomów nominalnych 50, 60, 70, 80, 90 i 95% (C-split, C-Mond, C-DtACI), osobno dla godzin spokojnych i burz silnych |

M1–M5 raportujemy także osobno dla faz burzy (§6, pkt 6).

## 9. Eksperymenty

| ID | Opis | Rola |
|---|---|---|
| R0 | testy poprawności pipeline'u na prognozach poza foldem z walidacji krzyżowej (train), §9.1 | warunek wstępny; bez danych testowych |
| P0 | analiza mocy na 24. cyklu (§10.4) | informacyjny; bez danych testowych |
| E-main | P, E, X, L × h ∈ {1, 6} × {C-split, C-Mond, C-DtACI} na test | konfirmacyjny (H1, H2, H3) |
| S1 | C-ACI z γ ∈ {0,005; 0,05} | czułość na sposób adaptacji |
| S2 | statystyki ważone godzinami zamiast jednostkami | czułość na agregację |
| S3 | okna burz [t_min − 12 h, t_min + 24 h] zamiast §6 | czułość na definicję okna |
| S4 | M6 i analog H1 względem modelu E | czy wniosek zależy od wyboru punktu odniesienia |
| S5 | P i X na wszystkich godzinach, w których same mają prognozę (XGBoost obsługuje braki) | czułość na wyłączenia z powodu braków danych |
| S6 | „zostaw jedną burzę”: każda statystyka konfirmacyjna liczona bez każdego okna burzy silnej z testu po kolei | stabilność rozstrzygnięć |
| S7 | statystyka H2 z pokryciem na teście liczonym z kwantyli C-Mond z każdej połowy calib osobno (2010–2014 i 2015–2019), czyli przy tej samej wielkości próby kalibracyjnej co M9; raportujemy obie wartości i ich średnią | symetria porównania w H2 |
| X1 | okna burz, których najniższe minimum przypada 10–11 maja i 10–11 października 2024: M2, M4 i M8 | eksploracyjny, wyłącznie opisowo |
| X2 | M1–M3 dla wszystkich modeli i metod przedziałów, osobno dla 2020 (Dst ostateczny) i 2021–2025 (Dst wstępny); kategorie bez okien w danym okresie pomijamy | eksploracyjny; wpływ wersji etykiet (§3, pkt 1) |
| X3 | odpowiedniki H1–H3 dla kategorii superintensywnej | wyłącznie opisowo |
| X4 | jedno powtórzenie treningu L (seed 42, oba horyzonty) z tą samą konfiguracją | skala szumu między przebiegami z niedeterminizmu GPU; wyłącznie opisowo |
| A1 | X i L bez żadnych wejść z Dst (X bez grupy „stan prądu pierścieniowego”, L bez kanału Dst) | ablacja: czy przewaga nad persystencją pochodzi z wiatru słonecznego, czy z przedłużania bieżącego Dst |
| A2 | X bez grupy „sprzężenie” | ablacja: wkład funkcji sprzężenia |
| A3 | X bez grupy „historia wymuszania” | ablacja: wkład energii zgromadzonej przed chwilą t |
| A4 | X wyłącznie na grupie „stan prądu pierścieniowego” | ablacja: model czysto autoregresyjny, uzupełnienie A1 |

**Ablacje A1–A4** używają seedów 42–46, konfiguracji wybranych w strojeniu modelu pełnego (bez ponownego strojenia), tych samych podziałów danych i wszystkich trzech metod przedziałów. Mierzą więc czułość przy ustalonych hiperparametrach. Dla każdej ablacji raportujemy M1, M2 i M7 w każdej kategorii oraz różnicę względem odpowiednika z E-main (ten sam model, horyzont i seedy), z 95% przedziałem bootstrapowym (§10.1).

### 9.1 Testy poprawności (R0)

Wykonywane na prognozach poza foldem (out-of-fold) z walidacji krzyżowej wybranej konfiguracji, na całym okresie train, przed jakimkolwiek kontaktem z calib i test. Niepowodzenie któregokolwiek oznacza błąd do znalezienia, a nie wynik; znaleziony błąd i jego poprawkę wpisujemy do §13.

1. **Kontrola pozytywna:** X i L muszą mieć MSE niższe niż P dla obu horyzontów. Model, który nie bije persystencji na danych z tego samego cyklu, jest najpewniej źle podłączony.
2. **Kontrola wycieku przez cel:** X trenowany z celem dodanym jako cecha musi osiągnąć MAE bliskie zeru (< 1 nT). Potwierdza, że pipeline potrafi wykryć wyciek, gdy ten istnieje.
3. **Kontrola negatywna:** X trenowany na celu losowo przetasowanym w obrębie train nie może być lepszy od prognozy stałej (średniej z train) o więcej niż 1% MSE.
4. **Model E:** (a) test jednostkowy: przy stałym wymuszaniu VBs = 5 mV/m i Pdyn = 2 nPa `Dst*` zbiega do stanu ustalonego `Q·τ` z dokładnością 1%; sprawdza jednostki (VBs w mV/m, Pdyn w nPa) i całkowanie. (b) Korelacja prognozy z obserwacją > 0,8 dla h = 1 h; to tylko kontrola, że model jest podłączony, bo przy h = 1 h korelacja wynika głównie z `Dst(t)`.

**Analiza S1–S5, S7, A1–A4 i X1–X4** jest opisowa, bez testów hipotez: estymata punktowa i 95% przedział bootstrapowy (§10).

**Kolejność przebiegów** (najpierw najważniejsze dla interpretacji): strojenie, R0, P0, E-main, S6, S7, A1, A4, S5, S2, S4, A2, A3, S1, S3, X4, X1, X3, X2.

## 10. Statystyka i reguły decyzyjne

### 10.1 Jednostki i bootstrap

**Jednostka niezależna** to okno burzy lub blok spokojny (§6), nie godzina, bo godziny jednej burzy są silnie skorelowane.

**Bootstrap.** W każdej replikacji losujemy ze zwracaniem okna burz osobno w każdej kategorii oraz bloki spokojne. Jeden zestaw losowań jest wspólny dla wszystkich modeli, horyzontów i metod przedziałów, bo porównania są sparowane. 10 000 replikacji, seed 2026. Przedziały 95% percentylowe. Jednostronna wartość p to `(b + 1)/(B + 1)`, gdzie b to liczba replikacji, w których statystyka leży na granicy hipotezy zerowej lub po jej stronie (granica: 0 dla H1, H2 i H3), a B = 10 000; dzięki temu p nigdy nie jest równe zeru. Okna burz silnych z calib, potrzebne do M9 w H2, losujemy niezależnie od okien testowych (osobny strumień losowań, seed 2028), wspólnie dla wszystkich modeli i horyzontów.

**Zakres wnioskowania.** Bootstrap obejmuje zmienność zbioru testowego (burz i bloków). Wnioskowanie dotyczy wytrenowanego zespołu, czyli średniej z 5 seedów; zmienności treningu nie obejmuje, więc raportujemy ją opisowo (rozrzut seedów, X4).

**Korekta.** Metoda Holma osobno w każdej rodzinie: 4 testy na hipotezę (X, L × 1 h, 6 h), α = 0,05. Błąd I rodzaju kontrolujemy osobno w każdej z trzech rodzin (H1, H2, H3); nie stosujemy korekty między rodzinami.

**Progi praktyczne** (0,05 dla skill score, 5 punktów procentowych pokrycia w H2, 3 punkty dla sparowanej różnicy w H3) są ustalone z góry jako najmniejsze różnice, które uznajemy za praktycznie istotne dla użytkownika prognozy. Nie wynikają z danych; P0 pokazuje, jakie efekty są przy nich wykrywalne.

### 10.2 Reguły

**H1.** Dla modelu v i horyzontu h: `Δ = M1(spokojne) − M1(silne)`. H1 uznajemy za popartą dla (v, h), gdy jednocześnie:

- `Δ ≥ 0,05` (próg praktyczny),
- p skorygowane metodą Holma < 0,05.

**H2.** Dla C-Mond na modelu v i horyzoncie h: `δ = M9(silne) − M2(test; silne)`, czyli spadek pokrycia w burzach silnych między 24. a 25. cyklem. H2 uznajemy za popartą, gdy jednocześnie:

- `δ ≥ 0,05` (spadek o co najmniej 5 punktów procentowych),
- p Holma < 0,05 dla `δ > 0`.

Bezwzględne pokrycie M2(test; silne) raportujemy obok. Wniosek z H2 dotyczy przejścia ze słabego 24. cyklu na silniejszy 25. (§3, pkt 5).

**H3.** `D = M2(silne; C-DtACI) − M2(silne; C-Mond)`. H3 uznajemy za popartą, gdy jednocześnie:

- `D ≥ 0,03`,
- p Holma < 0,05 dla `D > 0`,
- estymata punktowa `M2(silne; C-DtACI) ≥ 0,85`.

Raportujemy także stosunek median szerokości C-DtACI do C-Mond; nie wchodzi do reguły.

### 10.3 Odporność

- **Mało skupień.** Bootstrap po kilkunastu skupieniach bywa zbyt optymistyczny. Dla H3, gdzie porównanie jest sparowane w obrębie okna, dodatkowo raportujemy test permutacyjny ze zmianą znaku różnic `D_j` po oknach (100 000 losowych zmian znaku, seed 2027; p liczone tak samo jak w §10.1). Rozstrzyga reguła z §10.2; rozbieżność z testem permutacyjnym zaznaczamy w raporcie.
- **Wpływ pojedynczej burzy (S6).** Jeśli usunięcie jednego okna zmienia rozstrzygnięcie którejkolwiek hipotezy, rozstrzygnięcie pozostaje formalnie ważne, ale raportujemy je jako **niestabilne**, z nazwą tej burzy.
- **Braki danych.** Jeśli M10 w kategorii silnej przekracza 10%, wyniki H1–H3 raportujemy z zastrzeżeniem i obok wyników S5.

### 10.4 Analiza mocy (P0, informacyjnie)

P0 wykonujemy po zamrożeniu protokołu, przed oceną na teście. Jest wyłącznie informacyjne: nie zmienia progów, reguł ani tego, które hipotezy są rozstrzygane. Służy do interpretacji ewentualnego wyniku zerowego.

1. Na calib, z modelami trenowanymi na train i z przedziałami C-Mond i C-DtACI liczonymi krzyżowo jak w M9, liczymy statystyki z §10.2.
2. Ich błąd standardowy SE szacujemy bootstrapem (10 000 replikacji, seed 2029), losując ze zwracaniem tyle okien burz silnych i bloków spokojnych, ile jest w teście (§6; liczba bloków spokojnych liczona przy P0, wyłącznie z Dst). Dla H2 `SE = √(SE_M9² + SE_test²)`: pierwszy składnik z losowania 26 okien calib, drugi z losowania tylu okien, ile jest w teście, oba na danych calib.
3. Hipoteza jest poparta, gdy estymata odległości od granicy H0 przekracza zarówno próg praktyczny d, jak i z·SE, gdzie z = z_{1−0,0125} ≈ 2,24 (najostrzejszy poziom Holma przy 4 testach). W przybliżeniu normalnym najmniejszy efekt wykrywalny z mocą 80% to `MDE80 = max(d, 2,24·SE) + 0,84·SE`, gdzie d = 0,05 dla H1 i H2 oraz 0,03 dla H3.
4. Raportujemy MDE80 dla każdej hipotezy, modelu i horyzontu. Warunek pokrycia C-DtACI ≥ 0,85 w H3 pomijamy w tym przybliżeniu i zaznaczamy to w raporcie.

**Dodatkowo (zawsze):** M1–M13 dla wszystkich modeli, horyzontów i kategorii, z 95% przedziałami bootstrapowymi; rozrzut M1 i M3 między seedami; C-split jako punkt odniesienia.

**Wynik zerowy jest wynikiem.** Jeśli hipoteza nie jest poparta, raport tak to stwierdza i pokazuje rozkłady bootstrapowe. Nie wolno zmieniać punktu końcowego, progu, kategorii ani definicji burzy po obejrzeniu wyników.

## 11. Zagrożenia dla trafności

| Zagrożenie | Postępowanie |
|---|---|
| Różne wersje Dst w treningu i teście | Nazwane w §3; X2 porównuje rok z wersją ostateczną z latami z wersją wstępną; ponowna ocena po publikacji Dst ostatecznego (poza zakresem) |
| Dane OMNI przetworzone po fakcie | Zakres wniosków (§12) |
| Wyciek przez przesunięcie godzin | Sprawdzenie wyrównania na latach 1995–2014 (§5.2) i test przyczynowości cech (§14) |
| Cechy nieprzyczynowe lub przekształcenia dopasowane na wszystkich danych | Test przyczynowości każdej cechy (§14); przekształcenia uczone tylko na train; kontrole R0 (§9.1) |
| Dobór cech pod wynik | Lista cech zamrożona z protokołem (§7.1); brak selekcji cech |
| Hipotezy sformułowane po lekturze prac z wynikami na 25. cyklu (m.in. Yu i in. 2026: łączne pokrycie poniżej nominalnego) | Ujawnione w pracy; ślepa jest ocena naszych modeli, nie wiedza o literaturze |
| Autorka zna przebieg burz z 2024 roku | Ślepa jest ocena modeli, nie wiedza o zdarzeniach; definicje burz i reguły ustalone przed oceną |
| Okna burz wyznaczone z obserwowanego Dst | §3; S3 sprawdza czułość na definicję okna |
| Mało silnych burz w teście; bootstrap przy małej liczbie skupień | Próg 10 burz (§6); analiza mocy P0; test permutacyjny dla H3; kategoria superintensywna tylko opisowo |
| Jedna burza (np. z maja 2024) dominuje statystyki | Agregacja z równą wagą okien (§8); S6 i oznaczenie rozstrzygnięć niestabilnych |
| Braki danych skupione w burzach | M10, próg 10% i S5 |
| Niestabilny skill score w godzinach spokojnych | M1b jako uzupełnienie; agregacja błędów zamiast ilorazów |
| Model E poza zakresem dopasowania w silnych burzach | Tylko eksploracyjnie; nazwane w §2 |
| Pokrycie poniżej nominalnego może wynikać z trudności burz, a nie z przesunięcia cykli | H2 testuje spadek względem 24. cyklu (M9), a nie poziom bezwzględny (§3, pkt 4) |
| Słaby cykl kalibracji (24.) wobec silniejszego testowego (25.) | §3, pkt 5; zawężone sformułowanie wniosku z H2 (§12) |
| Dst na wejściu w wersji lepszej niż dostępna operacyjnie | §3, pkt 2; trafność przy h = 1 h jako górna granica (§12) |
| Hiperparametry dobrane pod warunki spokojne | Walidacja krzyżowa blokami lat zamiast okresu minimum słonecznego (§7.2) |
| Asymetria w H2: M9 korzysta z kwantyli z połowy calib (5 lat), a test z całego calib (10 lat). Mniejsza próba daje ostrożniejsze kwantyle i większy rozrzut, więc M9 może być lekko zawyżone, co sprzyja H2; bootstrap po burzach nie obejmuje niepewności reszt kalibracyjnych | Nazwane jako ograniczenie; S7 liczy pokrycie na teście przy tej samej wielkości próby kalibracyjnej co M9 |
| Asymetria w H3: C-DtACI bez podziału na klasy, C-Mond z podziałem | Nazwana jako ograniczenie; wariant łączony poza zakresem |
| Reszty godzinowe nie są wymienialne (autokorelacja) | Gwarancja C-split i C-Mond nie jest dokładna już na calib; raportujemy pokrycie na calib jako kontrolę |
| Hiperparametry strojone tylko na 23. cyklu | Ograniczenie; równy budżet strojenia dla X i L (§7.2) |
| Burze są rzadkie, a trening nie waży ich mocniej | Celowe: oceniamy typową praktykę uczenia na MSE; ograniczenie nazwane w dyskusji |
| Wybór kroku w metodzie adaptacyjnej | C-DtACI dobiera krok sam; siatka γ i okno I ustalone z góry; C-ACI ze stałym γ w S1 |
| Rewizje danych po pobraniu | Jednorazowe pobranie, sumy SHA-256 i data w `MANIFEST.json` |
| Przewaga ML wynika z przedłużania bieżącego Dst, a nie z fizyki wymuszania | A1 i A4; kontrola M7 |
| Błąd w implementacji statystyki | Kod z §10 sprawdzony na danych syntetycznych o znanym efekcie przed oceną końcową (§14) |
| Niedeterminizm treningu na GPU | X4 jako miara szumu; prognoza jako średnia z 5 seedów |

## 12. Zakres wniosków

Wnioski dotyczą: prognoz Dst z horyzontem 1 h i 6 h, danych OMNI i Kyoto z §5, modeli P, E, X i L zbudowanych w tym projekcie, wytrenowanych na latach 1995–2009 i ocenionych na latach 2020–2025, metod przedziałów z §2 oraz definicji burz z §6. **Nie dotyczą** prognozy operacyjnej na danych z L1 w czasie rzeczywistym, innych indeksów (SYM-H, Kp), dłuższych horyzontów ani modeli niewymienionych w §2. Wnioski o burzach superintensywnych (≤ −250 nT) są opisowe. Wniosek z H2 dotyczy przejścia ze słabego 24. cyklu na silniejszy 25. (§3, pkt 5). Trafność przy h = 1 h jest górną granicą także dlatego, że Dst na wejściu nie jest wersją czasu rzeczywistego (§3, pkt 2). Modele nie widzą danych z 24. cyklu: są trenowane na latach 1995–2009 i ocenione na 2020–2025 bez douczania, więc wnioski z H1 dotyczą modeli z ponad dziesięcioletnią przerwą w danych treningowych; model operacyjny byłby douczany na nowszych danych.

**Porównanie z literaturą** (m.in. Sandu i Beșliu-Ionescu 2026, Yu i in. 2026) jest wyłącznie opisowe i odbywa się w dyskusji, po ocenie końcowej. Różnice w okresach testowych, wersjach Dst i metrykach sprawiają, że liczby nie są bezpośrednio porównywalne; nie wpływa ono na żadne rozstrzygnięcie.

## 13. Dziennik odstępstw (zmiany po zamrożeniu)

| Data | Co zmieniono | Powód | Wpływ na analizę |
|---|---|---|---|
| 2026-11-24 | Ujawnienie (przed zamrożeniem): przy sprawdzaniu formatu pobranych plików Kyoto wyświetlono pierwsze i ostatnie wiersze plików z lat 2020, 2021, 2024 i 2025 (łącznie 8 godzinowych wartości Dst) oraz liczności kodów wersji w tych latach | Przejrzystość | Znikomy; żaden model nie był oceniany, definicje burz, reguły i progi nie zależą od tych wartości |
| 2026-11-24 | Dopisanie DOI zarchiwizowanego protokołu w Zenodo do §14 | Publiczny, niezmienialny znacznik czasu zamrożonego protokołu | Brak |
| 2026-11-24 | Nowy wiersz w §11 (asymetria wielkości próby kalibracyjnej w H2) i eksperyment S7 w §9 | Przegląd protokołu po zamrożeniu: M9 i test korzystają z kwantyli z prób o różnej wielkości | Eksploracyjne; reguła H2 bez zmian; dodane przed jakimkolwiek treningiem |
| 2026-11-24 | Zdanie w §12 o braku douczania modeli na danych z 24. cyklu | Przegląd protokołu po zamrożeniu: zakres wniosków z H1 | Brak; doprecyzowanie zakresu wniosków |
| 2026-11-24 | Wydanie wersji 1.1 (tag `protocol-v1.1`, nowa wersja archiwum w Zenodo) | Zebranie zmian z przeglądu po zamrożeniu w jednej wersji archiwalnej, przed pierwszym treningiem | Brak |

Zmiany przed zamrożeniem są widoczne w historii Git. Każde użycie danych testowych przed oceną końcową, także w testach kodu, jest ujawniane w tej tabeli, nawet gdy żaden model nie był na nich oceniany.

## 14. Reprodukowalność

- Dane: `scripts/download_data.py`, uruchamiany raz; `data/MANIFEST.json` z datą pobrania, adresami, licznościami wersji Dst w każdym zbiorze, przesunięciem godzin, wersjami pakietów i sumami SHA-256.
- Wszystkie parametry modeli, przedziałów i bootstrapu w `configs/*.yaml`. Raport zawiera hash konfiguracji i commita.
- Dokładne wersje wszystkich pakietów (`pip freeze`) zapisujemy w repozytorium przed strojeniem i nie zmieniamy ich do końca oceny; zmiana wersji to wpis w §13.
- Testy automatyczne: parser OMNI i wartości braku, wyrównanie godzin, rozłączność podziałów, definicja okien burz na danych syntetycznych, przyczynowość każdej cechy (zmiana danych po chwili t nie zmienia cechy w chwili t), opóźnienie informacji zwrotnej w C-ACI i C-DtACI, stan ustalony modelu E przy stałym wymuszaniu, zgodność implementacji C-DtACI z przykładem z pracy Gibbsa i Candèsa.
- Skrypt oceny końcowej odmawia działania, jeśli suma SHA-256 pliku `data/sealed/test.parquet` nie zgadza się z `MANIFEST.json` albo bieżący commit nie jest potomkiem tagu najnowszej wersji protokołu (obecnie `protocol-v1.1`).
- Implementacja bootstrapu, korekty Holma i reguł z §10 jest sprawdzona na danych syntetycznych o znanym efekcie (z efektem i bez) przed oceną końcową.
- Każdy przebieg zapisuje na starcie commit i flagę zmian niezacommitowanych. Przebiegi E-main i ablacji uruchamiamy wyłącznie z czystego repozytorium.
- XGBoost na CPU jest powtarzalny przy tym samym seedzie; trening LSTM na GPU nie jest bitowo powtarzalny (X4 mierzy skalę różnic).
- `scripts/make_tables.py` odtwarza każdą tabelę i rysunek z zapisanych prognoz, bez ręcznych kroków.
- Zamrożony protokół z kodem archiwizujemy w Zenodo w chwili zamrożenia; wyniki kroków zaplanowanych po zamrożeniu (§15), w tym R0 i P0, w kolejnej wersji archiwum, przed jakąkolwiek oceną na teście. Wersja `protocol-v1.0`: [10.5281/zenodo.24511114](https://doi.org/10.5281/zenodo.24511114); wersja `protocol-v1.1`: DOI dopisywany po wydaniu; wszystkie wersje: [10.5281/zenodo.24511113](https://doi.org/10.5281/zenodo.24511113).

## 15. Decyzje przed zamrożeniem (rozstrzygnięte)

**Rozstrzygnięte:**

- **Horyzonty:** 1 h i 6 h równorzędnie, z korektą Holma na 4 testy w rodzinie.
- **Podział:** głównie cyklami 23 / 24 / 25 (§5.3); test obejmuje lata 2020–2025, w całości w wersji ostatecznej lub wstępnej (zmiana z wersji 0.4, w której test kończył się w czerwcu 2025 według strony Kyoto; kody wersji z HAPI pokazały, że cały 2025 jest już w wersji wstępnej).
- **Poziom nominalny:** 90%, bo daje więcej przypadków poza przedziałem w burzach niż 95%, a więc większą moc.
- **Jednostka analizy:** okno burzy lub blok spokojny; agregacja z równą wagą jednostek.
- **Metody konfirmacyjne:** C-Mond (H2) i C-DtACI (H3). C-split nie jest testowany, bo nie obiecuje pokrycia warunkowego (zmiana z wersji 0.1).
- **Modele:** wszystkie budowane w projekcie od zera; cechy modelu X zamrożone w §7.1 (zmiana z wersji 0.2, w której model X opierał się na kodzie Sandu i Beșliu-Ionescu).
- **Wyrównanie godzin:** przesunięcie Kyoto względem OMNI wynosi 0 h, przy zgodności 100% na latach 1995–2014; dla innych przesunięć zgodność wynosi 7–15% (`MANIFEST.json`, pobranie 2026-11-24).
- **Kolumny OMNI:** numery i wartości braku w `OMNI_COLS` zgodne z `omni2.text`.
- **Stałe modelu E:** b = 7,26 nT·nPa^−1/2 i c = 11 nT, zgodnie z O'Brienem i McPherronem (2000).

- **Strojenie:** osobno dla każdego horyzontu, siatki z §7.2, walidacja krzyżowa 5 blokami po 3 lata.
- **H2:** testuje spadek pokrycia C-Mond względem 24. cyklu (M9), z niezależnym bootstrapem okien z obu cykli.
- **C-DtACI:** wariant deterministyczny, siatka γ, σ i η według Gibbs i Candès (2024), okno I = 72 h (§7.3).
- **Progi praktyczne:** Δ ≥ 0,05; δ ≥ 0,05; D ≥ 0,03 (§10.2); P0 jest informacyjne i nie wpływa na rozstrzygnięcia (§10.4).

- **Liczba okien burz** według §6 (`scripts/count_storms.py`, wyłącznie z Dst; `results/storm_counts.json`):

| Zbiór | Umiarkowane | Silne | W tym superintensywne |
|---|---|---|---|
| train | 203 | 80 | 8 |
| calib | 129 | 26 | 0 |
| test | 81 | 24 | 2 |

  Test ma 24 okna burz silnych, powyżej progu 10 z §6, więc H1–H3 są rozstrzygane formalnie. Calib nie zawiera burzy superintensywnej, więc przedziały nie były kalibrowane na takim przypadku.

**Kroki zaplanowane po zamrożeniu, przed oceną na teście** (wyniki wpisywane do §13 jako realizacja planu, nie odstępstwa):

1. Szacunek czasu obliczeń po teście dymnym na GTX 1660.
2. Strojenie i R0 (§7.2, §9.1).
3. Liczności klas C-Mond na calib i łączenie klas według reguły z §7.3.
4. P0 (§10.4).
5. Archiwizacja wyników kroków 1–4 w Zenodo, potem E-main.

## Literatura

- Echer, E., Gonzalez, W. D., Tsurutani, B. T. (2008). Interplanetary conditions leading to superintense geomagnetic storms (Dst ≤ −250 nT) during solar cycle 23. *Geophysical Research Letters*, 35. [doi.org/10.1029/2007GL031755](https://doi.org/10.1029/2007GL031755)
- Gibbs, I., Candès, E. (2021). Adaptive conformal inference under distribution shift. arXiv:2106.00170. [arxiv.org/abs/2106.00170](https://arxiv.org/abs/2106.00170)
- Gibbs, I., Candès, E. (2024). Conformal inference for online prediction with arbitrary distribution shifts. *Journal of Machine Learning Research*, 25(162), 1–36. [www.jmlr.org/papers/v25/22-1218.html](https://www.jmlr.org/papers/v25/22-1218.html)
- Gonzalez, W. D., Joselyn, J. A., Kamide, Y., Kroehl, H. W., Rostoker, G., Tsurutani, B. T., Vasyliunas, V. M. (1994). What is a geomagnetic storm? *Journal of Geophysical Research*, 99(A4), 5771–5792. [doi.org/10.1029/93JA02867](https://doi.org/10.1029/93JA02867)
- King, J. H., Papitashvili, N. E. (2005). Solar wind spatial scales in and comparisons of hourly Wind and ACE plasma and magnetic field data. *Journal of Geophysical Research*, 110, A02104. [doi.org/10.1029/2004JA010649](https://doi.org/10.1029/2004JA010649)
- Laperre, B., Amaya, J., Lapenta, G. (2020). Dynamic time warping as a new evaluation for Dst forecast with machine learning. arXiv:2006.04667. [arxiv.org/abs/2006.04667](https://arxiv.org/abs/2006.04667)
- Newell, P. T., Sotirelis, T., Liou, K., Meng, C.-I., Rich, F. J. (2007). A nearly universal solar wind–magnetosphere coupling function inferred from 10 magnetospheric state variables. *Journal of Geophysical Research*, 112, A01206. [doi.org/10.1029/2006JA012015](https://doi.org/10.1029/2006JA012015)
- NOAA (2020). Analiza Solar Cycle 25 Prediction Panel: minimum między cyklami 24 i 25 w grudniu 2019; cykl 24 najsłabszy od 100 lat. [weather.gov/news/201509-solar-cycle](https://weather.gov/news/201509-solar-cycle)
- O'Brien, T. P., McPherron, R. L. (2000). An empirical phase space analysis of ring current dynamics: Solar wind control of injection and decay. *Journal of Geophysical Research*, 105(A4), 7707–7719. [doi.org/10.1029/1998JA000437](https://doi.org/10.1029/1998JA000437)
- Sandu, I. C., Beșliu-Ionescu, D. (2026). Operational Dst forecasting across solar cycle using physics-motivated gradient boosting. *Frontiers in Astronomy and Space Sciences*. [www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2026.1835176/full](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2026.1835176/full)
- World Data Center for Geomagnetism, Kyoto, Nose, M., Iyemori, T., Sugiura, M., Kamei, T. et al. (2015). Geomagnetic Dst index. [doi.org/10.17593/14515-74000](https://doi.org/10.17593/14515-74000)
- Yu, Zou, Pulkkinen, Chen (2026). Stable multivariate functional time series prediction for major geomagnetic indices. arXiv:2606.14417. [arxiv.org/abs/2606.14417](https://arxiv.org/abs/2606.14417)
