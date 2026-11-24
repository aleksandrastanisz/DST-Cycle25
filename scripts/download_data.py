"""Pobiera dane do projektu DST-Cycle25 i dzieli je na trening / kalibrację / test.

Źródła:
  - NASA SPDF, OMNI2 godzinowe (wiatr słoneczny, IMF), pliki roczne omni2_RRRR.dat
  - Kyoto WDC, Dst godzinowy przez HAPI (dataset "hour_dst")

Uruchom RAZ, z katalogu projektu:
    python scripts/download_data.py

Wynik:
    data/raw/                surowe pliki + migawka strony Kyoto z zakresami wersji Dst
    data/processed/train.parquet   1995-01-01 .. 2009-12-31  (23. cykl)
    data/processed/calib.parquet   2010-01-01 .. 2019-12-31  (24. cykl)
    data/sealed/test.parquet       2020-01-01 .. 2025-06-30  (25. cykl) - NIE OTWIERAĆ przed zamrożeniem protokołu
    data/MANIFEST.json       data pobrania, adresy, sumy SHA-256, wersje pakietów

Konwencja czasu (protokół, sekcja "Dane i podział"):
    t_start = początek godziny, której średnią opisuje rekord.
    Wartość za przedział [t_start, t_start + 1 h) jest znana dopiero w chwili t_start + 1 h.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SEALED = ROOT / "data" / "sealed"

FIRST_YEAR, LAST_YEAR = 1995, 2025

SPLITS = {  # zamknięte przedziały dat (UTC), zgodnie z protokołem
    "train": ("1995-01-01", "2009-12-31 23:00"),
    "calib": ("2010-01-01", "2019-12-31 23:00"),
    "test": ("2020-01-01", "2025-06-30 23:00"),
}

OMNI_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
OMNI_FMT_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text"
KYOTO_HAPI = "https://wdc.kugi.kyoto-u.ac.jp/hapi"
KYOTO_DATASET = "hour_dst"
KYOTO_DST_HOME = "https://wdc.kugi.kyoto-u.ac.jp/dstdir/"

# Kolumny OMNI2 numerowane od 1, jak w omni2.text -> (nazwa, wartość oznaczająca brak).
# Skrypt zapisuje omni2.text do data/raw/; numery i wartości braku SPRAWDŹ z tym plikiem.
OMNI_COLS = {
    1: ("year", None),
    2: ("doy", None),
    3: ("hour", None),
    9: ("B", 999.9),            # średnia |B|, nT
    16: ("By_gsm", 999.9),      # nT
    17: ("Bz_gsm", 999.9),      # nT
    23: ("T", 9999999.0),       # temperatura protonów, K
    24: ("n", 999.9),           # gęstość protonów, cm^-3
    25: ("V", 9999.0),          # prędkość, km/s
    29: ("Pdyn", 99.99),        # ciśnienie dynamiczne, nPa
    36: ("E", 999.99),          # pole elektryczne, mV/m
    41: ("Dst_omni", 99999.0),  # Dst w OMNI, tylko do sprawdzenia wyrównania godzin
}
OMNI_N_COLS = 55


# ---------------------------------------------------------------- pobieranie

def get(url: str, params: dict | None = None, tries: int = 4) -> requests.Response:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            wait = 5 * 2**i
            print(f"  błąd ({e}); ponawiam za {wait} s")
            time.sleep(wait)
    raise RuntimeError("nieosiągalne")


def save(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- OMNI

def parse_omni(text: str) -> pd.DataFrame:
    """Parsuje plik omni2_RRRR.dat; wartości braku zamienia na NaN."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    bad = [i for i, r in enumerate(rows) if len(r) != OMNI_N_COLS]
    if bad:
        raise ValueError(f"{len(bad)} wierszy OMNI nie ma {OMNI_N_COLS} kolumn (pierwszy: {bad[0]})")
    arr = np.array(rows, dtype=float)
    df = pd.DataFrame({name: arr[:, col - 1] for col, (name, _) in OMNI_COLS.items()})
    for col, (name, fill) in OMNI_COLS.items():
        if fill is not None:
            # porównanie z tolerancją, bo wartości braku są zapisane jako tekst
            df.loc[np.isclose(df[name], fill), name] = np.nan
    df["t_start"] = (
        pd.to_datetime(df["year"].astype(int).astype(str) + "-01-01", utc=True)
        + pd.to_timedelta(df["doy"] - 1, unit="D")
        + pd.to_timedelta(df["hour"], unit="h")
    )
    return df.drop(columns=["year", "doy", "hour"])


def download_omni() -> pd.DataFrame:
    save(RAW / "omni" / "omni2.text", get(OMNI_FMT_URL).content)
    frames = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        path = RAW / "omni" / f"omni2_{year}.dat"
        if not path.exists():
            print(f"OMNI {year}")
            save(path, get(OMNI_URL.format(year=year)).content)
        frames.append(parse_omni(path.read_text()))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- Kyoto

def hapi_get(endpoint: str, dataset: str, extra: dict | None = None) -> requests.Response:
    """HAPI 3 (dataset/start/stop), a jeśli serwer odmówi - HAPI 2 (id/time.min/time.max)."""
    extra = extra or {}
    try:
        return get(f"{KYOTO_HAPI}/{endpoint}", {"dataset": dataset, **extra}, tries=2)
    except requests.HTTPError:
        renamed = {{"start": "time.min", "stop": "time.max"}.get(k, k): v for k, v in extra.items()}
        return get(f"{KYOTO_HAPI}/{endpoint}", {"id": dataset, **renamed})


def kyoto_version_ranges() -> dict:
    """Czyta ze strony Kyoto zakresy wersji Dst, np. 'Final Dst index [1957 - 2020]'."""
    html = get(KYOTO_DST_HOME).text
    save(RAW / "kyoto" / "dstdir_snapshot.html", html.encode())
    final = re.search(r"Final Dst index\s*\[(\d{4})\s*-\s*(\d{4})\]", html)
    prov = re.search(r"Provisional Dst index\s*\[(\d{4})/(\d{2})\s*-\s*(\d{4})/(\d{2})\]", html)
    if not (final and prov):
        sys.exit("Nie umiem odczytać zakresów wersji Dst ze strony Kyoto. "
                 "Sprawdź data/raw/kyoto/dstdir_snapshot.html i wpisz je ręcznie.")
    final_end = pd.Timestamp(f"{int(final.group(2)) + 1}-01-01", tz="UTC")
    prov_end = pd.Timestamp(f"{prov.group(3)}-{prov.group(4)}-01", tz="UTC") + pd.offsets.MonthBegin(1)
    return {"final_until_exclusive": final_end, "provisional_until_exclusive": prov_end,
            "final_text": final.group(0), "provisional_text": prov.group(0)}


def download_kyoto() -> tuple[pd.DataFrame, dict]:
    info = hapi_get("info", KYOTO_DATASET).json()
    save(RAW / "kyoto" / "hapi_info.json", json.dumps(info, indent=2).encode())
    params = [p["name"] for p in info["parameters"]]
    print(f"Kyoto HAPI: parametry {params}")
    frames = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        path = RAW / "kyoto" / f"dst_{year}.csv"
        if not path.exists():
            print(f"Kyoto Dst {year}")
            r = hapi_get("data", KYOTO_DATASET,
                         {"start": f"{year}-01-01T00:00:00Z", "stop": f"{year + 1}-01-01T00:00:00Z",
                          "format": "csv"})
            save(path, r.content)
        df = pd.read_csv(path, header=None, names=params)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    time_col, value_cols = params[0], params[1:]
    if len(value_cols) != 1:
        sys.exit(f"Oczekiwałam jednej kolumny z Dst, jest {value_cols}. Sprawdź hapi_info.json.")
    fill = next((p.get("fill") for p in info["parameters"] if p["name"] == value_cols[0]), None)
    out = pd.DataFrame({"t_kyoto": pd.to_datetime(df[time_col], utc=True),
                        "Dst": pd.to_numeric(df[value_cols[0]], errors="coerce")})
    if fill is not None:
        out.loc[np.isclose(out["Dst"], float(fill)), "Dst"] = np.nan
    return out.drop_duplicates("t_kyoto").sort_values("t_kyoto"), info


# ---------------------------------------------------------------- wyrównanie godzin

def align_kyoto(omni: pd.DataFrame, kyoto: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ustala, jak znacznik czasu z HAPI ma się do t_start z OMNI.

    Dst w OMNI pochodzi z Kyoto, więc dla lat, w których obie bazy mają wersję ostateczną,
    wartości muszą się zgadzać co do joty przy właściwym przesunięciu. Sprawdzamy tylko
    lata 1995-2014 (trening i wczesna kalibracja), żeby nie dotykać danych testowych.
    """
    ref = omni[(omni.t_start >= "1995-01-01") & (omni.t_start < "2015-01-01")][["t_start", "Dst_omni"]]
    results = {}
    for lag_h in range(-3, 4):
        k = kyoto.assign(t_start=kyoto.t_kyoto + pd.Timedelta(hours=lag_h))[["t_start", "Dst"]]
        m = ref.merge(k, on="t_start").dropna()
        results[lag_h] = float((m.Dst_omni == m.Dst).mean()) if len(m) else 0.0
    best = max(results, key=results.get)
    print("Zgodność Dst Kyoto z Dst w OMNI (1995-2014) dla przesunięć [h]:")
    for lag_h, frac in results.items():
        print(f"  {lag_h:+d} h: {frac:.4f}" + ("   <- wybrane" if lag_h == best else ""))
    if results[best] < 0.99:
        sys.exit("Żadne przesunięcie nie daje >= 99% zgodności. Zatrzymuję się: wyrównanie trzeba "
                 "wyjaśnić ręcznie, zanim cokolwiek pójdzie dalej.")
    out = kyoto.assign(t_start=kyoto.t_kyoto + pd.Timedelta(hours=best))
    return out.drop(columns="t_kyoto"), {"lag_hours": best, "match_fraction": results}


def label_version(t: pd.Series, ranges: dict) -> pd.Series:
    v = pd.Series("realtime", index=t.index)
    v[t < ranges["provisional_until_exclusive"]] = "provisional"
    v[t < ranges["final_until_exclusive"]] = "final"
    return v


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    if (ROOT / "data" / "MANIFEST.json").exists():
        sys.exit("data/MANIFEST.json już istnieje: dane pobiera się raz. "
                 "Jeśli naprawdę chcesz pobrać od nowa, usuń katalog data/ i zapisz powód w rejestrze odstępstw.")

    downloaded_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    omni = download_omni()
    kyoto, hapi_info = download_kyoto()
    ranges = kyoto_version_ranges()
    kyoto, alignment = align_kyoto(omni, kyoto)

    df = omni.merge(kyoto, on="t_start", how="left").sort_values("t_start").reset_index(drop=True)
    df["dst_version"] = label_version(df.t_start, ranges)
    if df.t_start.duplicated().any():
        sys.exit("Zduplikowane godziny po połączeniu danych.")

    for d in (PROC, SEALED):
        d.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, (lo, hi) in SPLITS.items():
        part = df[(df.t_start >= pd.Timestamp(lo, tz="UTC")) & (df.t_start <= pd.Timestamp(hi, tz="UTC"))]
        path = (SEALED if name == "test" else PROC) / f"{name}.parquet"
        part.to_parquet(path, index=False)
        outputs[name] = path
        expected = int((pd.Timestamp(hi) - pd.Timestamp(lo)) / pd.Timedelta(hours=1)) + 1
        print(f"\n{name}: {len(part)} godzin (oczekiwano {expected}), wersje Dst: "
              f"{part.dst_version.value_counts().to_dict()}")
        if name != "test":  # statystyk testu celowo nie pokazujemy
            print(part.drop(columns=["t_start", "dst_version"]).isna().mean().round(3).to_string())

    raw_files = sorted(p for p in RAW.rglob("*") if p.is_file())
    manifest = {
        "downloaded_at_utc": downloaded_at,
        "sources": {"omni": OMNI_URL, "omni_format": OMNI_FMT_URL,
                    "kyoto_hapi": f"{KYOTO_HAPI} dataset={KYOTO_DATASET}", "kyoto_versions": KYOTO_DST_HOME},
        "kyoto_version_ranges": {k: str(v) for k, v in ranges.items()},
        "kyoto_hapi_version": hapi_info.get("HAPI"),
        "time_alignment": alignment,
        "splits": SPLITS,
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "requests": requests.__version__},
        "sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
                   for p in raw_files + list(outputs.values())},
    }
    (ROOT / "data" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("\nGotowe. Zapisano data/MANIFEST.json - zacommituj go (same dane są w .gitignore).")


if __name__ == "__main__":
    main()
