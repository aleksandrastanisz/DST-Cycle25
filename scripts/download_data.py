"""Download the data for DST-Cycle25 once and split it into train / calib / test.

Sources:
  - NASA SPDF, hourly OMNI2 (solar wind, IMF), yearly files omni2_YYYY.dat
  - Kyoto WDC, hourly Dst via HAPI (dataset "hour_dst"), with a version code for every hour

Run ONCE, from the project root:
    python scripts/download_data.py

Output:
    data/raw/                      raw files (already downloaded files are not fetched again)
    data/processed/train.parquet   1995-01-01 .. 2009-12-31  (mainly cycle 23)
    data/processed/calib.parquet   2010-01-01 .. 2019-12-31  (cycle 24)
    data/sealed/test.parquet       2020-01-01 .. 2025-12-31  (cycle 25) - DO NOT OPEN before the protocol is frozen
    data/MANIFEST.json             download date, URLs, Dst versions, hour offset, SHA-256 checksums

Time convention (protocol, section 5.2):
    t_start = start of the hour whose mean the record describes.
    The value for [t_start, t_start + 1 h) is known only at time t_start + 1 h.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
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

SPLITS = {  # closed date ranges (UTC), as in the protocol
    "train": ("1995-01-01", "2009-12-31 23:00"),
    "calib": ("2010-01-01", "2019-12-31 23:00"),
    "test": ("2020-01-01", "2025-12-31 23:00"),
}

OMNI_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
OMNI_FMT_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text"
KYOTO_HAPI = "https://wdc.kugi.kyoto-u.ac.jp/hapi"
KYOTO_DATASET = "hour_dst"

# OMNI2 columns numbered from 1, as in omni2.text -> (name, fill value).
# omni2.text is saved to data/raw/omni/; CHECK these numbers and fill values against it.
OMNI_COLS = {
    1: ("year", None),
    2: ("doy", None),
    3: ("hour", None),
    9: ("B", 999.9),            # mean |B|, nT
    16: ("By_gsm", 999.9),      # nT
    17: ("Bz_gsm", 999.9),      # nT
    23: ("T", 9999999.0),       # proton temperature, K
    24: ("n", 999.9),           # proton density, cm^-3
    25: ("V", 9999.0),          # flow speed, km/s
    29: ("Pdyn", 99.99),        # flow pressure, nPa
    36: ("E", 999.99),          # electric field, mV/m
    41: ("Dst_omni", 99999.0),  # Dst as stored in OMNI, used only to check hour alignment
}
OMNI_N_COLS = 55

# Kyoto versionCode = 10*M + m; M: 0 = real-time, 1 = provisional, 2 = final
KYOTO_VERSIONS = {0: "realtime", 1: "provisional", 2: "final"}


# ---------------------------------------------------------------- download helpers

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
            print(f"  error ({e}); retrying in {wait} s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def save(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def fetch_once(path: Path, url: str, params: dict | None = None) -> None:
    if not path.exists():
        save(path, get(url, params).content)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- OMNI

def parse_omni(text: str) -> pd.DataFrame:
    """Parse an omni2_YYYY.dat file; fill values become NaN."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    bad = [i for i, r in enumerate(rows) if len(r) != OMNI_N_COLS]
    if bad:
        raise ValueError(f"{len(bad)} OMNI rows do not have {OMNI_N_COLS} columns (first: row {bad[0]})")
    arr = np.array(rows, dtype=float)
    df = pd.DataFrame({name: arr[:, col - 1] for col, (name, _) in OMNI_COLS.items()})
    for _, (name, fill) in OMNI_COLS.items():
        if fill is not None:
            df.loc[np.isclose(df[name], fill), name] = np.nan
    df["t_start"] = (
        pd.to_datetime(df["year"].astype(int).astype(str) + "-01-01", utc=True)
        + pd.to_timedelta(df["doy"] - 1, unit="D")
        + pd.to_timedelta(df["hour"], unit="h")
    )
    return df.drop(columns=["year", "doy", "hour"])


def download_omni() -> pd.DataFrame:
    fetch_once(RAW / "omni" / "omni2.text", OMNI_FMT_URL)
    frames = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        path = RAW / "omni" / f"omni2_{year}.dat"
        if not path.exists():
            print(f"OMNI {year}")
            fetch_once(path, OMNI_URL.format(year=year))
        frames.append(parse_omni(path.read_text()))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- Kyoto

def hapi_get(endpoint: str, extra: dict | None = None) -> requests.Response:
    """HAPI 3 (dataset/start/stop); if the server refuses, HAPI 2 (id/time.min/time.max)."""
    extra = extra or {}
    try:
        return get(f"{KYOTO_HAPI}/{endpoint}", {"dataset": KYOTO_DATASET, **extra}, tries=2)
    except requests.HTTPError:
        renamed = {{"start": "time.min", "stop": "time.max"}.get(k, k): v for k, v in extra.items()}
        return get(f"{KYOTO_HAPI}/{endpoint}", {"id": KYOTO_DATASET, **renamed})


def parse_kyoto(df: pd.DataFrame, info: dict) -> pd.DataFrame:
    """Convert raw HAPI rows (Time, dstValue, versionCode) to t_start, Dst, dst_version."""
    names = [p["name"] for p in info["parameters"]]
    if names != ["Time", "dstValue", "versionCode"]:
        sys.exit(f"Unexpected Kyoto HAPI parameters {names}; check data/raw/kyoto/hapi_info.json.")
    if info.get("cadence") != "PT1H":
        sys.exit(f"Unexpected Kyoto cadence {info.get('cadence')}, expected PT1H.")

    t = pd.to_datetime(df["Time"], utc=True)
    location = info.get("timeStampLocation", "begin")
    if location == "center":
        if not ((t.dt.minute == 29) & (t.dt.second == 30)).all():
            sys.exit("Kyoto timestamps are declared as bin centres but are not all at HH:29:30.")
        t_start = t.dt.floor("h")
    elif location == "begin":
        t_start = t
    else:
        sys.exit(f"Unsupported timeStampLocation '{location}'.")

    fill = next(p.get("fill") for p in info["parameters"] if p["name"] == "dstValue")
    dst = pd.to_numeric(df["dstValue"], errors="coerce").astype(float)
    if fill is not None:
        dst[dst == float(fill)] = np.nan

    code = pd.to_numeric(df["versionCode"], errors="raise").astype(int)
    major = code // 10
    unknown = sorted(set(major) - set(KYOTO_VERSIONS))
    if unknown:
        sys.exit(f"Unknown Kyoto version codes (major part {unknown}).")

    out = pd.DataFrame({"t_kyoto": t_start, "Dst": dst, "dst_version": major.map(KYOTO_VERSIONS),
                        "dst_version_code": code})
    if out.t_kyoto.duplicated().any():
        sys.exit("Duplicate hours in Kyoto Dst.")
    return out.sort_values("t_kyoto").reset_index(drop=True)


def download_kyoto() -> tuple[pd.DataFrame, dict]:
    info_path = RAW / "kyoto" / "hapi_info.json"
    if not info_path.exists():
        save(info_path, json.dumps(hapi_get("info").json(), indent=2).encode())
    info = json.loads(info_path.read_text())
    frames = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        path = RAW / "kyoto" / f"dst_{year}.csv"
        if not path.exists():
            print(f"Kyoto Dst {year}")
            r = hapi_get("data", {"start": f"{year}-01-01T00:00:00Z", "stop": f"{year + 1}-01-01T00:00:00Z",
                                  "format": "csv"})
            save(path, r.content)
        frames.append(pd.read_csv(path, header=None, names=["Time", "dstValue", "versionCode"]))
    return parse_kyoto(pd.concat(frames, ignore_index=True), info), info


# ---------------------------------------------------------------- hour alignment

def align_kyoto(omni: pd.DataFrame, kyoto: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Check how Kyoto hours map onto OMNI t_start.

    Dst in OMNI comes from Kyoto, so for years where both hold the final version the values
    must agree exactly at the right offset. Only 1995-2014 is used, so test data are not touched.
    """
    ref = omni[(omni.t_start >= "1995-01-01") & (omni.t_start < "2015-01-01")][["t_start", "Dst_omni"]]
    results = {}
    for lag_h in range(-3, 4):
        k = kyoto.assign(t_start=kyoto.t_kyoto + pd.Timedelta(hours=lag_h))[["t_start", "Dst"]]
        m = ref.merge(k, on="t_start").dropna()
        results[lag_h] = float((m.Dst_omni == m.Dst).mean()) if len(m) else 0.0
    best = max(results, key=results.get)
    print("Agreement of Kyoto Dst with Dst in OMNI (1995-2014) by offset [h]:")
    for lag_h, frac in results.items():
        print(f"  {lag_h:+d} h: {frac:.4f}")
    print(f"Offset: {best:+d} h")
    if results[best] < 0.99:
        sys.exit("No offset reaches 99% agreement. Stopping: the alignment must be explained "
                 "before anything else is done.")
    out = kyoto.assign(t_start=kyoto.t_kyoto + pd.Timedelta(hours=best))
    return out.drop(columns="t_kyoto"), {"lag_hours": best, "match_fraction": results}


# ---------------------------------------------------------------- main

def main() -> None:
    if (ROOT / "data" / "MANIFEST.json").exists():
        sys.exit("data/MANIFEST.json already exists: data are downloaded once. To download again, "
                 "delete data/ and record the reason in the deviations log.")

    downloaded_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    omni = download_omni()
    kyoto, hapi_info = download_kyoto()
    kyoto, alignment = align_kyoto(omni, kyoto)

    df = omni.merge(kyoto, on="t_start", how="left").sort_values("t_start").reset_index(drop=True)
    if df.t_start.duplicated().any():
        sys.exit("Duplicate hours after merging OMNI and Kyoto.")

    for d in (PROC, SEALED):
        d.mkdir(parents=True, exist_ok=True)
    outputs, versions = {}, {}
    for name, (lo, hi) in SPLITS.items():
        part = df[(df.t_start >= pd.Timestamp(lo, tz="UTC")) & (df.t_start <= pd.Timestamp(hi, tz="UTC"))]
        expected = int((pd.Timestamp(hi) - pd.Timestamp(lo)) / pd.Timedelta(hours=1)) + 1
        versions[name] = {str(k): int(v) for k, v in part.dst_version.value_counts(dropna=False).items()}
        print(f"\n{name}: {len(part)} hours (expected {expected}), Dst versions: {versions[name]}")
        if name in ("train", "calib") and set(part.dst_version.dropna()) != {"final"}:
            sys.exit(f"{name} must contain final Dst only.")
        if name == "test" and "realtime" in set(part.dst_version.dropna()):
            print("  WARNING: test contains real-time Dst; record this in the protocol before freezing.")
        path = (SEALED if name == "test" else PROC) / f"{name}.parquet"
        part.to_parquet(path, index=False)
        outputs[name] = path
        if name != "test":  # test statistics are deliberately not printed
            print("  fraction missing:")
            print(part.drop(columns=["t_start", "dst_version", "dst_version_code"]).isna().mean()
                  .round(3).to_string())

    raw_files = sorted(p for p in RAW.rglob("*") if p.is_file())
    manifest = {
        "downloaded_at_utc": downloaded_at,
        "sources": {"omni": OMNI_URL, "omni_format": OMNI_FMT_URL,
                    "kyoto_hapi": f"{KYOTO_HAPI} dataset={KYOTO_DATASET}"},
        "kyoto_hapi": {k: hapi_info.get(k) for k in
                       ("HAPI", "modificationDate", "timeStampLocation", "resourceID", "title")},
        "dst_versions_per_split": versions,
        "time_alignment": alignment,
        "splits": SPLITS,
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "requests": requests.__version__},
        "sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
                   for p in raw_files + list(outputs.values())},
    }
    (ROOT / "data" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("\nWrote data/MANIFEST.json")


if __name__ == "__main__":
    main()
