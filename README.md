# DST-Cycle25

[![DOI](https://zenodo.org/badge/1470490178.svg)](https://doi.org/10.5281/zenodo.24511113)

Do machine-learning forecasts of the Dst index add anything beyond persistence during intense storms of solar cycle 25, and do uncertainty-interval methods that aim to maintain coverage during storms keep it on a cycle they have not seen?

Models are trained on 1995–2009 (mainly solar cycle 23), conformal intervals are calibrated on 2010–2019 (cycle 24), and evaluation happens once, under a frozen protocol, on cycle 25 (2020–2025).

**Status:** protocol v1.1, frozen on 2026-11-24 before the first training run and archived on Zenodo (all versions: [10.5281/zenodo.24511113](https://doi.org/10.5281/zenodo.24511113)); no model has been evaluated on test data. Changes since v1.0 are listed in the deviations log (protocol section 13).

## Protocol

The full study plan, with hypotheses, decision rules and a deviations log: [PROTOCOL_en.md](PROTOCOL_en.md) · [PROTOCOL_pl.md](PROTOCOL_pl.md) (Polish).

In short:

- **Models:** persistence, the O'Brien–McPherron empirical model, XGBoost on physical features and an LSTM; horizons of 1 h and 6 h.
- **Uncertainty:** split, Mondrian and adaptive (DtACI) conformal prediction at a nominal 90% level.
- **Hypotheses:** H1 – the ML advantage over persistence shrinks in intense storms; H2 – Mondrian coverage in intense storms drops from cycle 24 to cycle 25; H3 – the adaptive method restores it.
- **Statistics:** the unit of analysis is a storm, not an hour; storm-level bootstrap, Holm correction and practical thresholds fixed in advance.

## Usage

```bash
pip install -r requirements.txt
python -m pytest              # tests on synthetic data, no network needed
python scripts/download_data.py
python scripts/count_storms.py
```

The script downloads the data once and writes `data/MANIFEST.json` with the download date, the Dst version counts per split and SHA-256 checksums of every file. Test data go to `data/sealed/` and are not opened before the final evaluation. `count_storms.py` counts storm windows per split (protocol section 6) from Dst alone.

## Data sources

- [NASA OMNI2, hourly data](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/)
- [Kyoto WDC, Dst index](https://wdc.kugi.kyoto-u.ac.jp/dstdir/): final version up to 2020, provisional from 2021

The data are not part of this repository and remain subject to their providers' terms. The Kyoto Dst index is for non-commercial use and should be cited as [doi:10.17593/14515-74000](https://doi.org/10.17593/14515-74000); OMNI data should be acknowledged as described by [NASA SPDF](https://omniweb.gsfc.nasa.gov/).

## License

Code and documentation in this repository: [MIT](LICENSE).
