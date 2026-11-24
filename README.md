# DST-Cycle25

Do machine-learning forecasts of the Dst index add anything beyond persistence during intense storms of solar cycle 25, and do uncertainty-interval methods that aim to maintain coverage during storms keep it on a cycle they have not seen?

Models are trained on 1995–2009 (mainly solar cycle 23), conformal intervals are calibrated on 2010–2019 (cycle 24), and evaluation happens once, under a frozen protocol, on cycle 25 (2020–2025).

**Status:** protocol version 0.5 (draft, not frozen); no model has been trained yet or evaluated on test data.

## Protocol

The full study plan, with hypotheses, decision rules and a deviations log: [PROTOCOL_en.md](PROTOCOL_en.md) · [PROTOCOL_pl.md](PROTOCOL_pl.md) (Polish).

In short:

- **Models:** persistence, the O'Brien–McPherron empirical model, XGBoost on physical features and an LSTM; horizons of 1 h and 6 h.
- **Uncertainty:** split, Mondrian and adaptive (DtACI) conformal prediction at a nominal 90% level.
- **Hypotheses:** H1 – the ML advantage over persistence shrinks in intense storms; H2 – Mondrian coverage drops during storms of the new cycle; H3 – the adaptive method restores it.
- **Statistics:** the unit of analysis is a storm, not an hour; storm-level bootstrap, Holm correction and practical thresholds fixed in advance.

## Usage

```bash
pip install -r requirements.txt
python -m pytest              # tests of the download logic, no network needed
python scripts/download_data.py
```

The script downloads the data once and writes `data/MANIFEST.json` with the download date, the Dst version counts per split and SHA-256 checksums of every file. Test data go to `data/sealed/` and are not opened before the protocol is frozen.

## Data sources

- [NASA OMNI2, hourly data](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/)
- [Kyoto WDC, Dst index](https://wdc.kugi.kyoto-u.ac.jp/dstdir/): final version up to 2020, provisional from 2021
