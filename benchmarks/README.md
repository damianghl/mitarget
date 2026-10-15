# Benchmarks

Two scripts reproduce the experiments of the original notebooks
(`mutual_information_project.ipynb` and `encoder_bakeoff.ipynb` in `code_new.zip`).
Both print a summary per scenario or task and, with `--out DIR`, write raw and
summary CSVs.

## Synthetic scenarios

```bash
python benchmarks/synthetic_scenarios.py --fast          # about 1.5 minutes
python benchmarks/synthetic_scenarios.py --out results/  # original settings
python benchmarks/synthetic_scenarios.py --scenarios 4 --reps 20 --no-crp
```

| Scenario | Truth | Oracle |
|---|---|---|
| 1 | single shared beta = 2.3, Pitman-Yor q_x | knows beta |
| 2 | two equal clusters, beta = 0.2 and 10 | knows betas and weights |
| 3 | the paper's spherical-symmetry example (289 distinct values) | none |
| 4 | parity: q(y=1\|x) in {1/2, q0, 1-q0}, q0 = 0.1 | knows q0, or fits it |

`mitarget.synthetic.scenario_junk_mixture` builds Scenario 4 directly as a
mixture with any junk fraction and q0 (drawn from U[0, 1] when not given).

Metrics: MI error `I_hat - I_true` and excess cross-entropy
`E_x[KL(q_true(y|x) || q_hat(y|x))]`, in nats. All methods and oracles are
scored on the same draws.

## Real data

The ARFF files are the ones in `code_new.zip/code/data/` (OpenML exports). They
are not committed; copy them into `data/` (git-ignored) or pass any folder:

```bash
python benchmarks/realdata_bakeoff.py --data-dir data/ --fast
python benchmarks/realdata_bakeoff.py --data-dir data/ --tasks employee_salaries --out results/
```

| Task | Feature | kx | Target |
|---|---|---|---|
| `employee_salaries` | employee_position_title | 385 | salary quartile (4 classes) |
| `churn_state` | state | 51 | churn |
| `churn_pseudo_id` | state x area_code x account_length | 4393 | churn |
| `airlines_airport` | AirportFrom | 283 | delay |
| `airlines_flight` | Flight | 5857 | delay |
| `click_advertiser` | advertiser_id | 6064 | click |
| `click_user` | user_id | 30114 | click |

Metric: held-out log-loss over stratified k-fold splits (plus AUC for binary
targets). It equals excess cross-entropy plus the constant H(Y|X), so rankings
within a task are comparable to the synthetic metric. CRP is skipped above
`--crp-max-kx` (default 8000, 500 with `--fast`) because its Gibbs sweep cost
grows linearly in kx.
