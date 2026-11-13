# Screening metrics study, 2026-11-12

Simulations behind `roc_auc`, `bedroc`, `topn_recovery` and the permutation p. They establish general statistical facts that the
estimator pages quote. They are not unit tests of claimcheck and take about two minutes, so they are kept out of `tests/`:

    python -m pytest -rP docs/studies/test_screening_metrics_study.py

run from the repository root with numpy, scipy and RDKit installed. Every test asserts the number it found and prints its table.
The tables below are the output of the run of 2026-11-12 (14 tests). The planned design they refer to is 7 actives among
357 compounds per target, alpha = 20.

Terms. N compounds, n actives, Ra = n / N. Position 1 is the best-scored compound. "Screen" is one simulated ranking of one
library. The sd columns are sampling standard deviations across simulated screens. Every generator is seeded.

## 1. What alpha = 20 weights

The share of BEDROC's weight in the first fraction z of the list is (1 - exp(-alpha z)) / (1 - exp(-alpha)). At alpha = 20 and
z = 0.08 it is 0.798103. Exactly 80% is reached at 8.047% of the list. With discrete positions: the first 80 of 1,000 carry
0.798103, the first 28 of 357 (7.84% of the list) carry 0.791669.

## 2. BEDROC does not saturate

A perfect ranking gives exactly 1 and the worst exactly 0 at every active fraction tested, by the paper's closed form, by the
min-max form and by RDKit's `CalcBEDROC`. What moves with the active fraction is the ceiling of RIE and the value a random ranking
scores.

| n of N | alpha * Ra | BEDROC, perfect | BEDROC, worst | RIE ceiling | BEDROC of a random ranking |
|---|---|---|---|---|---|
| 7 of 357 | 0.392 | 1.000000 | 0.000000 | 16.544 | 0.0604 |
| 10 of 1,000 | 0.200 | 1.000000 | 0.000000 | 18.127 | 0.0552 |
| 36 of 357 | 2.017 | 1.000000 | 0.000000 | 8.597 | 0.1163 |
| 100 of 357 | 5.602 | 1.000000 | 0.000000 | 3.557 | 0.2811 |
| 178 of 357 | 9.972 | 1.000000 | 0.000000 | 2.006 | 0.4986 |
| 300 of 357 | 16.807 | 1.000000 | 0.000000 | 1.190 | 0.8335 |

At 7 of 357 the best ranking that is not perfect (positions 1 to 6 and 8) scores 0.99346.

## 3. The sd of BEDROC and the 1/sqrt(8n) planning figure

Truchon and Bayly (doi 10.1021/ci600426e) give 1/sqrt(8n), n the number of actives, as an empirically fitted maximum of the sd of
BEDROC over simulations in which active positions follow an exponential distribution with parameter lambda (N = 25,000). The paper
was read here through a summarising fetch of the publisher's page, not the PDF, and its maxima of 0.11, 0.05, 0.035 and 0.025 at
n = 10, 50, 100 and 200 match the formula; that agreement checks the fetched text against itself. At n = 7 the figure is 0.1336.

The same model at the planned design, 20,000 screens per lambda:

| lambda | mean BEDROC | sd BEDROC | mean AUC | sd AUC |
|---|---|---|---|---|
| 0.001 | 0.0616 | 0.0685 | 0.5019 | 0.1104 |
| 2 | 0.1260 | 0.0939 | 0.6576 | 0.1006 |
| 5 | 0.2391 | 0.1173 | 0.8100 | 0.0697 |
| 10 | 0.3945 | 0.1269 | 0.9058 | 0.0377 |
| 12.4 | 0.4502 | 0.1271 | 0.9251 | 0.0307 |
| 15 | 0.5038 | 0.1248 | 0.9397 | 0.0251 |
| 20 | 0.5851 | 0.1198 | 0.9567 | 0.0188 |
| 30 | 0.6981 | 0.1051 | 0.9734 | 0.0123 |
| 40 | 0.7704 | 0.0924 | 0.9817 | 0.0091 |
| 60 | 0.8598 | 0.0705 | 0.9899 | 0.0058 |
| 100 | 0.9383 | 0.0428 | 0.9959 | 0.0031 |

The largest sd is 0.1271, below 0.1336, at lambda about 12 (the fetched text places the maximum at alpha about equal to lambda).

It is not a bound. A screener that places each active at the very top or the very bottom with probability one half has sd 0.1894
at n = 7, which is 1/sqrt(4n) = 0.1890, the bound for a mean of n independent terms in [0, 1].

## 4. Bootstrap interval widths of AUC and BEDROC at 7 actives

7 actives among 357, 120 screens per row, 300 bootstrap draws each over compounds (unstratified, draws with no active dropped, as
`bootstrap_ci` does). Widths are medians of the 95% percentile interval. Two models: binormal scores (decoys N(0, 1), actives
N(d, 1)), and active positions drawn with probability proportional to exp(-lambda k / N). Chance is 0.5 for AUC and 0.0604 for
BEDROC, so the two widths are on different scales.

| model | mean AUC | mean BEDROC | sd AUC | sd BEDROC | width AUC | width BEDROC | share of screens with BEDROC wider |
|---|---|---|---|---|---|---|---|
| binormal d = 0 | 0.516 | 0.064 | 0.118 | 0.067 | 0.430 | 0.126 | 0.10 |
| binormal d = 0.5 | 0.640 | 0.149 | 0.103 | 0.117 | 0.417 | 0.339 | 0.38 |
| binormal d = 0.954 | 0.760 | 0.262 | 0.095 | 0.133 | 0.324 | 0.519 | 0.77 |
| binormal d = 1.5 | 0.846 | 0.401 | 0.069 | 0.143 | 0.245 | 0.581 | 0.96 |
| binormal d = 1.812 | 0.901 | 0.530 | 0.056 | 0.156 | 0.177 | 0.591 | 0.99 |
| binormal d = 2.5 | 0.962 | 0.758 | 0.036 | 0.126 | 0.060 | 0.466 | 1.00 |
| binormal d = 3.5 | 0.994 | 0.938 | 0.009 | 0.057 | 0.010 | 0.155 | 1.00 |
| exponential lambda = 0.001 | 0.502 | 0.065 | 0.112 | 0.068 | 0.450 | 0.112 | 0.11 |
| exponential lambda = 5 | 0.812 | 0.217 | 0.064 | 0.109 | 0.227 | 0.431 | 0.76 |
| exponential lambda = 10 | 0.908 | 0.405 | 0.035 | 0.119 | 0.123 | 0.525 | 1.00 |
| exponential lambda = 20 | 0.957 | 0.568 | 0.016 | 0.108 | 0.062 | 0.478 | 1.00 |
| exponential lambda = 40 | 0.981 | 0.764 | 0.010 | 0.096 | 0.033 | 0.399 | 1.00 |
| exponential lambda = 100 | 0.996 | 0.935 | 0.004 | 0.050 | 0.012 | 0.180 | 1.00 |

BEDROC has the wider interval for a screener with skill (mean AUC 0.76 and above, every row) and the narrower one at or near
chance, where AUC's is three to four times wider. The ordering changes between mean AUC 0.64 and 0.76. The AUC width of 0.010 at
mean AUC 0.994 is the percentile bootstrap collapsing against the boundary with 7 actives.

## 5. Clopper-Pearson when the actives compete for the shortlist

`topn_recovery` reports the Clopper-Pearson interval on k of K actives in the first N of M. The estimand, written before the run:
the expected share of actives a scorer places in the first N, for actives drawn like these (binormal, shift d), decoys redrawn
every screen. The figure that would have withheld the interval: coverage below 0.93 in any design. 20,000 screens per row.

| K | M | N | d | expected share | coverage of the 95% interval | var(k) / binomial variance |
|---|---|---|---|---|---|---|
| 7 | 357 | 50 | 0 | 0.1389 | 0.9922 | 0.971 |
| 7 | 357 | 50 | 0.5 | 0.2753 | 0.9829 | 0.969 |
| 7 | 357 | 50 | 0.954 | 0.4404 | 0.9808 | 0.962 |
| 7 | 357 | 50 | 1.5 | 0.6469 | 0.9919 | 0.965 |
| 7 | 357 | 50 | 2.5 | 0.9109 | 0.9815 | 0.981 |
| 7 | 357 | 18 | 0 | 0.0505 | 0.9967 | 0.984 |
| 7 | 357 | 18 | 0.5 | 0.1248 | 0.9951 | 0.963 |
| 7 | 357 | 18 | 0.954 | 0.2346 | 0.9930 | 0.940 |
| 7 | 357 | 18 | 1.5 | 0.4143 | 0.9637 | 0.904 |
| 7 | 357 | 18 | 2.5 | 0.7603 | 0.9916 | 0.928 |
| 30 | 1,500 | 75 | 0 | 0.0499 | 0.9860 | 0.984 |
| 30 | 1,500 | 75 | 0.5 | 0.1225 | 0.9590 | 0.951 |
| 30 | 1,500 | 75 | 0.954 | 0.2337 | 0.9760 | 0.930 |
| 30 | 1,500 | 75 | 1.5 | 0.4129 | 0.9822 | 0.912 |
| 30 | 1,500 | 75 | 2.5 | 0.7568 | 0.9786 | 0.913 |
| 40 | 200 | 50 | 0 | 0.2497 | 0.9846 | 0.812 |
| 40 | 200 | 50 | 0.5 | 0.3874 | 0.9852 | 0.749 |
| 40 | 200 | 50 | 0.954 | 0.5194 | 0.9842 | 0.721 |
| 40 | 200 | 50 | 1.5 | 0.6708 | 0.9831 | 0.714 |
| 40 | 200 | 50 | 2.5 | 0.8805 | 0.9904 | 0.764 |
| 20 | 100 | 10 | 0 | 0.1005 | 0.9967 | 0.793 |
| 20 | 100 | 10 | 0.5 | 0.1808 | 0.9899 | 0.684 |
| 20 | 100 | 10 | 0.954 | 0.2678 | 0.9953 | 0.557 |
| 20 | 100 | 10 | 1.5 | 0.3669 | 0.9998 | 0.369 |
| 20 | 100 | 10 | 2.5 | 0.4772 | 1.0000 | 0.086 |

Coverage is 0.959 to 1.000 and never below 0.95; the count is less variable than a binomial in every design. The interval is
conservative and is reported. The binomial tail at the chance share against the exact hypergeometric tail, for 7 actives and the
first 50 of 357, is in `docs/topn_recovery.md`.

## 6. Which methods lost exact ties in the permutation p

Every order of y on 7 rows, each taken as the observed one against every other, decided in rational arithmetic. Counted: the
comparisons that `abs(r - centre) >= abs(obs - centre)` in floating point gets wrong, and that `relate.as_extreme` gets wrong.

| design | comparisons | plain float wrong | `as_extreme` wrong |
|---|---|---|---|
| pearson, integer data | 16,129 | 82 | 0 |
| pearson, one-decimal data | 1,106,704 | 1,259 | 0 |
| spearman, untied | 3,249 | 0 | 0 |
| spearman, tied | 6,400 | 0 | 0 |
| spearman, tied, 3 levels | 1,024 | 0 | 0 |
| roc_auc, 3 against 4 | 169 | 3 | 0 |
| roc_auc, 3 against 4, tied | 100 | 1 | 0 |

`roc_auc` (centre 0.5) and `pearson` (centre 0) were affected; `spearman` was exact on these designs, and so was `kendall`, whose
numerator is an integer (checked in `tests/test_permutation_p.py`). On continuous data two different permutations almost never give the
same |r|, so the Pearson cases arise on discrete or rounded data. Every miss made p smaller than its exact value.

The size of the error in the `roc_auc` p before the fix, from the same permutation draws with the plain comparison and with
`as_extreme`: with 3 actives against 3 decoys and AUC 1/9, where the exact two-sided p is 0.2000, the plain comparison gave 0.1514
and `as_extreme` 0.2019 (20,000 draws). With 7 actives among 357 and 300 permutations, a score and its negation, whose exact p
values are equal, differed in 5 of 40 uninformative screens under the plain comparison, by at most 0.00664 (two permutation
draws), and in 0 of 40 with `as_extreme`.

## 7. The donor rule of the permutation null

`permutation_p` fills each row of a recipient unit with a draw, with replacement, from the donor unit's y values. Its docstring
said "with replacement when the two differ in size". The alternative is a within-block permutation when the sizes are equal. With
one row per unit the two are the same exact permutation. For units of several rows, the rejection rate at 0.05 under a null with
unit-level effects in both columns (sd 1, plus row noise), Spearman, 600 datasets per design, 200 permutations each:

| units x rows | row noise sd | as coded | within-block permutation | mean absolute difference in p |
|---|---|---|---|---|
| 15 x 4 | 0.5 | 0.055 | 0.055 | 0.031 |
| 8 x 6 | 0.5 | 0.047 | 0.055 | 0.031 |
| 15 x 4 | 0.05 | 0.045 | 0.043 | 0.030 |
| 30 x 2 | 0.5 | 0.037 | 0.045 | 0.020 |

The standard error of each rate is 0.009. Neither rule is distinguishable from 0.05 or from the other, and the difference in p
between them is the size of the Monte Carlo error of 200 permutations. The code was left as it is and the docstring corrected:
changing the code would have moved the random stream of every clustered p for no measured gain.

## 8. The percentile indices of the bootstrap interval

`bootstrap_ci` reads the sorted draws at index int(0.025 B) and int(0.975 B): 50 and 1950 of 2,000, so 50 draws lie below the
lower bound and 49 above the upper. Linear interpolation (numpy's default) would read positions 49.975 and 1949.025, so both
bounds sit within one order statistic of that convention, each on its upper side. This is not an error and was not changed: a
change would move every published interval by one order statistic for no gain in accuracy.
