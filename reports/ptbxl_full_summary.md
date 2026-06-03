# PTB-XL Full Benchmark — All Rules

- Records evaluated: **21778** (skipped 21)
- Rules: **56** diseases, **133** variants
- Fold filter: ALL (1-10)

## Diseases WITH SCP Ground Truth

| Disease | Pos | Neg | TP | FP | FN | TN | Sens | Spec | PPV | NPV | F1 | ROC AUC | PR AUC | Prev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LVH | 2354 | 19424 | 975 | 1630 | 1379 | 17794 | 41.4% | 91.6% | 37.4% | 92.8% | 39.3% | 0.665 | 0.155 | 10.8% |
| STDepressionIschemia | 3002 | 18776 | 1453 | 3484 | 1549 | 15292 | 48.4% | 81.4% | 29.4% | 90.8% | 36.6% | 0.649 | 0.142 | 13.8% |
| RBBB | 537 | 21241 | 345 | 1861 | 192 | 19380 | 64.2% | 91.2% | 15.6% | 99.0% | 25.2% | 0.777 | 0.100 | 2.5% |
| PRInterval | 786 | 20992 | 353 | 2285 | 433 | 18707 | 44.9% | 89.1% | 13.4% | 97.7% | 20.6% | 0.670 | 0.060 | 3.6% |
| AtrialEnlargement | 506 | 21272 | 60 | 121 | 446 | 21151 | 11.9% | 99.4% | 33.1% | 97.9% | 17.5% | 0.556 | 0.039 | 2.3% |
| RVH | 126 | 21652 | 15 | 35 | 111 | 21617 | 11.9% | 99.8% | 30.0% | 99.5% | 17.0% | 0.559 | 0.036 | 0.6% |
| Hemiblocks | 2537 | 19241 | 1206 | 10732 | 1331 | 8509 | 47.5% | 44.2% | 10.1% | 86.5% | 16.7% | 0.459 | 0.048 | 11.6% |
| TWaveIschemia | 2594 | 19184 | 488 | 3185 | 2106 | 15999 | 18.8% | 83.4% | 13.3% | 88.4% | 15.6% | 0.511 | 0.025 | 11.9% |
| NonspecificIVCB | 785 | 20993 | 388 | 6769 | 397 | 14224 | 49.4% | 67.8% | 5.4% | 97.3% | 9.8% | 0.586 | 0.027 | 3.6% |
| IncompleteBundleBlocks | 1194 | 20584 | 85 | 689 | 1109 | 19895 | 7.1% | 96.7% | 11.0% | 94.7% | 8.6% | 0.519 | 0.008 | 5.5% |
| QWaveMI | 5283 | 16495 | 213 | 206 | 5070 | 16289 | 4.0% | 98.8% | 50.8% | 76.3% | 7.5% | 0.514 | 0.020 | 24.3% |
| LowVoltageQRS | 182 | 21596 | 109 | 2637 | 73 | 18959 | 59.9% | 87.8% | 4.0% | 99.6% | 7.4% | 0.738 | 0.024 | 0.8% |
| ProlongedQT | 117 | 21661 | 8 | 115 | 109 | 21546 | 6.8% | 99.5% | 6.5% | 99.5% | 6.7% | 0.532 | 0.004 | 0.5% |
| LBBB | 532 | 21246 | 103 | 3492 | 429 | 17754 | 19.4% | 83.6% | 2.9% | 97.6% | 5.0% | 0.515 | 0.006 | 2.4% |
| STElevationInjury | 351 | 21427 | 37 | 1812 | 314 | 19615 | 10.5% | 91.5% | 2.0% | 98.4% | 3.4% | 0.510 | 0.002 | 1.6% |
| WPW | 79 | 21699 | 2 | 769 | 77 | 20930 | 2.5% | 96.5% | 0.3% | 99.6% | 0.5% | 0.495 | 0.000 | 0.4% |
| AVBlock | 30 | 21748 | 0 | 538 | 30 | 21210 | 0.0% | 97.5% | 0.0% | 99.9% | 0.0% | 0.488 | 0.000 | 0.1% |
| BiventricularHypertrophy | 0 | 21778 | 0 | 1 | 0 | 21777 | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% | nan | nan | 0.0% |
| AFib | 1510 | 20268 | 0 | 0 | 1510 | 20268 | 0.0% | 100.0% | 0.0% | 93.1% | 0.0% | 0.500 | 0.000 | 6.9% |
| AtrialFlutter | 70 | 21708 | 0 | 0 | 70 | 21708 | 0.0% | 100.0% | 0.0% | 99.7% | 0.0% | 0.500 | 0.000 | 0.3% |
| Ectopy | 1484 | 20294 | 0 | 0 | 1484 | 20294 | 0.0% | 100.0% | 0.0% | 93.2% | 0.0% | 0.500 | 0.000 | 6.8% |
| Pacing | 285 | 21493 | 0 | 0 | 285 | 21493 | 0.0% | 100.0% | 0.0% | 98.7% | 0.0% | 0.500 | 0.000 | 1.3% |
| SinusArrhythmia | 771 | 21007 | 0 | 0 | 771 | 21007 | 0.0% | 100.0% | 0.0% | 96.5% | 0.0% | 0.500 | 0.000 | 3.5% |
| SinusRhythm | 18206 | 3572 | 0 | 0 | 18206 | 3572 | 0.0% | 100.0% | 0.0% | 16.4% | 0.0% | 0.500 | 0.000 | 83.6% |

## Diseases WITHOUT SCP Ground Truth (fire rate only)

| Disease | Fires | Fire rate |
|---|---:|---:|
| NonspecificTWave | 16081 | 73.84% |
| NonspecificSTDepression | 11403 | 52.36% |
| UndeterminedRhythm | 8717 | 40.03% |
| PoorRWaveProgression | 6691 | 30.72% |
| AbnormalQRSTAngle | 4662 | 21.41% |
| STElevationMechanismUnknown | 4126 | 18.95% |
| QRSAxis | 3640 | 16.71% |
| NonspecificSTElevation | 2105 | 9.67% |
| JunctionalRhythm | 1184 | 5.44% |
| PulmonaryDiseasePattern | 116 | 0.53% |
| AcuteMISTEMI | 67 | 0.31% |
| PediatricConduction | 39 | 0.18% |
| PedNonspecificBlock | 37 | 0.17% |
| PediatricTWave | 26 | 0.12% |
| PericarditisOrEarlyRepol | 25 | 0.11% |
| Brugada | 14 | 0.06% |
| PediatricRVH | 10 | 0.05% |
| PedSTDepression | 9 | 0.04% |
| PediatricLowVoltageQRS | 7 | 0.03% |
| PedEarlyRepolPericarditis | 3 | 0.01% |
| PediatricAtrialEnlargement | 2 | 0.01% |
| PediatricSTElevation | 2 | 0.01% |
| PediatricLVH | 1 | 0.00% |
| EctopicAtrialRhythm | 0 | 0.00% |
| ElectrodeReversals | 0 | 0.00% |
| PedBiventricularHypertrophy | 0 | 0.00% |
| PediatricBrugada | 0 | 0.00% |
| PediatricDextrocardia | 0 | 0.00% |
| PediatricQWaveMI | 0 | 0.00% |
| PediatricProlongedQT | 0 | 0.00% |
| PediatricQRSAxis | 0 | 0.00% |
| PediatricWPW | 0 | 0.00% |

## Variant-level fire rates (top 20 by fires)

| Disease | Variant | Fires | Fire rate |
|---|---|---:|---:|
| NonspecificTWave | NT_small_or_shallow_in_two_leads | 16081 | 73.84% |
| NonspecificSTDepression | NST_STJ_below_minus_0p05_two_leads_no_aVR | 11403 | 52.36% |
| Hemiblocks | IVCD_nonspecific | 11077 | 50.86% |
| UndeterminedRhythm | WideQRSRhythm | 8451 | 38.81% |
| NonspecificIVCB | IVCB_QRS_wide_no_BBB | 7157 | 32.86% |
| PoorRWaveProgression | PRWP_morphology | 6691 | 30.72% |
| STDepressionIschemia | Generalized_ST_Depression | 4686 | 21.52% |
| AbnormalQRSTAngle | AQRST_discordant_axes | 4662 | 21.41% |
| STDepressionIschemia | Junctional_ST_Depression | 4149 | 19.05% |
| STElevationMechanismUnknown | SERYR1_STJ_two_leads_over_0p10 | 4126 | 18.95% |
| LBBB | LBBB_standard | 3631 | 16.67% |
| TWaveIschemia | Lateral_T_Inversion | 2832 | 13.00% |
| STDepressionIschemia | Inferior_Subendocardial | 2726 | 12.52% |
| QRSAxis | Left_axis_deviation | 2701 | 12.40% |
| LVH | Cornell_product | 2565 | 11.78% |
| RBBB | Complete_RBBB | 2265 | 10.40% |
| NonspecificSTElevation | NST_ST_elevation_and_nontall_T | 2219 | 10.19% |
| STDepressionIschemia | Lateral_Subendocardial | 2088 | 9.59% |
| STDepressionIschemia | Anterior_Subendocardial | 2033 | 9.34% |
| LowVoltageQRS | LowV_all_limb | 1674 | 7.69% |