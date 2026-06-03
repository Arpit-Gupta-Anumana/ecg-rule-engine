# PTB-XL rule-engine run

- records evaluated: **21778**  (skipped 21)
- rules: **19** diseases, **58** variants
- fold filter: ALL (1-10)

## Confusion matrix vs SCP-coded GT

| Disease | SCP GT codes | TP | FP | FN | TN | Prev | Sens | Spec | PPV | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AFib | `AFIB` | 1510 | 0 | 0 | 20268 | 6.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| AtrialFlutter | `AFLT` | 70 | 0 | 0 | 21708 | 0.3% | 100.0% | 100.0% | 100.0% | 100.0% |
| LVH | `LVH,VCLVH` | 975 | 1630 | 1379 | 17794 | 10.8% | 41.4% | 91.6% | 37.4% | 39.3% |
| STDepressionIschemia | `ISCAL,ISCAN,ISCAS,ISCIL,ISCIN,ISCLA,ISC_,STD_` | 1453 | 3484 | 1549 | 15292 | 13.8% | 48.4% | 81.4% | 29.4% | 36.6% |
| RBBB | `CRBBB` | 345 | 1861 | 192 | 19380 | 2.5% | 64.2% | 91.2% | 15.6% | 25.2% |
| AtrialEnlargement | `LAO/LAE,RAO/RAE` | 60 | 121 | 446 | 21151 | 2.3% | 11.9% | 99.4% | 33.1% | 17.5% |
| RVH | `RVH` | 15 | 35 | 111 | 21617 | 0.6% | 11.9% | 99.8% | 30.0% | 17.0% |
| Hemiblocks | `IVCD,LAFB,LPFB` | 1206 | 10732 | 1331 | 8509 | 11.6% | 47.5% | 44.2% | 10.1% | 16.7% |
| TWaveIschemia | `INVT,LOWT,NDT,NT_,TAB_` | 488 | 3185 | 2106 | 15999 | 11.9% | 18.8% | 83.4% | 13.3% | 15.6% |
| WPW | `WPW` | 40 | 572 | 39 | 21127 | 0.4% | 50.6% | 97.4% | 6.5% | 11.6% |
| IncompleteBundleBlocks | `ILBBB,IRBBB` | 85 | 689 | 1109 | 19895 | 5.5% | 7.1% | 96.7% | 11.0% | 8.6% |
| QWaveMI | `ALMI,AMI,ASMI,ILMI,IMI,IPLMI,IPMI,LMI,PMI` | 213 | 206 | 5070 | 16289 | 24.3% | 4.0% | 98.8% | 50.8% | 7.5% |
| LowVoltageQRS | `LVOLT` | 109 | 2637 | 73 | 18959 | 0.8% | 59.9% | 87.8% | 4.0% | 7.4% |
| ProlongedQT | `LNGQT` | 8 | 115 | 109 | 21546 | 0.5% | 6.8% | 99.5% | 6.5% | 6.7% |
| LBBB | `CLBBB` | 103 | 3492 | 429 | 17754 | 2.4% | 19.4% | 83.6% | 2.9% | 5.0% |
| STElevationInjury | `INJAL,INJAS,INJIL,INJIN,INJLA,STE_` | 37 | 1812 | 314 | 19615 | 1.6% | 10.5% | 91.5% | 2.0% | 3.4% |
| QRSAxis | `(none)` | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Brugada | `(none)` | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| PericarditisOrEarlyRepol | `(none)` | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Variant fire counts (top 10 per disease)

| Disease | Variant | Fires | FP-on-NORM | Fire rate |
|---|---|---:|---:|---:|
| AFib | AFib_Test1_Irregular | 1510 | 37 | 6.93% |
| AFib | AFib_Test2_AtrialRate | 0 | 0 | 0.00% |
| AtrialEnlargement | RAE | 183 | 33 | 0.84% |
| AtrialEnlargement | LAE_literature_proxy | 0 | 0 | 0.00% |
| AtrialFlutter | AFL_via_ge_flag | 70 | 0 | 0.32% |
| AtrialFlutter | AFL_atrial_rate_adult | 0 | 0 | 0.00% |
| AtrialFlutter | AFL_atrial_rate_pediatric | 0 | 0 | 0.00% |
| Brugada | Brugada_type1_V2 | 13 | 0 | 0.06% |
| Brugada | Brugada_type1_V1 | 9 | 0 | 0.04% |
| Hemiblocks | IVCD_nonspecific | 11077 | 5505 | 50.86% |
| Hemiblocks | LAFB | 985 | 79 | 4.52% |
| Hemiblocks | LPFB | 101 | 12 | 0.46% |
| IncompleteBundleBlocks | ILBBB | 386 | 115 | 1.77% |
| IncompleteBundleBlocks | IRBBB | 359 | 100 | 1.65% |
| IncompleteBundleBlocks | RSR_pattern_V1 | 47 | 0 | 0.22% |
| LBBB | LBBB_standard | 3631 | 1526 | 16.67% |
| LVH | Cornell_product | 2565 | 395 | 11.78% |
| LVH | R_in_aVL | 591 | 18 | 2.71% |
| LVH | Sokolow_Lyon | 549 | 28 | 2.52% |
| LVH | Romhilt_Estes | 104 | 1 | 0.48% |
| LowVoltageQRS | LowV_all_limb | 1674 | 625 | 7.69% |
| LowVoltageQRS | LowV_all_12 | 1646 | 680 | 7.56% |
| PericarditisOrEarlyRepol | Acute_Pericarditis | 22 | 3 | 0.10% |
| PericarditisOrEarlyRepol | Early_Repolarization | 17 | 4 | 0.08% |
| ProlongedQT | QTc_480_to_500 | 146 | 9 | 0.67% |
| ProlongedQT | QTc_ge_500 | 120 | 3 | 0.55% |
| QRSAxis | Left_axis_deviation | 2701 | 275 | 12.40% |
| QRSAxis | Rightward_axis | 428 | 197 | 1.97% |
| QRSAxis | Right_axis_deviation | 268 | 40 | 1.23% |
| QRSAxis | Right_superior_axis_deviation | 240 | 19 | 1.10% |
| QRSAxis | Indeterminate_axis | 45 | 18 | 0.21% |
| QWaveMI | Posterior_MI | 247 | 71 | 1.13% |
| QWaveMI | Septal_MI | 152 | 3 | 0.70% |
| QWaveMI | Anterior_MI | 104 | 0 | 0.48% |
| QWaveMI | Inferior_MI | 32 | 2 | 0.15% |
| QWaveMI | Lateral_MI | 3 | 0 | 0.01% |
| RBBB | Complete_RBBB | 2265 | 562 | 10.40% |
| RBBB | RBBB_with_RVH | 2 | 0 | 0.01% |
| RVH | Possible_RVH_voltage_axis | 57 | 1 | 0.26% |
| RVH | RVH_voltage_axis | 11 | 0 | 0.05% |
| RVH | RVH_with_RBBB | 3 | 0 | 0.01% |
| STDepressionIschemia | Generalized_ST_Depression | 4686 | 692 | 21.52% |
| STDepressionIschemia | Junctional_ST_Depression | 4149 | 701 | 19.05% |
| STDepressionIschemia | Inferior_Subendocardial | 2726 | 483 | 12.52% |
| STDepressionIschemia | Lateral_Subendocardial | 2088 | 232 | 9.59% |
| STDepressionIschemia | Anterior_Subendocardial | 2033 | 254 | 9.34% |
| STDepressionIschemia | Septal_Subendocardial | 1342 | 216 | 6.16% |
| STElevationInjury | Anterior_STEMI | 1591 | 228 | 7.31% |
| STElevationInjury | Lateral_STEMI | 688 | 26 | 3.16% |
| STElevationInjury | Inferior_STEMI | 343 | 44 | 1.57% |
| TWaveIschemia | Lateral_T_Inversion | 2832 | 27 | 13.00% |
| TWaveIschemia | Inferior_T_Inversion | 1573 | 144 | 7.22% |
| TWaveIschemia | Anterior_T_Inversion | 1393 | 44 | 6.40% |
| TWaveIschemia | Anterior_T_Inversion_Marked | 513 | 19 | 2.36% |
| TWaveIschemia | Lateral_T_Inversion_Marked | 169 | 0 | 0.78% |
| TWaveIschemia | Inferior_T_Inversion_Marked | 147 | 1 | 0.67% |
| WPW | WPW_PR_proxy | 771 | 243 | 3.54% |
| WPW | WPW_via_ge_flag | 41 | 0 | 0.19% |