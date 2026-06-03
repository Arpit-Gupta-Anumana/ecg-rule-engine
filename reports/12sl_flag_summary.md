# 12SL eval: measurements vs derived flags vs statement flags

ECGs evaluated: **21799**

## Flag sources

| Mode | Flag source |
|------|-------------|
| `measurements_only` | 122 measurement features only |
| `derived_flags` | Pass-1 rule fires → flags (same ECG, no statements) |
| `statement_flags` | 12SL `statements` column (input labels) |

## Derived flags that ever fire (8 / 13)

| Flag | Rule output source | % ECGs ON |
|------|-------------------|-----------|
| `rhythm_sinus_flag` | SinusRhythm | 89.756% |
| `lbbb_flag` | LBBB | 4.986% |
| `rbbb_flag` | RBBB | 3.028% |
| `lafb_flag` | Hemiblocks.LAFB | 2.491% |
| `irbbb_flag` | IncompleteBundleBlocks.IRBBB | 1.904% |
| `ilbbb_flag` | IncompleteBundleBlocks.ILBBB | 1.193% |
| `wpw_flag` | WPW | 0.229% |
| `rhythm_afib_flag` | AFib | 0.009% |

**Never derived from measurements-only rules:**

- `rhythm_aflutter_flag`
- `paced_flag`
- `lpfb_flag`
- `ivcb_flag`
- `rvh_flag`

## Top F1 improvements (derived vs measurements)

```
             disease  F1_measurements_only  F1_derived_flags  F1_statement_flags  F1_delta_derived_vs_meas
          PRInterval              0.000000          0.935522            0.998884                  0.935522
   AtrialEnlargement              0.000000          0.518519            0.535044                  0.518519
                 RVH              0.000000          0.086957            0.494382                  0.086957
        AcuteMISTEMI              0.138235          0.215457            0.216092                  0.077221
   STElevationInjury              0.062500          0.107075            0.112403                  0.044575
STDepressionIschemia              0.221404          0.251331            0.251841                  0.029927
       TWaveIschemia              0.368197          0.392523            0.405235                  0.024326
             QWaveMI              0.670241          0.681678            0.697425                  0.011437
                LBBB              0.659407          0.663403            0.695153                  0.003996
       LowVoltageQRS              0.766416          0.767241            0.767241                  0.000826
```