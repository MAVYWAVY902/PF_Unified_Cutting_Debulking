# Stage 2.3.5 sharpness sensitivity sweep results

## Protocol
- Sharpness values: [14.0, 17.0, 20.0, 23.0, 26.0] (default 20)
- Checkpoints: ['200k', '350k', '500k']
- N episodes per condition: see CSVs

## Per-checkpoint deltas (max-min across 5 sharpness values)

| ckpt | metric | min | max | Δ |
|---|---|---|---|---|
| 200k | interior_mean | 0.0073 | 0.0180 | 1.07% |
| 200k | cap_mean | 0.0231 | 0.0321 | 0.90% |
| 200k | catastrophic_rate | 0.0000 | 0.0000 | 0.00% |
| 350k | interior_mean | 0.0058 | 0.0400 | 3.42% |
| 350k | cap_mean | 0.0206 | 0.0867 | 6.61% |
| 350k | catastrophic_rate | 0.0000 | 0.0000 | 0.00% |
| 500k | interior_mean | 0.0055 | 0.0619 | 5.64% |
| 500k | cap_mean | 0.0200 | 0.1047 | 8.47% |
| 500k | catastrophic_rate | 0.0000 | 0.0000 | 0.00% |

## Verdict
Max Δ = 8.47%. **SENSITIVE** — defer to v4 (lean toward F/T purchase).
