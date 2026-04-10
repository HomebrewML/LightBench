# Benchmark Results
Generated: 2026-04-06 18:14:10.572215
Last updated: 2026-04-06 18:14:10.572230

## Summary (In Progress)

| Storage | Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|---------|-----------|---|---|---------|----------|------|
| fp32 | ADOPT | No | No | 0/6 | 0.00s | 0.0 |
| fp32 | AdEMAMix | No | No | 3/12 | 99.49s | 9.0 |
| fp32 | AdamC | No | No | 8/12 | 149.89s | 10.9 |
| fp32 | AdamW | No | No | 6/6 | 61.71s | 7.0 |
| fp32 | HyperBallAdamW | No | No | 10/12 | 108.34s | 8.6 |
| fp32 | LATHER | No | No | 3/18 | 135.42s | 6.0 |
| fp32 | LaPropOrtho | No | No | 0/6 | 0.00s | 0.0 |
| fp32 | MSAMLaProp | No | No | 1/12 | 252.34s | 44.0 |
| fp32 | Muon | No | No | 2/6 | 200.11s | 4.0 |
| fp32 | MuonLaProp | No | No | 6/18 | 164.43s | 12.0 |
| fp32 | NAdam | No | No | 0/12 | 0.00s | 0.0 |
| fp32 | OrthoLaProp | No | No | 0/6 | 0.00s | 0.0 |
| fp32 | PSGDKron | No | No | 11/36 | 101.69s | 4.9 |
| fp32 | PSGDLRA | No | No | 7/12 | 191.02s | 3.0 |
| fp32 | PSGDPRO | No | No | 0/6 | 0.00s | 0.0 |
| fp32 | RMSprop | No | No | 12/12 | 82.49s | 2.7 |
| fp32 | SAMWrapper | No | No | 0/18 | 0.00s | 0.0 |
| fp32 | SFAdamW | No | No | 6/12 | 199.65s | 9.2 |
| fp32 | SGD | No | No | 4/24 | 60.47s | 5.0 |
| fp32 | SOAP | No | No | 0/6 | 0.00s | 0.0 |
| fp32 | SOAPAdEMAMix | No | No | 2/6 | 186.18s | 4.0 |
| fp32 | SOAPNAdam | No | No | 0/12 | 0.00s | 0.0 |
| fp32 | SOLP | No | No | 1/12 | 154.50s | 13.0 |
| fp32 | Scion | No | No | 1/12 | 273.45s | 9.0 |
| fp32 | SignLaProp | No | No | 1/6 | 208.18s | 2.0 |
| fp32 | UnscaledAdamW | No | No | 1/12 | 220.18s | 4.0 |
| bf16 | ADOPT | No | No | 2/6 | 41.88s | 3.0 |
| bf16 | AdEMAMix | No | No | 3/24 | 153.51s | 4.7 |
| bf16 | AdamC | No | No | 0/6 | 0.00s | 0.0 |
| bf16 | AdamW | No | No | 1/12 | 159.98s | 11.0 |
| bf16 | LATHER | No | No | 0/6 | 0.00s | 0.0 |
| bf16 | LaPropOrtho | No | No | 0/6 | 0.00s | 0.0 |
| bf16 | MSAMLaProp | No | No | 0/6 | 0.00s | 0.0 |
| bf16 | Muon | No | No | 2/12 | 52.45s | 3.0 |
| bf16 | MuonAdamW | No | No | 0/6 | 0.00s | 0.0 |
| bf16 | MuonLaProp | No | No | 0/12 | 0.00s | 0.0 |
| bf16 | NAdam | No | No | 0/12 | 0.00s | 0.0 |
| bf16 | OrthoLaProp | No | No | 0/6 | 0.00s | 0.0 |
| bf16 | PSGDKron | No | No | 2/6 | 260.68s | 3.0 |
| bf16 | PSGDLRA | No | No | 4/18 | 152.17s | 6.2 |
| bf16 | PSGDPRO | No | No | 5/12 | 96.06s | 7.8 |
| bf16 | RMSprop | No | No | 6/12 | 33.74s | 3.5 |
| bf16 | SAMWrapper | No | No | 0/18 | 0.00s | 0.0 |
| bf16 | SFAdamW | No | No | 2/12 | 208.27s | 9.0 |
| bf16 | SGD | No | No | 7/12 | 52.58s | 3.6 |
| bf16 | SOAP | No | No | 0/6 | 0.00s | 0.0 |
| bf16 | SOAPAdEMAMix | No | No | 0/12 | 0.00s | 0.0 |
| bf16 | SOLP | No | No | 6/18 | 224.45s | 3.5 |
| bf16 | SUDSAdamW | No | No | 3/6 | 116.31s | 6.0 |
| bf16 | Scion | No | No | 13/18 | 2.54s | 1.0 |
| bf16 | SignLaProp | No | No | 2/12 | 252.57s | 17.5 |
| bf16+ecc | ADOPT | No | No | 0/6 | 0.00s | 0.0 |
| bf16+ecc | AdEMAMix | No | No | 2/6 | 91.61s | 10.5 |
| bf16+ecc | AdamC | No | No | 5/18 | 130.01s | 8.8 |
| bf16+ecc | AdamW | No | No | 1/30 | 141.64s | 7.0 |
| bf16+ecc | HyperBallAdamW | No | No | 0/6 | 0.00s | 0.0 |
| bf16+ecc | LATHER | No | No | 1/12 | 70.95s | 3.0 |
| bf16+ecc | LaProp | No | No | 3/24 | 153.26s | 6.3 |
| bf16+ecc | LaPropOrtho | No | No | 0/12 | 0.00s | 0.0 |
| bf16+ecc | MSAMLaProp | No | No | 6/6 | 93.31s | 7.3 |
| bf16+ecc | Muon | No | No | 0/6 | 0.00s | 0.0 |
| bf16+ecc | MuonLaProp | No | No | 3/12 | 195.35s | 10.3 |
| bf16+ecc | NAdam | No | No | 0/6 | 0.00s | 0.0 |
| bf16+ecc | OrthoLaProp | No | No | 3/12 | 180.94s | 4.0 |
| bf16+ecc | PSGDKron | No | No | 0/18 | 0.00s | 0.0 |
| bf16+ecc | PSGDLRA | No | No | 0/12 | 0.00s | 0.0 |
| bf16+ecc | PSGDPRO | No | No | 2/12 | 76.49s | 3.0 |
| bf16+ecc | SAMWrapper | No | No | 3/18 | 10.98s | 1.0 |
| bf16+ecc | SGD | No | No | 6/18 | 110.75s | 12.0 |
| bf16+ecc | SOAP | No | No | 5/12 | 262.00s | 3.0 |
| bf16+ecc | SOAPAdEMAMix | No | No | 0/6 | 0.00s | 0.0 |
| bf16+ecc | SOAPNAdam | No | No | 0/6 | 0.00s | 0.0 |
| bf16+ecc | SOLP | No | No | 0/6 | 0.00s | 0.0 |
| bf16+ecc | SUDSAdamW | No | No | 3/24 | 237.86s | 11.3 |
| bf16+ecc | SignLaProp | No | No | 1/6 | 224.06s | 4.0 |
| fp16 | ADOPT | No | No | 1/6 | 199.06s | 11.0 |
| fp16 | AdEMAMix | No | No | 0/6 | 0.00s | 0.0 |
| fp16 | AdamC | No | No | 0/6 | 0.00s | 0.0 |
| fp16 | LATHER | No | No | 4/16 | 35.82s | 3.0 |
| fp16 | LaProp | No | No | 0/6 | 0.00s | 0.0 |
| fp16 | LaPropOrtho | No | No | 0/6 | 0.00s | 0.0 |
| fp16 | MSAMLaProp | No | No | 5/15 | 187.12s | 8.6 |
| fp16 | Muon | No | No | 1/1 | 166.27s | 2.0 |
| fp16 | MuonAdamW | No | No | 7/12 | 119.47s | 13.6 |
| fp16 | MuonLaProp | No | No | 0/6 | 0.00s | 0.0 |
| fp16 | PSGDKron | No | No | 2/12 | 125.14s | 5.0 |
| fp16 | PSGDLRA | No | No | 0/6 | 0.00s | 0.0 |
| fp16 | PSGDPRO | No | No | 0/6 | 0.00s | 0.0 |
| fp16 | RMSprop | No | No | 2/18 | 71.19s | 15.5 |
| fp16 | SAMWrapper | No | No | 0/12 | 0.00s | 0.0 |
| fp16 | SFAdamW | No | No | 1/12 | 165.04s | 9.0 |
| fp16 | SGD | No | No | 4/6 | 68.18s | 7.0 |
| fp16 | SOAP | No | No | 1/6 | 233.84s | 19.0 |
| fp16 | SOLP | No | No | 1/6 | 200.03s | 4.0 |
| fp16 | SUDSAdamW | No | No | 0/12 | 0.00s | 0.0 |
| fp16 | Scion | No | No | 3/6 | 3.48s | 1.0 |
| fp16 | SignLaProp | No | No | 3/24 | 216.68s | 15.3 |
| fp16 | UnscaledAdamW | No | No | 0/12 | 0.00s | 0.0 |
| fp16+ecc | AdEMAMix | No | No | 0/6 | 0.00s | 0.0 |
| fp16+ecc | AdamC | No | No | 0/6 | 0.00s | 0.0 |
| fp16+ecc | AdamW | No | No | 1/6 | 167.88s | 11.0 |
| fp16+ecc | HyperBallAdamW | No | No | 4/12 | 240.18s | 7.8 |
| fp16+ecc | LATHER | No | No | 5/12 | 52.71s | 3.8 |
| fp16+ecc | LaProp | No | No | 3/24 | 110.85s | 3.0 |
| fp16+ecc | LaPropOrtho | No | No | 0/6 | 0.00s | 0.0 |
| fp16+ecc | MuonLaProp | No | No | 0/18 | 0.00s | 0.0 |
| fp16+ecc | NAdam | No | No | 7/12 | 153.73s | 7.4 |
| fp16+ecc | PSGDKron | No | No | 0/18 | 0.00s | 0.0 |
| fp16+ecc | PSGDPRO | No | No | 1/12 | 73.92s | 3.0 |
| fp16+ecc | RMSprop | No | No | 6/18 | 70.35s | 1.0 |
| fp16+ecc | SAMWrapper | No | No | 2/12 | 14.91s | 1.0 |
| fp16+ecc | SFAdamW | No | No | 1/6 | 143.88s | 6.0 |
| fp16+ecc | SGD | No | No | 2/12 | 138.09s | 2.0 |
| fp16+ecc | SOAP | No | No | 0/12 | 0.00s | 0.0 |
| fp16+ecc | SOAPAdEMAMix | No | No | 0/12 | 0.00s | 0.0 |
| fp16+ecc | SOAPNAdam | No | No | 1/12 | 211.41s | 14.0 |
| fp16+ecc | SOLP | No | No | 7/12 | 114.18s | 4.9 |
| fp16+ecc | Scion | No | No | 6/6 | 2.79s | 1.0 |
| fp16+ecc | UnscaledAdamW | No | No | 3/7 | 99.72s | 3.0 |

## Details

| Benchmark | Storage | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-------|-----------|---------|---|---|----------|------|---|---|----------------|
| CIFAR100-easy | bf16 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| CIFAR100-easy | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| CIFAR100-extreme | bf16 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| CIFAR100-extreme | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| CIFAR100-hard | bf16 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| CIFAR100-hard | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| CIFAR100-medium | bf16 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| CIFAR100-medium | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| CIFAR100-nightmare | bf16 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| CIFAR100-nightmare | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| CIFAR100-trivial | bf16 | MSAMLaProp | No | No | ✗ | 316.50s | inf | 1 | 1 | `bf16-MSAMLaProp(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| CIFAR100-trivial | bf16+ecc | PSGDKron | No | No | ✗ | 564.93s | inf | 1 | 1 | `bf16+ecc-PSGDKron(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| CIFAR10_wide-easy | bf16 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| CIFAR10_wide-easy | bf16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-easy | bf16+ecc | NAdam | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-easy | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| CIFAR10_wide-easy | fp16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-easy | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-easy | fp16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-extreme | bf16 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| CIFAR10_wide-extreme | bf16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-extreme | bf16+ecc | NAdam | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-extreme | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| CIFAR10_wide-extreme | fp16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-extreme | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-extreme | fp16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-hard | bf16 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| CIFAR10_wide-hard | bf16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-hard | bf16+ecc | NAdam | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-hard | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| CIFAR10_wide-hard | fp16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-hard | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-hard | fp16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-medium | bf16 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| CIFAR10_wide-medium | bf16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-medium | bf16+ecc | NAdam | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-medium | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| CIFAR10_wide-medium | fp16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-medium | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-medium | fp16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-nightmare | bf16 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| CIFAR10_wide-nightmare | bf16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-nightmare | bf16+ecc | NAdam | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-nightmare | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| CIFAR10_wide-nightmare | fp16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-nightmare | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-nightmare | fp16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| CIFAR10_wide-trivial | bf16 | LaPropOrtho | No | No | ✗ | 323.15s | inf | 1 | 7 | `bf16-LaPropOrtho(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| CIFAR10_wide-trivial | bf16+ecc | AdamC | No | No | ✗ | 308.59s | inf | 1 | 6 | `bf16+ecc-AdamC(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| CIFAR10_wide-trivial | bf16+ecc | NAdam | No | No | ✗ | 316.33s | inf | 1 | 6 | `bf16+ecc-NAdam(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| CIFAR10_wide-trivial | bf16+ecc | PSGDKron | No | No | ✗ | 1382.20s | inf | 1 | 7 | `bf16+ecc-PSGDKron(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| CIFAR10_wide-trivial | fp16 | LATHER | No | No | ✗ | 569.05s | inf | 1 | 6 | `fp16-LATHER(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| CIFAR10_wide-trivial | fp16+ecc | PSGDKron | No | No | ✗ | 1374.14s | inf | 1 | 6 | `fp16+ecc-PSGDKron(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| CIFAR10_wide-trivial | fp16+ecc | SOAP | No | No | ✗ | 358.65s | inf | 1 | 6 | `fp16+ecc-SOAP(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| MNIST-easy | bf16+ecc | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| MNIST-easy | bf16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| MNIST-extreme | bf16+ecc | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| MNIST-extreme | bf16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| MNIST-hard | bf16+ecc | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| MNIST-hard | bf16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| MNIST-medium | bf16+ecc | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| MNIST-medium | bf16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| MNIST-nightmare | bf16+ecc | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| MNIST-nightmare | bf16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| MNIST-trivial | bf16+ecc | PSGDLRA | No | No | ✗ | 307.47s | inf | 1 | 2 | `bf16+ecc-PSGDLRA(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| MNIST-trivial | bf16+ecc | SOAPAdEMAMix | No | No | ✗ | 313.37s | inf | 1 | 7 | `bf16+ecc-SOAPAdEMAMix(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| SVHN-easy | fp16 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| SVHN-easy | fp32 | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| SVHN-extreme | fp16 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| SVHN-extreme | fp32 | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| SVHN-hard | fp16 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| SVHN-hard | fp32 | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| SVHN-medium | fp16 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| SVHN-medium | fp32 | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| SVHN-nightmare | fp16 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| SVHN-nightmare | fp32 | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| SVHN-trivial | fp16 | UnscaledAdamW | No | No | ✗ | 307.32s | inf | 1 | 3 | `fp16-UnscaledAdamW(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| SVHN-trivial | fp32 | SOAPNAdam | No | No | ✗ | 301.22s | inf | 1 | 6 | `fp32-SOAPNAdam(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| Tolstoi_RNN-easy | bf16 | AdamC | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| Tolstoi_RNN-easy | bf16+ecc | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| Tolstoi_RNN-easy | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| Tolstoi_RNN-easy | fp16 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| Tolstoi_RNN-extreme | bf16 | AdamC | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| Tolstoi_RNN-extreme | bf16+ecc | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| Tolstoi_RNN-extreme | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| Tolstoi_RNN-extreme | fp16 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| Tolstoi_RNN-hard | bf16 | AdamC | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| Tolstoi_RNN-hard | bf16+ecc | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| Tolstoi_RNN-hard | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| Tolstoi_RNN-hard | fp16 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| Tolstoi_RNN-medium | bf16 | AdamC | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| Tolstoi_RNN-medium | bf16+ecc | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| Tolstoi_RNN-medium | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| Tolstoi_RNN-medium | fp16 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| Tolstoi_RNN-nightmare | bf16 | AdamC | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| Tolstoi_RNN-nightmare | bf16+ecc | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| Tolstoi_RNN-nightmare | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| Tolstoi_RNN-nightmare | fp16 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| Tolstoi_RNN-trivial | bf16 | AdamC | No | No | ✗ | 309.54s | inf | 1 | 0 | `bf16-AdamC(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| Tolstoi_RNN-trivial | bf16+ecc | ADOPT | No | No | ✗ | 302.01s | inf | 1 | 5 | `bf16+ecc-ADOPT(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| Tolstoi_RNN-trivial | bf16+ecc | LaProp | No | No | ✗ | 309.17s | inf | 1 | 2 | `bf16+ecc-LaProp(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| Tolstoi_RNN-trivial | fp16 | LaPropOrtho | No | No | ✗ | 323.75s | inf | 1 | 5 | `fp16-LaPropOrtho(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| absolute_varying_scale-easy | bf16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| absolute_varying_scale-easy | bf16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| absolute_varying_scale-easy | fp16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| absolute_varying_scale-easy | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| absolute_varying_scale-easy | fp32 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| absolute_varying_scale-extreme | bf16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| absolute_varying_scale-extreme | bf16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| absolute_varying_scale-extreme | fp16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| absolute_varying_scale-extreme | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| absolute_varying_scale-extreme | fp32 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| absolute_varying_scale-hard | bf16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| absolute_varying_scale-hard | bf16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| absolute_varying_scale-hard | fp16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| absolute_varying_scale-hard | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| absolute_varying_scale-hard | fp32 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| absolute_varying_scale-medium | bf16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| absolute_varying_scale-medium | bf16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| absolute_varying_scale-medium | fp16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| absolute_varying_scale-medium | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| absolute_varying_scale-medium | fp32 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| absolute_varying_scale-nightmare | bf16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| absolute_varying_scale-nightmare | bf16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| absolute_varying_scale-nightmare | fp16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| absolute_varying_scale-nightmare | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| absolute_varying_scale-nightmare | fp32 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| absolute_varying_scale-trivial | bf16 | MuonLaProp | No | No | ✗ | 300.39s | 6.19e-06 | 33 | 2 | `bf16-MuonLaProp(lr=0.00019, betas=(0.915, 0.9962), shampoo_beta=0.996) | EMAβ=0.90(lr=0.00120, betas=(0.932, 0.9981), shampoo_beta=0.988)` |
| absolute_varying_scale-trivial | bf16 | SignLaProp | No | No | ✗ | 300.02s | 3.31e-05 | 28 | 0 | `bf16-SignLaProp(lr=0.00032, betas=(0.909, 0.9984), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00279, betas=(0.948, 0.9986), shampoo_beta=1.000)` |
| absolute_varying_scale-trivial | fp16+ecc | AdamC | No | No | ✗ | 300.65s | 4.60e-05 | 36 | 0 | `fp16+ecc-AdamC(lr=0.00047, betas=(0.921, 0.9957), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00202, betas=(0.931, 0.9985), shampoo_beta=1.000)` |
| absolute_varying_scale-trivial | fp16+ecc | LaProp | No | No | ✗ | 300.02s | 7.56e-06 | 32 | 6 | `fp16+ecc-LaProp(lr=0.00028, betas=(0.977, 0.9889), shampoo_beta=0.917) | EMAβ=0.90(lr=0.00124, betas=(0.965, 1.0000), shampoo_beta=0.993)` |
| absolute_varying_scale-trivial | fp32 | AdEMAMix | No | No | ✗ | 300.27s | 2.08e-06 | 33 | 5 | `fp32-AdEMAMix(lr=0.00003, betas=(0.992, 1.0000), shampoo_beta=0.947) | EMAβ=0.90(lr=0.00005, betas=(0.997, 1.0000), shampoo_beta=0.937)` |
| adversarial_gradient-easy | bf16 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| adversarial_gradient-easy | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| adversarial_gradient-easy | fp16+ecc | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| adversarial_gradient-extreme | bf16 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| adversarial_gradient-extreme | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| adversarial_gradient-extreme | fp16+ecc | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| adversarial_gradient-hard | bf16 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| adversarial_gradient-hard | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| adversarial_gradient-hard | fp16+ecc | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| adversarial_gradient-medium | bf16 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| adversarial_gradient-medium | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| adversarial_gradient-medium | fp16+ecc | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| adversarial_gradient-nightmare | bf16 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| adversarial_gradient-nightmare | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| adversarial_gradient-nightmare | fp16+ecc | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| adversarial_gradient-trivial | bf16 | SOLP | No | No | ✗ | 137.56s | -2.40e-01 | 50 | 2 | `bf16-SOLP(lr=2.24352, betas=(0.969, 1.0000), shampoo_beta=0.983) | EMAβ=0.90(lr=1.09039, betas=(0.998, 0.9810), shampoo_beta=0.999)` |
| adversarial_gradient-trivial | bf16+ecc | AdamW | No | No | ✗ | 170.70s | -5.49e-02 | 50 | 7 | `bf16+ecc-AdamW(lr=0.25461, betas=(0.841, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.14571, betas=(0.971, 0.9999), shampoo_beta=1.000)` |
| adversarial_gradient-trivial | fp16+ecc | RMSprop | No | No | ✗ | 137.30s | -2.40e-01 | 50 | 0 | `fp16+ecc-RMSprop(lr=0.14448, betas=(0.204, 0.7327), shampoo_beta=0.600) | EMAβ=0.90(lr=0.00798, betas=(0.999, 0.9952), shampoo_beta=0.998)` |
| batch_size_scaling-easy | bf16+ecc | SUDSAdamW | No | No | ✓ | 278.82s | 2.80e-17 | 17 | 2 | `bf16+ecc-SUDSAdamW(lr=0.04757, betas=(0.729, 1.0000), shampoo_beta=0.989) | EMAβ=0.90(lr=0.00927, betas=(0.999, 1.0000), shampoo_beta=0.992)` |
| batch_size_scaling-easy | fp16 | MuonAdamW | No | No | ✓ | 103.86s | 3.24e-18 | 13 | 6 | `fp16-MuonAdamW(lr=0.06749, betas=(0.892, 1.0000), shampoo_beta=0.999) | EMAβ=0.90(lr=0.01322, betas=(0.998, 1.0000), shampoo_beta=0.974)` |
| batch_size_scaling-easy | fp32 | AdamC | No | No | ✓ | 96.40s | 2.66e-18 | 12 | 2 | `fp32-AdamC(lr=0.03345, betas=(0.990, 1.0000), shampoo_beta=0.196) | EMAβ=0.90(lr=0.01061, betas=(0.999, 1.0000), shampoo_beta=0.963)` |
| batch_size_scaling-extreme | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| batch_size_scaling-extreme | fp16 | MuonAdamW | No | No | ✓ | 119.85s | 4.12e-17 | 14 | 6 | `fp16-MuonAdamW(lr=0.01209, betas=(0.839, 1.0000), shampoo_beta=0.582) | EMAβ=0.90(lr=0.00343, betas=(0.997, 1.0000), shampoo_beta=0.932)` |
| batch_size_scaling-extreme | fp32 | AdamC | No | No | ✓ | 135.66s | 5.54e-18 | 13 | 2 | `fp32-AdamC(lr=0.00865, betas=(0.995, 0.9997), shampoo_beta=0.144) | EMAβ=0.90(lr=0.00322, betas=(0.998, 1.0000), shampoo_beta=0.933)` |
| batch_size_scaling-hard | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| batch_size_scaling-hard | fp16 | MuonAdamW | No | No | ✓ | 104.80s | 5.78e-18 | 13 | 6 | `fp16-MuonAdamW(lr=0.06749, betas=(0.892, 1.0000), shampoo_beta=0.999) | EMAβ=0.90(lr=0.01322, betas=(0.998, 1.0000), shampoo_beta=0.974)` |
| batch_size_scaling-hard | fp32 | AdamC | No | No | ✓ | 92.43s | 3.40e-41 | 12 | 2 | `fp32-AdamC(lr=0.23164, betas=(0.837, 1.0000), shampoo_beta=0.476) | EMAβ=0.90(lr=0.04157, betas=(0.996, 1.0000), shampoo_beta=0.962)` |
| batch_size_scaling-medium | bf16+ecc | SUDSAdamW | No | No | ✗ | 300.16s | 5.54e-03 | 19 | 2 | `bf16+ecc-SUDSAdamW(lr=0.00314, betas=(1.000, 1.0000), shampoo_beta=0.549) | EMAβ=0.90(lr=0.00555, betas=(1.000, 0.9999), shampoo_beta=0.965)` |
| batch_size_scaling-medium | fp16 | MuonAdamW | No | No | ✓ | 109.91s | 1.10e-35 | 13 | 6 | `fp16-MuonAdamW(lr=0.12964, betas=(0.671, 1.0000), shampoo_beta=0.869) | EMAβ=0.90(lr=0.01443, betas=(0.998, 1.0000), shampoo_beta=0.951)` |
| batch_size_scaling-medium | fp32 | AdamC | No | No | ✓ | 135.11s | 1.21e-17 | 13 | 2 | `fp32-AdamC(lr=0.00865, betas=(0.995, 0.9997), shampoo_beta=0.144) | EMAβ=0.90(lr=0.00263, betas=(0.998, 1.0000), shampoo_beta=0.934)` |
| batch_size_scaling-nightmare | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| batch_size_scaling-nightmare | fp16 | MuonAdamW | No | No | ✓ | 124.75s | 5.01e-17 | 14 | 6 | `fp16-MuonAdamW(lr=0.01209, betas=(0.839, 1.0000), shampoo_beta=0.582) | EMAβ=0.90(lr=0.00343, betas=(0.997, 1.0000), shampoo_beta=0.932)` |
| batch_size_scaling-nightmare | fp32 | AdamC | No | No | ✓ | 193.25s | 2.53e-18 | 17 | 2 | `fp32-AdamC(lr=0.03509, betas=(0.986, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.02704, betas=(0.998, 1.0000), shampoo_beta=0.997)` |
| batch_size_scaling-trivial | bf16+ecc | SUDSAdamW | No | No | ✓ | 146.89s | 7.60e-19 | 10 | 2 | `bf16+ecc-SUDSAdamW(lr=0.05890, betas=(0.008, 1.0000), shampoo_beta=0.958) | EMAβ=0.90(lr=0.00211, betas=(0.958, 1.0000), shampoo_beta=0.995)` |
| batch_size_scaling-trivial | fp16 | MuonAdamW | No | No | ✓ | 106.74s | 6.53e-27 | 13 | 6 | `fp16-MuonAdamW(lr=0.07405, betas=(0.565, 0.9994), shampoo_beta=0.181) | EMAβ=0.90(lr=0.01501, betas=(0.997, 1.0000), shampoo_beta=0.916)` |
| batch_size_scaling-trivial | fp32 | AdamC | No | No | ✓ | 98.67s | 1.35e-18 | 9 | 2 | `fp32-AdamC(lr=0.07468, betas=(0.994, 1.0000), shampoo_beta=0.031) | EMAβ=0.90(lr=0.00403, betas=(0.999, 1.0000), shampoo_beta=0.983)` |
| beale-easy | bf16 | AdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| beale-easy | bf16 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-easy | bf16 | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| beale-easy | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| beale-easy | bf16 | Scion | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-easy | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| beale-easy | bf16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-easy | fp16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| beale-easy | fp16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-easy | fp32 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| beale-easy | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| beale-easy | fp32 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| beale-extreme | bf16 | AdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| beale-extreme | bf16 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-extreme | bf16 | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| beale-extreme | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| beale-extreme | bf16 | Scion | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-extreme | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| beale-extreme | bf16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-extreme | fp16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| beale-extreme | fp16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-extreme | fp32 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| beale-extreme | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| beale-extreme | fp32 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| beale-hard | bf16 | AdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| beale-hard | bf16 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-hard | bf16 | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| beale-hard | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| beale-hard | bf16 | Scion | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-hard | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| beale-hard | bf16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-hard | fp16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| beale-hard | fp16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-hard | fp32 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| beale-hard | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| beale-hard | fp32 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| beale-medium | bf16 | AdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| beale-medium | bf16 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-medium | bf16 | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| beale-medium | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| beale-medium | bf16 | Scion | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-medium | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| beale-medium | bf16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-medium | fp16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| beale-medium | fp16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-medium | fp32 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| beale-medium | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| beale-medium | fp32 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| beale-nightmare | bf16 | AdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| beale-nightmare | bf16 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-nightmare | bf16 | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| beale-nightmare | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| beale-nightmare | bf16 | Scion | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-nightmare | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| beale-nightmare | bf16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-nightmare | fp16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| beale-nightmare | fp16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| beale-nightmare | fp32 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| beale-nightmare | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| beale-nightmare | fp32 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| beale-trivial | bf16 | AdamW | No | No | ✗ | 301.10s | 4.15e-01 | 13 | 1 | `bf16-AdamW(lr=0.09939, betas=(0.978, 0.9791), shampoo_beta=0.430) | EMAβ=0.90(lr=0.01325, betas=(0.995, 0.9989), shampoo_beta=0.955)` |
| beale-trivial | bf16 | NAdam | No | No | ✗ | 301.26s | 4.10e-02 | 16 | 7 | `bf16-NAdam(lr=2.38489, betas=(1.000, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.61127, betas=(1.000, 1.0000), shampoo_beta=0.999)` |
| beale-trivial | bf16 | OrthoLaProp | No | No | ✗ | 301.01s | 7.35e-01 | 10 | 3 | `bf16-OrthoLaProp(lr=0.02942, betas=(0.932, 1.0000), shampoo_beta=0.295) | EMAβ=0.90(lr=0.00370, betas=(0.994, 1.0000), shampoo_beta=0.994)` |
| beale-trivial | bf16 | SAMWrapper | No | No | ✗ | 300.03s | 3.09e+37 | 4 | 1 | `bf16-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| beale-trivial | bf16 | Scion | No | No | ✓ | 4.06s | 1.03e-13 | 1 | 7 | `bf16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| beale-trivial | bf16+ecc | AdamW | No | No | ✗ | 300.23s | 9.17e+00 | 13 | 4 | `bf16+ecc-AdamW(lr=0.00711, betas=(0.999, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00486, betas=(0.999, 1.0000), shampoo_beta=0.997)` |
| beale-trivial | bf16+ecc | SAMWrapper | No | No | ✗ | 300.71s | 3.09e+37 | 4 | 7 | `bf16+ecc-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| beale-trivial | fp16 | SAMWrapper | No | No | ✗ | 302.13s | 3.09e+37 | 4 | 0 | `fp16-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| beale-trivial | fp16 | SFAdamW | No | No | ✗ | 301.06s | 7.12e+03 | 6 | 7 | `fp16-SFAdamW(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00025, betas=(0.996, 0.9997), shampoo_beta=1.000)` |
| beale-trivial | fp32 | NAdam | No | No | ✗ | 301.15s | 3.00e-02 | 20 | 0 | `fp32-NAdam(lr=4.01865, betas=(1.000, 1.0000), shampoo_beta=0.992) | EMAβ=0.90(lr=0.59571, betas=(1.000, 1.0000), shampoo_beta=0.995)` |
| beale-trivial | fp32 | PSGDKron | No | No | ✗ | 292.07s | 2.35e+00 | 50 | 3 | `fp32-PSGDKron(lr=0.00026, betas=(1.000, 1.0000), shampoo_beta=0.940) | EMAβ=0.90(lr=0.00511, betas=(1.000, 1.0000), shampoo_beta=0.936)` |
| beale-trivial | fp32 | SOAP | No | No | ✗ | 300.58s | 3.56e-01 | 12 | 4 | `fp32-SOAP(lr=3.06450, betas=(0.983, 1.0000), shampoo_beta=0.247) | EMAβ=0.90(lr=0.04091, betas=(0.998, 1.0000), shampoo_beta=0.974)` |
| class_imbalance_rare-easy | bf16 | SOLP | No | No | ✓ | 244.06s | 4.36e-03 | 2 | 5 | `bf16-SOLP(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-easy | bf16+ecc | SOAP | No | No | ✓ | 254.68s | 8.46e-03 | 4 | 5 | `bf16+ecc-SOAP(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| class_imbalance_rare-easy | fp32 | PSGDLRA | No | No | ✓ | 129.22s | 3.91e-02 | 2 | 6 | `fp32-PSGDLRA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-easy | fp32 | RMSprop | No | No | ✓ | 67.02s | 1.47e-02 | 2 | 2 | `fp32-RMSprop(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-extreme | bf16 | SOLP | No | No | ✓ | 249.27s | 1.37e-03 | 2 | 5 | `bf16-SOLP(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-extreme | bf16+ecc | SOAP | No | No | ✓ | 282.82s | 8.17e-03 | 3 | 5 | `bf16+ecc-SOAP(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| class_imbalance_rare-extreme | fp32 | PSGDLRA | No | No | ✓ | 208.48s | 9.95e-10 | 3 | 6 | `fp32-PSGDLRA(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| class_imbalance_rare-extreme | fp32 | RMSprop | No | No | ✓ | 13.40s | 4.05e-02 | 1 | 2 | `fp32-RMSprop(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-hard | bf16 | SOLP | No | No | ✓ | 223.98s | 3.66e-02 | 2 | 5 | `bf16-SOLP(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-hard | bf16+ecc | SOAP | No | No | ✓ | 278.82s | 2.44e-02 | 2 | 5 | `bf16+ecc-SOAP(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-hard | fp32 | PSGDLRA | No | No | ✓ | 202.41s | 9.32e-11 | 3 | 6 | `fp32-PSGDLRA(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| class_imbalance_rare-hard | fp32 | RMSprop | No | No | ✓ | 23.16s | 3.59e-02 | 1 | 2 | `fp32-RMSprop(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-medium | bf16 | SOLP | No | No | ✓ | 187.45s | 3.79e-02 | 2 | 5 | `bf16-SOLP(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-medium | bf16+ecc | SOAP | No | No | ✓ | 205.94s | 3.77e-02 | 2 | 5 | `bf16+ecc-SOAP(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-medium | fp32 | PSGDLRA | No | No | ✓ | 182.31s | 3.19e-03 | 2 | 6 | `fp32-PSGDLRA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-medium | fp32 | RMSprop | No | No | ✓ | 49.67s | 4.00e-02 | 1 | 2 | `fp32-RMSprop(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-nightmare | bf16 | SOLP | No | No | ✗ | 300.95s | inf | 1 | 5 | `bf16-SOLP(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| class_imbalance_rare-nightmare | bf16+ecc | SOAP | No | No | ✗ | 302.63s | inf | 1 | 5 | `bf16+ecc-SOAP(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| class_imbalance_rare-nightmare | fp32 | PSGDLRA | No | No | ✓ | 148.67s | 3.94e-02 | 2 | 6 | `fp32-PSGDLRA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-nightmare | fp32 | RMSprop | No | No | ✓ | 53.39s | 4.00e-02 | 1 | 2 | `fp32-RMSprop(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-trivial | bf16 | SOLP | No | No | ✓ | 195.69s | 3.28e-02 | 2 | 5 | `bf16-SOLP(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-trivial | bf16+ecc | SOAP | No | No | ✓ | 287.75s | 0.00e+00 | 4 | 5 | `bf16+ecc-SOAP(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| class_imbalance_rare-trivial | fp16 | Muon | No | No | ✓ | 166.27s | 8.82e-03 | 2 | 0 | `fp16-Muon(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| class_imbalance_rare-trivial | fp32 | PSGDLRA | No | No | ✓ | 203.28s | 6.44e-14 | 3 | 6 | `fp32-PSGDLRA(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| class_imbalance_rare-trivial | fp32 | RMSprop | No | No | ✓ | 81.62s | 2.08e-02 | 2 | 2 | `fp32-RMSprop(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| constrained_optimization_log-easy | fp16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| constrained_optimization_log-extreme | fp16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| constrained_optimization_log-hard | fp16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| constrained_optimization_log-medium | fp16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| constrained_optimization_log-nightmare | fp16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| constrained_optimization_log-trivial | fp16+ecc | SAMWrapper | No | No | ✗ | 83.29s | -7.03e+00 | 50 | 2 | `fp16+ecc-SAMWrapper(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00016, betas=(0.979, 0.9996), shampoo_beta=0.999)` |
| constrained_optimization_relu-easy | bf16 | SGD | No | No | ✗ | 300.04s | 1.11e+00 | 25 | 6 | `bf16-SGD(lr=0.00000, betas=(1.000, 0.9715), shampoo_beta=0.606) | EMAβ=0.90(lr=0.00000, betas=(1.000, 0.9726), shampoo_beta=1.000)` |
| constrained_optimization_relu-easy | bf16 | SignLaProp | No | No | ✓ | 252.82s | 1.00e+00 | 20 | 5 | `bf16-SignLaProp(lr=0.00026, betas=(0.997, 1.0000), shampoo_beta=0.893) | EMAβ=0.90(lr=0.00044, betas=(0.999, 1.0000), shampoo_beta=0.949)` |
| constrained_optimization_relu-easy | bf16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| constrained_optimization_relu-easy | fp16 | SignLaProp | No | No | ✓ | 295.12s | 1.00e+00 | 24 | 0 | `fp16-SignLaProp(lr=0.00029, betas=(0.997, 1.0000), shampoo_beta=0.750) | EMAβ=0.90(lr=0.00111, betas=(1.000, 1.0000), shampoo_beta=0.945)` |
| constrained_optimization_relu-easy | fp32 | AdEMAMix | No | No | ✓ | 37.05s | 1.00e+00 | 2 | 7 | `fp32-AdEMAMix(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| constrained_optimization_relu-easy | fp32 | HyperBallAdamW | No | No | ✓ | 112.40s | 1.00e+00 | 7 | 4 | `fp32-HyperBallAdamW(lr=0.00172, betas=(0.999, 1.0000), shampoo_beta=0.778) | EMAβ=0.90(lr=0.00223, betas=(0.999, 1.0000), shampoo_beta=0.998)` |
| constrained_optimization_relu-extreme | bf16 | SGD | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| constrained_optimization_relu-extreme | bf16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| constrained_optimization_relu-extreme | bf16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| constrained_optimization_relu-extreme | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| constrained_optimization_relu-extreme | fp32 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| constrained_optimization_relu-extreme | fp32 | HyperBallAdamW | No | No | ✗ | 300.05s | 1.01e+00 | 25 | 4 | `fp32-HyperBallAdamW(lr=0.00030, betas=(0.935, 0.9885), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00118, betas=(0.982, 0.9998), shampoo_beta=1.000)` |
| constrained_optimization_relu-hard | bf16 | SGD | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| constrained_optimization_relu-hard | bf16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| constrained_optimization_relu-hard | bf16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| constrained_optimization_relu-hard | fp16 | SignLaProp | No | No | ✗ | 300.06s | 2.44e+00 | 14 | 0 | `fp16-SignLaProp(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00001, betas=(0.997, 0.9998), shampoo_beta=1.000)` |
| constrained_optimization_relu-hard | fp32 | AdEMAMix | No | No | ✗ | 300.06s | 1.00e+00 | 32 | 7 | `fp32-AdEMAMix(lr=0.00010, betas=(1.000, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00173, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| constrained_optimization_relu-hard | fp32 | HyperBallAdamW | No | No | ✓ | 251.08s | 1.00e+00 | 18 | 4 | `fp32-HyperBallAdamW(lr=0.00023, betas=(0.452, 0.2702), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00034, betas=(0.971, 0.9994), shampoo_beta=0.991)` |
| constrained_optimization_relu-medium | bf16 | SGD | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| constrained_optimization_relu-medium | bf16 | SignLaProp | No | No | ✗ | 300.30s | 1.83e+00 | 16 | 5 | `bf16-SignLaProp(lr=0.00007, betas=(0.999, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00007, betas=(1.000, 0.9983), shampoo_beta=1.000)` |
| constrained_optimization_relu-medium | bf16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| constrained_optimization_relu-medium | fp16 | SignLaProp | No | No | ✓ | 228.87s | 1.00e+00 | 11 | 0 | `fp16-SignLaProp(lr=0.00017, betas=(1.000, 0.9982), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00004, betas=(0.999, 0.9999), shampoo_beta=0.999)` |
| constrained_optimization_relu-medium | fp32 | AdEMAMix | No | No | ✓ | 220.42s | 1.00e+00 | 23 | 7 | `fp32-AdEMAMix(lr=0.01527, betas=(0.053, 0.9996), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00080, betas=(0.994, 1.0000), shampoo_beta=1.000)` |
| constrained_optimization_relu-medium | fp32 | HyperBallAdamW | No | No | ✓ | 224.97s | 1.00e+00 | 20 | 4 | `fp32-HyperBallAdamW(lr=0.00113, betas=(0.281, 0.8886), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00104, betas=(0.982, 1.0000), shampoo_beta=1.000)` |
| constrained_optimization_relu-nightmare | bf16 | SGD | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| constrained_optimization_relu-nightmare | bf16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| constrained_optimization_relu-nightmare | bf16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| constrained_optimization_relu-nightmare | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| constrained_optimization_relu-nightmare | fp32 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| constrained_optimization_relu-nightmare | fp32 | HyperBallAdamW | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| constrained_optimization_relu-trivial | bf16 | SGD | No | No | ✓ | 110.39s | 1.00e+00 | 9 | 6 | `bf16-SGD(lr=0.00001, betas=(0.992, 0.9902), shampoo_beta=0.329) | EMAβ=0.90(lr=0.00001, betas=(0.997, 0.9995), shampoo_beta=0.997)` |
| constrained_optimization_relu-trivial | bf16 | SignLaProp | No | No | ✓ | 252.33s | 1.00e+00 | 15 | 5 | `bf16-SignLaProp(lr=0.03810, betas=(0.998, 0.9885), shampoo_beta=0.960) | EMAβ=0.90(lr=0.04534, betas=(0.995, 0.9999), shampoo_beta=0.966)` |
| constrained_optimization_relu-trivial | bf16+ecc | SOAP | No | No | ✗ | 300.21s | 1.00e+00 | 23 | 5 | `bf16+ecc-SOAP(lr=0.00352, betas=(0.640, 0.9993), shampoo_beta=0.998) | EMAβ=0.90(lr=0.03167, betas=(0.995, 1.0000), shampoo_beta=0.999)` |
| constrained_optimization_relu-trivial | fp16 | SignLaProp | No | No | ✓ | 126.04s | 1.00e+00 | 11 | 0 | `fp16-SignLaProp(lr=0.03319, betas=(0.996, 1.0000), shampoo_beta=0.782) | EMAβ=0.90(lr=0.01560, betas=(0.999, 1.0000), shampoo_beta=0.986)` |
| constrained_optimization_relu-trivial | fp32 | AdEMAMix | No | No | ✓ | 41.00s | 1.00e+00 | 2 | 7 | `fp32-AdEMAMix(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| constrained_optimization_relu-trivial | fp32 | HyperBallAdamW | No | No | ✓ | 92.36s | 1.00e+00 | 7 | 4 | `fp32-HyperBallAdamW(lr=0.00066, betas=(0.999, 1.0000), shampoo_beta=0.470) | EMAβ=0.90(lr=0.00259, betas=(0.998, 1.0000), shampoo_beta=0.997)` |
| discontinuous_gradient-easy | bf16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| discontinuous_gradient-easy | fp16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| discontinuous_gradient-easy | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| discontinuous_gradient-extreme | bf16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| discontinuous_gradient-extreme | fp16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| discontinuous_gradient-extreme | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| discontinuous_gradient-hard | bf16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| discontinuous_gradient-hard | fp16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| discontinuous_gradient-hard | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| discontinuous_gradient-medium | bf16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| discontinuous_gradient-medium | fp16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| discontinuous_gradient-medium | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| discontinuous_gradient-nightmare | bf16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| discontinuous_gradient-nightmare | fp16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| discontinuous_gradient-nightmare | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| discontinuous_gradient-trivial | bf16+ecc | SGD | No | No | ✓ | 106.63s | 8.39e-09 | 8 | 3 | `bf16+ecc-SGD(lr=0.37437, betas=(0.726, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.02229, betas=(0.986, 0.9898), shampoo_beta=1.000)` |
| discontinuous_gradient-trivial | fp16+ecc | AdamW | No | No | ✓ | 167.88s | 2.97e-09 | 11 | 1 | `fp16+ecc-AdamW(lr=0.00245, betas=(0.196, 0.9400), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00198, betas=(0.938, 0.9998), shampoo_beta=0.998)` |
| discontinuous_gradient-trivial | fp32 | SGD | No | No | ✓ | 129.41s | 2.09e-23 | 7 | 3 | `fp32-SGD(lr=23.19339, betas=(0.083, 0.2367), shampoo_beta=0.984) | EMAβ=0.90(lr=0.01897, betas=(0.979, 0.9968), shampoo_beta=1.000)` |
| dynamic_landscape-easy | fp16 | LATHER | No | No | ✓ | 31.99s | 1.70e-05 | 3 | 0 | `fp16-LATHER(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| dynamic_landscape-easy | fp16 | RMSprop | No | No | ✓ | 68.31s | 8.64e-03 | 18 | 1 | `fp16-RMSprop(lr=0.14882, betas=(1.000, 0.1854), shampoo_beta=0.143) | EMAβ=0.90(lr=0.00954, betas=(0.998, 0.9975), shampoo_beta=0.972)` |
| dynamic_landscape-easy | fp16+ecc | LATHER | No | No | ✓ | 25.18s | 2.07e-03 | 3 | 0 | `fp16+ecc-LATHER(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| dynamic_landscape-easy | fp32 | SGD | No | No | ✓ | 55.42s | 1.75e-03 | 6 | 1 | `fp32-SGD(lr=15.98094, betas=(0.984, 0.9970), shampoo_beta=0.883) | EMAβ=0.90(lr=0.00269, betas=(0.995, 0.9995), shampoo_beta=0.999)` |
| dynamic_landscape-extreme | fp16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| dynamic_landscape-extreme | fp16+ecc | LATHER | No | No | ✓ | 127.25s | 4.44e-03 | 7 | 0 | `fp16+ecc-LATHER(lr=17.10960, betas=(0.919, 0.9072), shampoo_beta=1.000) | EMAβ=0.90(lr=0.01321, betas=(0.988, 0.9964), shampoo_beta=1.000)` |
| dynamic_landscape-extreme | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| dynamic_landscape-hard | fp16 | LATHER | No | No | ✓ | 44.57s | 5.13e-03 | 3 | 0 | `fp16-LATHER(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| dynamic_landscape-hard | fp16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| dynamic_landscape-hard | fp16+ecc | LATHER | No | No | ✓ | 35.80s | 5.14e-03 | 3 | 0 | `fp16+ecc-LATHER(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| dynamic_landscape-hard | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| dynamic_landscape-medium | fp16 | LATHER | No | No | ✓ | 46.51s | 4.33e-04 | 3 | 0 | `fp16-LATHER(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| dynamic_landscape-medium | fp16 | RMSprop | No | No | ✗ | 200.00s | 1.25e-01 | 50 | 1 | `fp16-RMSprop(lr=0.00465, betas=(0.245, 0.9990), shampoo_beta=0.111) | EMAβ=0.90(lr=0.00507, betas=(1.000, 0.9999), shampoo_beta=1.000)` |
| dynamic_landscape-medium | fp16+ecc | LATHER | No | No | ✓ | 39.66s | 4.33e-04 | 3 | 0 | `fp16+ecc-LATHER(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| dynamic_landscape-medium | fp32 | SGD | No | No | ✗ | 300.24s | 5.02e-01 | 24 | 1 | `fp32-SGD(lr=0.04066, betas=(0.988, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.07759, betas=(0.981, 1.0000), shampoo_beta=1.000)` |
| dynamic_landscape-nightmare | fp16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| dynamic_landscape-nightmare | fp16+ecc | LATHER | No | No | ✗ | 300.18s | 6.08e-01 | 14 | 0 | `fp16+ecc-LATHER(lr=0.00003, betas=(0.570, 0.9986), shampoo_beta=0.941) | EMAβ=0.90(lr=0.00001, betas=(0.980, 0.9999), shampoo_beta=0.980)` |
| dynamic_landscape-nightmare | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| dynamic_landscape-trivial | fp16 | LATHER | No | No | ✓ | 20.22s | 8.61e-06 | 3 | 0 | `fp16-LATHER(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| dynamic_landscape-trivial | fp16 | RMSprop | No | No | ✓ | 74.07s | 7.34e-03 | 13 | 1 | `fp16-RMSprop(lr=0.00859, betas=(1.000, 0.9742), shampoo_beta=0.991) | EMAβ=0.90(lr=0.00178, betas=(0.999, 0.9992), shampoo_beta=0.967)` |
| dynamic_landscape-trivial | fp16+ecc | LATHER | No | No | ✓ | 35.66s | 6.96e-06 | 3 | 0 | `fp16+ecc-LATHER(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| dynamic_landscape-trivial | fp32 | SGD | No | No | ✓ | 24.06s | 4.85e-03 | 3 | 1 | `fp32-SGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| exploding_gradient-easy | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| exploding_gradient-easy | bf16+ecc | SOLP | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| exploding_gradient-easy | fp16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| exploding_gradient-easy | fp16 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| exploding_gradient-extreme | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| exploding_gradient-extreme | bf16+ecc | SOLP | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| exploding_gradient-extreme | fp16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| exploding_gradient-extreme | fp16 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| exploding_gradient-hard | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| exploding_gradient-hard | bf16+ecc | SOLP | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| exploding_gradient-hard | fp16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| exploding_gradient-hard | fp16 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| exploding_gradient-medium | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| exploding_gradient-medium | bf16+ecc | SOLP | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| exploding_gradient-medium | fp16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| exploding_gradient-medium | fp16 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| exploding_gradient-nightmare | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| exploding_gradient-nightmare | bf16+ecc | SOLP | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| exploding_gradient-nightmare | fp16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| exploding_gradient-nightmare | fp16 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| exploding_gradient-trivial | bf16+ecc | LaProp | No | No | ✗ | 2.94s | inf | 50 | 2 | `bf16+ecc-LaProp(lr=nan, betas=(nan, nan), shampoo_beta=nan) | EMAβ=0.90(lr=0.00224, betas=(0.998, 0.9996), shampoo_beta=1.000)` |
| exploding_gradient-trivial | bf16+ecc | SOLP | No | No | ✗ | 5.87s | inf | 50 | 5 | `bf16+ecc-SOLP(lr=nan, betas=(nan, nan), shampoo_beta=nan) | EMAβ=0.90(lr=0.00224, betas=(0.998, 0.9996), shampoo_beta=1.000)` |
| exploding_gradient-trivial | fp16 | RMSprop | No | No | ✗ | 1.78s | inf | 50 | 5 | `fp16-RMSprop(lr=nan, betas=(nan, nan), shampoo_beta=nan) | EMAβ=0.90(lr=0.00224, betas=(0.998, 0.9996), shampoo_beta=1.000)` |
| exploding_gradient-trivial | fp16 | UnscaledAdamW | No | No | ✗ | 1.90s | inf | 50 | 0 | `fp16-UnscaledAdamW(lr=nan, betas=(nan, nan), shampoo_beta=nan) | EMAβ=0.90(lr=0.00224, betas=(0.998, 0.9996), shampoo_beta=1.000)` |
| generalization_summod-easy | fp16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| generalization_summod-extreme | fp16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| generalization_summod-hard | fp16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| generalization_summod-medium | fp16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| generalization_summod-nightmare | fp16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| generalization_summod-trivial | fp16+ecc | LaPropOrtho | No | No | ✗ | 300.54s | 2.65e+00 | 9 | 1 | `fp16+ecc-LaPropOrtho(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00237, betas=(0.999, 1.0000), shampoo_beta=0.994)` |
| generalization_weighted_summod-easy | bf16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| generalization_weighted_summod-easy | fp16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| generalization_weighted_summod-easy | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| generalization_weighted_summod-extreme | bf16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| generalization_weighted_summod-extreme | fp16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| generalization_weighted_summod-extreme | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| generalization_weighted_summod-hard | bf16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| generalization_weighted_summod-hard | fp16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| generalization_weighted_summod-hard | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| generalization_weighted_summod-medium | bf16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| generalization_weighted_summod-medium | fp16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| generalization_weighted_summod-medium | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| generalization_weighted_summod-nightmare | bf16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| generalization_weighted_summod-nightmare | fp16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| generalization_weighted_summod-nightmare | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| generalization_weighted_summod-trivial | bf16+ecc | SOAPNAdam | No | No | ✗ | 301.79s | 4.25e+00 | 2 | 4 | `bf16+ecc-SOAPNAdam(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| generalization_weighted_summod-trivial | fp16+ecc | LATHER | No | No | ✗ | 302.83s | inf | 1 | 6 | `fp16+ecc-LATHER(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| generalization_weighted_summod-trivial | fp16+ecc | LaProp | No | No | ✗ | 300.81s | 2.37e-01 | 14 | 0 | `fp16+ecc-LaProp(lr=0.00476, betas=(0.990, 1.0000), shampoo_beta=0.785) | EMAβ=0.90(lr=0.00893, betas=(0.987, 1.0000), shampoo_beta=0.968)` |
| gradient_delay-easy | bf16 | AdEMAMix | No | No | ✓ | 179.91s | 2.40e-06 | 4 | 7 | `bf16-AdEMAMix(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_delay-easy | bf16 | PSGDKron | No | No | ✓ | 285.96s | 2.87e-06 | 3 | 6 | `bf16-PSGDKron(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| gradient_delay-easy | bf16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_delay-easy | bf16+ecc | OrthoLaProp | No | No | ✓ | 185.51s | 3.62e-05 | 4 | 4 | `bf16+ecc-OrthoLaProp(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_delay-easy | fp32 | RMSprop | No | No | ✓ | 129.24s | 1.83e-06 | 4 | 4 | `fp32-RMSprop(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_delay-easy | fp32 | SOAPAdEMAMix | No | No | ✓ | 192.57s | 6.79e-05 | 4 | 1 | `fp32-SOAPAdEMAMix(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_delay-extreme | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| gradient_delay-extreme | bf16 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| gradient_delay-extreme | bf16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_delay-extreme | bf16+ecc | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| gradient_delay-extreme | fp32 | RMSprop | No | No | ✓ | 7.57s | 9.77e-08 | 4 | 4 | `fp32-RMSprop(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_delay-extreme | fp32 | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_delay-hard | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| gradient_delay-hard | bf16 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| gradient_delay-hard | bf16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_delay-hard | bf16+ecc | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| gradient_delay-hard | fp32 | RMSprop | No | No | ✓ | 188.09s | 1.95e-07 | 4 | 4 | `fp32-RMSprop(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_delay-hard | fp32 | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_delay-medium | bf16 | AdEMAMix | No | No | ✗ | 300.31s | 1.71e-01 | 3 | 7 | `bf16-AdEMAMix(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| gradient_delay-medium | bf16 | PSGDKron | No | No | ✗ | 301.49s | 4.55e-01 | 2 | 6 | `bf16-PSGDKron(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| gradient_delay-medium | bf16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_delay-medium | bf16+ecc | OrthoLaProp | No | No | ✗ | 300.31s | 2.65e-01 | 5 | 4 | `bf16+ecc-OrthoLaProp(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_delay-medium | fp32 | RMSprop | No | No | ✓ | 255.23s | 4.58e-07 | 4 | 4 | `fp32-RMSprop(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_delay-medium | fp32 | SOAPAdEMAMix | No | No | ✗ | 301.07s | 4.58e-01 | 2 | 1 | `fp32-SOAPAdEMAMix(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| gradient_delay-nightmare | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| gradient_delay-nightmare | bf16 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| gradient_delay-nightmare | bf16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_delay-nightmare | bf16+ecc | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| gradient_delay-nightmare | fp32 | RMSprop | No | No | ✓ | 15.07s | 2.86e-08 | 4 | 4 | `fp32-RMSprop(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_delay-nightmare | fp32 | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_delay-trivial | bf16 | AdEMAMix | No | No | ✓ | 176.06s | 4.66e-06 | 4 | 7 | `bf16-AdEMAMix(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_delay-trivial | bf16 | PSGDKron | No | No | ✓ | 235.40s | 5.01e-07 | 3 | 6 | `bf16-PSGDKron(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| gradient_delay-trivial | bf16+ecc | LaPropOrtho | No | No | ✗ | 300.61s | 1.03e+00 | 5 | 0 | `bf16+ecc-LaPropOrtho(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_delay-trivial | bf16+ecc | OrthoLaProp | No | No | ✓ | 180.76s | 9.59e-05 | 4 | 4 | `bf16+ecc-OrthoLaProp(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_delay-trivial | fp32 | RMSprop | No | No | ✓ | 106.41s | 3.67e-06 | 4 | 4 | `fp32-RMSprop(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_delay-trivial | fp32 | SOAPAdEMAMix | No | No | ✓ | 179.79s | 2.40e-05 | 4 | 1 | `fp32-SOAPAdEMAMix(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| gradient_noise_scale_heavy_tailed-easy | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| gradient_noise_scale_heavy_tailed-easy | bf16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_heavy_tailed-easy | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_heavy_tailed-easy | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| gradient_noise_scale_heavy_tailed-extreme | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| gradient_noise_scale_heavy_tailed-extreme | bf16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_heavy_tailed-extreme | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_heavy_tailed-extreme | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| gradient_noise_scale_heavy_tailed-hard | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| gradient_noise_scale_heavy_tailed-hard | bf16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_heavy_tailed-hard | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_heavy_tailed-hard | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| gradient_noise_scale_heavy_tailed-medium | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| gradient_noise_scale_heavy_tailed-medium | bf16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_heavy_tailed-medium | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_heavy_tailed-medium | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| gradient_noise_scale_heavy_tailed-nightmare | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| gradient_noise_scale_heavy_tailed-nightmare | bf16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_heavy_tailed-nightmare | bf16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_heavy_tailed-nightmare | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| gradient_noise_scale_heavy_tailed-trivial | bf16+ecc | AdamW | No | No | ✗ | 110.39s | 3.24e+00 | 50 | 7 | `bf16+ecc-AdamW(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00005, betas=(0.999, 1.0000), shampoo_beta=0.988)` |
| gradient_noise_scale_heavy_tailed-trivial | bf16+ecc | LATHER | No | No | ✗ | 301.69s | 2.46e+00 | 49 | 1 | `bf16+ecc-LATHER(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00001, betas=(1.000, 1.0000), shampoo_beta=0.999)` |
| gradient_noise_scale_heavy_tailed-trivial | bf16+ecc | PSGDKron | No | No | ✗ | 300.33s | 2.30e+00 | 50 | 1 | `bf16+ecc-PSGDKron(lr=0.65627, betas=(0.995, 0.2932), shampoo_beta=1.000) | EMAβ=0.90(lr=0.08688, betas=(1.000, 0.9996), shampoo_beta=0.999)` |
| gradient_noise_scale_heavy_tailed-trivial | fp16+ecc | PSGDKron | No | No | ✗ | 291.02s | 2.57e+00 | 50 | 6 | `fp16+ecc-PSGDKron(lr=0.00010, betas=(1.000, 1.0000), shampoo_beta=0.943) | EMAβ=0.90(lr=0.00001, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| gradient_noise_scale_instantaneous_brown-easy | bf16+ecc | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| gradient_noise_scale_instantaneous_brown-easy | fp32 | MSAMLaProp | No | No | ✗ | 278.67s | 3.70e-04 | 50 | 7 | `fp32-MSAMLaProp(lr=0.01111, betas=(0.928, 0.9975), shampoo_beta=0.922) | EMAβ=0.90(lr=0.03902, betas=(0.985, 0.9880), shampoo_beta=0.938)` |
| gradient_noise_scale_instantaneous_brown-extreme | bf16+ecc | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| gradient_noise_scale_instantaneous_brown-extreme | fp32 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| gradient_noise_scale_instantaneous_brown-hard | bf16+ecc | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| gradient_noise_scale_instantaneous_brown-hard | fp32 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| gradient_noise_scale_instantaneous_brown-medium | bf16+ecc | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| gradient_noise_scale_instantaneous_brown-medium | fp32 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| gradient_noise_scale_instantaneous_brown-nightmare | bf16+ecc | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| gradient_noise_scale_instantaneous_brown-nightmare | fp32 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| gradient_noise_scale_instantaneous_brown-trivial | bf16+ecc | PSGDLRA | No | No | ✗ | 300.52s | 1.79e-04 | 11 | 5 | `bf16+ecc-PSGDLRA(lr=0.06333, betas=(0.564, 0.2165), shampoo_beta=0.997) | EMAβ=0.90(lr=0.11866, betas=(0.975, 0.9905), shampoo_beta=0.999)` |
| gradient_noise_scale_instantaneous_brown-trivial | fp32 | MSAMLaProp | No | No | ✓ | 252.34s | 9.37e-05 | 44 | 7 | `fp32-MSAMLaProp(lr=0.01963, betas=(0.922, 1.0000), shampoo_beta=0.992) | EMAβ=0.90(lr=0.03844, betas=(0.981, 1.0000), shampoo_beta=0.998)` |
| gradient_noise_scale_temporal_brown-easy | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_noise_scale_temporal_brown-easy | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| gradient_noise_scale_temporal_brown-extreme | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_noise_scale_temporal_brown-extreme | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| gradient_noise_scale_temporal_brown-hard | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_noise_scale_temporal_brown-hard | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| gradient_noise_scale_temporal_brown-medium | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_noise_scale_temporal_brown-medium | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| gradient_noise_scale_temporal_brown-nightmare | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_noise_scale_temporal_brown-nightmare | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| gradient_noise_scale_temporal_brown-trivial | fp16+ecc | MuonLaProp | No | No | ✗ | 300.12s | 7.24e-06 | 8 | 0 | `fp16+ecc-MuonLaProp(lr=2.18378, betas=(0.993, 1.0000), shampoo_beta=0.285) | EMAβ=0.90(lr=0.00548, betas=(0.997, 0.9999), shampoo_beta=0.997)` |
| gradient_noise_scale_temporal_brown-trivial | fp32 | SGD | No | No | ✗ | 300.05s | 4.34e-07 | 20 | 6 | `fp32-SGD(lr=91.24062, betas=(0.777, 0.5124), shampoo_beta=0.996) | EMAβ=0.90(lr=12.33985, betas=(0.898, 0.9997), shampoo_beta=1.000)` |
| gradient_noise_scale_white-easy | bf16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_noise_scale_white-easy | bf16 | SFAdamW | No | No | ✓ | 207.56s | 4.11e-03 | 9 | 3 | `bf16-SFAdamW(lr=1.04325, betas=(0.996, 1.0000), shampoo_beta=0.997) | EMAβ=0.90(lr=0.01849, betas=(0.997, 1.0000), shampoo_beta=0.992)` |
| gradient_noise_scale_white-easy | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| gradient_noise_scale_white-easy | fp32 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_white-extreme | bf16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_noise_scale_white-extreme | bf16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| gradient_noise_scale_white-extreme | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| gradient_noise_scale_white-extreme | fp32 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_white-hard | bf16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_noise_scale_white-hard | bf16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| gradient_noise_scale_white-hard | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| gradient_noise_scale_white-hard | fp32 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_white-medium | bf16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_noise_scale_white-medium | bf16 | SFAdamW | No | No | ✗ | 300.48s | 1.56e-02 | 13 | 3 | `bf16-SFAdamW(lr=0.00957, betas=(0.962, 1.0000), shampoo_beta=0.996) | EMAβ=0.90(lr=0.02144, betas=(0.998, 1.0000), shampoo_beta=0.998)` |
| gradient_noise_scale_white-medium | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| gradient_noise_scale_white-medium | fp32 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_white-nightmare | bf16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| gradient_noise_scale_white-nightmare | bf16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| gradient_noise_scale_white-nightmare | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| gradient_noise_scale_white-nightmare | fp32 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| gradient_noise_scale_white-trivial | bf16 | MuonLaProp | No | No | ✗ | 300.06s | 9.87e-04 | 20 | 0 | `bf16-MuonLaProp(lr=0.00163, betas=(0.745, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00452, betas=(0.975, 0.9997), shampoo_beta=1.000)` |
| gradient_noise_scale_white-trivial | bf16 | SFAdamW | No | No | ✓ | 208.98s | 1.03e-03 | 9 | 3 | `bf16-SFAdamW(lr=0.57652, betas=(0.999, 0.9996), shampoo_beta=0.876) | EMAβ=0.90(lr=0.01849, betas=(0.997, 1.0000), shampoo_beta=0.992)` |
| gradient_noise_scale_white-trivial | fp16+ecc | MuonLaProp | No | No | ✗ | 300.21s | 1.02e-03 | 22 | 3 | `fp16+ecc-MuonLaProp(lr=0.00188, betas=(0.671, 0.8688), shampoo_beta=0.999) | EMAβ=0.90(lr=0.00320, betas=(0.980, 0.9057), shampoo_beta=0.999)` |
| gradient_noise_scale_white-trivial | fp32 | NAdam | No | No | ✗ | 300.07s | 1.04e-03 | 11 | 1 | `fp32-NAdam(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.03042, betas=(0.998, 1.0000), shampoo_beta=0.987)` |
| grokking-easy | bf16 | MuonAdamW | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-easy | bf16 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| grokking-easy | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-easy | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-easy | fp16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-easy | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| grokking-easy | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-easy | fp16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-easy | fp32 | Scion | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| grokking-extreme | bf16 | MuonAdamW | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-extreme | bf16 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| grokking-extreme | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-extreme | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-extreme | fp16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-extreme | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| grokking-extreme | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-extreme | fp16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-extreme | fp32 | Scion | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| grokking-hard | bf16 | MuonAdamW | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-hard | bf16 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| grokking-hard | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-hard | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-hard | fp16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-hard | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| grokking-hard | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-hard | fp16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-hard | fp32 | Scion | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| grokking-medium | bf16 | MuonAdamW | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-medium | bf16 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| grokking-medium | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-medium | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-medium | fp16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-medium | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| grokking-medium | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-medium | fp16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-medium | fp32 | Scion | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| grokking-nightmare | bf16 | MuonAdamW | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-nightmare | bf16 | NAdam | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| grokking-nightmare | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-nightmare | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-nightmare | fp16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-nightmare | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| grokking-nightmare | fp16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| grokking-nightmare | fp16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| grokking-nightmare | fp32 | Scion | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| grokking-trivial | bf16 | MuonAdamW | No | No | ✗ | 0.35s | inf | 50 | 0 | N/A |
| grokking-trivial | bf16 | NAdam | No | No | ✗ | 0.01s | inf | 50 | 3 | N/A |
| grokking-trivial | bf16+ecc | AdamW | No | No | ✗ | 0.72s | inf | 50 | 6 | N/A |
| grokking-trivial | bf16+ecc | LaProp | No | No | ✗ | 0.36s | inf | 50 | 0 | N/A |
| grokking-trivial | fp16 | LATHER | No | No | ✗ | 0.00s | inf | 50 | 6 | N/A |
| grokking-trivial | fp16 | SignLaProp | No | No | ✗ | 0.39s | inf | 50 | 4 | N/A |
| grokking-trivial | fp16+ecc | MuonLaProp | No | No | ✗ | 0.33s | inf | 50 | 0 | N/A |
| grokking-trivial | fp16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.01s | inf | 50 | 6 | N/A |
| grokking-trivial | fp32 | Scion | No | No | ✗ | 0.00s | inf | 50 | 5 | N/A |
| ill_conditioned-easy | bf16 | ADOPT | No | No | ✓ | 35.55s | 6.86e-04 | 3 | 2 | `bf16-ADOPT(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| ill_conditioned-easy | bf16+ecc | MSAMLaProp | No | No | ✓ | 74.30s | 7.24e-04 | 4 | 5 | `bf16+ecc-MSAMLaProp(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| ill_conditioned-easy | fp16 | SGD | No | No | ✓ | 19.77s | 6.51e-08 | 3 | 0 | `fp16-SGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| ill_conditioned-easy | fp32 | AdamW | No | No | ✓ | 57.98s | 2.74e-06 | 7 | 5 | `fp32-AdamW(lr=2.18378, betas=(0.993, 1.0000), shampoo_beta=0.285) | EMAβ=0.90(lr=0.00562, betas=(0.998, 1.0000), shampoo_beta=0.997)` |
| ill_conditioned-easy | fp32 | HyperBallAdamW | No | No | ✓ | 48.97s | 4.83e-05 | 4 | 2 | `fp32-HyperBallAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| ill_conditioned-easy | fp32 | PSGDKron | No | No | ✓ | 56.40s | 7.47e-04 | 3 | 0 | `fp32-PSGDKron(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| ill_conditioned-extreme | bf16 | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| ill_conditioned-extreme | bf16+ecc | MSAMLaProp | No | No | ✓ | 96.10s | 1.78e-06 | 6 | 5 | `bf16+ecc-MSAMLaProp(lr=0.57652, betas=(0.999, 0.9996), shampoo_beta=0.876) | EMAβ=0.90(lr=0.00132, betas=(0.997, 0.9997), shampoo_beta=0.999)` |
| ill_conditioned-extreme | fp16 | SGD | No | No | ✗ | 131.63s | 3.92e-03 | 50 | 0 | `fp16-SGD(lr=0.00115, betas=(0.988, 1.0000), shampoo_beta=0.999) | EMAβ=0.90(lr=0.06635, betas=(0.973, 1.0000), shampoo_beta=0.997)` |
| ill_conditioned-extreme | fp32 | AdamW | No | No | ✓ | 56.30s | 3.51e-13 | 7 | 5 | `fp32-AdamW(lr=0.36845, betas=(0.962, 1.0000), shampoo_beta=0.923) | EMAβ=0.90(lr=0.00399, betas=(0.998, 1.0000), shampoo_beta=0.998)` |
| ill_conditioned-extreme | fp32 | HyperBallAdamW | No | No | ✓ | 81.40s | 1.37e-04 | 8 | 2 | `fp32-HyperBallAdamW(lr=0.00438, betas=(0.999, 0.9999), shampoo_beta=0.178) | EMAβ=0.90(lr=0.00352, betas=(0.998, 1.0000), shampoo_beta=0.988)` |
| ill_conditioned-extreme | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| ill_conditioned-hard | bf16 | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| ill_conditioned-hard | bf16+ecc | MSAMLaProp | No | No | ✓ | 98.16s | 3.56e-06 | 7 | 5 | `bf16+ecc-MSAMLaProp(lr=0.50652, betas=(0.986, 1.0000), shampoo_beta=0.995) | EMAβ=0.90(lr=0.00396, betas=(0.996, 0.9999), shampoo_beta=0.998)` |
| ill_conditioned-hard | fp16 | SGD | No | No | ✓ | 152.81s | 8.94e-04 | 14 | 0 | `fp16-SGD(lr=0.00356, betas=(0.996, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00001, betas=(0.997, 0.9998), shampoo_beta=1.000)` |
| ill_conditioned-hard | fp32 | AdamW | No | No | ✓ | 63.78s | 1.28e-04 | 7 | 5 | `fp32-AdamW(lr=2.18378, betas=(0.993, 1.0000), shampoo_beta=0.285) | EMAβ=0.90(lr=0.00562, betas=(0.998, 1.0000), shampoo_beta=0.997)` |
| ill_conditioned-hard | fp32 | HyperBallAdamW | No | No | ✓ | 84.86s | 8.63e-04 | 7 | 2 | `fp32-HyperBallAdamW(lr=0.00172, betas=(0.999, 1.0000), shampoo_beta=0.778) | EMAβ=0.90(lr=0.00246, betas=(0.998, 1.0000), shampoo_beta=0.996)` |
| ill_conditioned-hard | fp32 | PSGDKron | No | No | ✗ | 300.20s | 1.98e-03 | 19 | 0 | `fp32-PSGDKron(lr=0.00366, betas=(0.998, 1.0000), shampoo_beta=0.984) | EMAβ=0.90(lr=0.00709, betas=(0.995, 1.0000), shampoo_beta=0.996)` |
| ill_conditioned-medium | bf16 | ADOPT | No | No | ✗ | 101.27s | 2.35e-02 | 50 | 2 | `bf16-ADOPT(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=4.33203, betas=(1.000, 1.0000), shampoo_beta=0.802)` |
| ill_conditioned-medium | bf16+ecc | MSAMLaProp | No | No | ✓ | 72.10s | 5.93e-04 | 4 | 5 | `bf16+ecc-MSAMLaProp(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| ill_conditioned-medium | fp16 | SGD | No | No | ✓ | 78.77s | 3.02e-04 | 8 | 0 | `fp16-SGD(lr=0.12161, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00129, betas=(0.998, 0.9999), shampoo_beta=1.000)` |
| ill_conditioned-medium | fp32 | AdamW | No | No | ✓ | 61.38s | 1.81e-05 | 7 | 5 | `fp32-AdamW(lr=2.18378, betas=(0.993, 1.0000), shampoo_beta=0.285) | EMAβ=0.90(lr=0.00562, betas=(0.998, 1.0000), shampoo_beta=0.997)` |
| ill_conditioned-medium | fp32 | HyperBallAdamW | No | No | ✓ | 48.85s | 2.81e-04 | 4 | 2 | `fp32-HyperBallAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| ill_conditioned-medium | fp32 | PSGDKron | No | No | ✓ | 56.48s | 9.68e-04 | 3 | 0 | `fp32-PSGDKron(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| ill_conditioned-nightmare | bf16 | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| ill_conditioned-nightmare | bf16+ecc | MSAMLaProp | No | No | ✓ | 142.24s | 3.11e-06 | 19 | 5 | `bf16+ecc-MSAMLaProp(lr=0.13600, betas=(0.987, 1.0000), shampoo_beta=0.947) | EMAβ=0.90(lr=0.02753, betas=(0.999, 1.0000), shampoo_beta=0.926)` |
| ill_conditioned-nightmare | fp16 | SGD | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| ill_conditioned-nightmare | fp32 | AdamW | No | No | ✓ | 70.14s | 1.94e-12 | 7 | 5 | `fp32-AdamW(lr=0.36845, betas=(0.962, 1.0000), shampoo_beta=0.923) | EMAβ=0.90(lr=0.00399, betas=(0.998, 1.0000), shampoo_beta=0.998)` |
| ill_conditioned-nightmare | fp32 | HyperBallAdamW | No | No | ✓ | 89.79s | 1.89e-04 | 7 | 2 | `fp32-HyperBallAdamW(lr=0.00183, betas=(0.981, 1.0000), shampoo_beta=0.504) | EMAβ=0.90(lr=0.00119, betas=(0.997, 0.9999), shampoo_beta=0.996)` |
| ill_conditioned-nightmare | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| ill_conditioned-trivial | bf16 | ADOPT | No | No | ✓ | 48.20s | 1.47e-04 | 3 | 2 | `bf16-ADOPT(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| ill_conditioned-trivial | bf16+ecc | MSAMLaProp | No | No | ✓ | 76.97s | 2.07e-04 | 4 | 5 | `bf16+ecc-MSAMLaProp(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| ill_conditioned-trivial | fp16 | SGD | No | No | ✓ | 21.38s | 5.53e-12 | 3 | 0 | `fp16-SGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| ill_conditioned-trivial | fp16+ecc | UnscaledAdamW | No | No | ✓ | 104.82s | 2.13e-04 | 3 | 5 | `fp16+ecc-UnscaledAdamW(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| ill_conditioned-trivial | fp32 | AdamW | No | No | ✓ | 60.70s | 2.29e-05 | 7 | 5 | `fp32-AdamW(lr=21.36936, betas=(0.994, 1.0000), shampoo_beta=0.194) | EMAβ=0.90(lr=0.00870, betas=(0.998, 1.0000), shampoo_beta=0.997)` |
| ill_conditioned-trivial | fp32 | HyperBallAdamW | No | No | ✓ | 48.76s | 1.19e-05 | 4 | 2 | `fp32-HyperBallAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| ill_conditioned-trivial | fp32 | PSGDKron | No | No | ✓ | 53.61s | 1.23e-04 | 3 | 0 | `fp32-PSGDKron(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| ill_conditioned_mismatched-easy | fp16+ecc | NAdam | No | No | ✓ | 127.21s | 8.45e-04 | 11 | 5 | `fp16+ecc-NAdam(lr=0.00293, betas=(0.683, 1.0000), shampoo_beta=0.428) | EMAβ=0.90(lr=0.00651, betas=(0.996, 1.0000), shampoo_beta=0.967)` |
| ill_conditioned_mismatched-extreme | fp16+ecc | NAdam | No | No | ✗ | 300.23s | 5.52e-02 | 23 | 5 | `fp16+ecc-NAdam(lr=0.01393, betas=(0.451, 0.0997), shampoo_beta=0.843) | EMAβ=0.90(lr=0.00341, betas=(0.996, 0.9998), shampoo_beta=1.000)` |
| ill_conditioned_mismatched-hard | fp16+ecc | NAdam | No | No | ✓ | 124.88s | 3.78e-04 | 13 | 5 | `fp16+ecc-NAdam(lr=0.00520, betas=(1.000, 0.6396), shampoo_beta=0.968) | EMAβ=0.90(lr=0.00314, betas=(1.000, 1.0000), shampoo_beta=0.990)` |
| ill_conditioned_mismatched-medium | fp16+ecc | NAdam | No | No | ✓ | 100.41s | 9.26e-04 | 9 | 5 | `fp16+ecc-NAdam(lr=0.01408, betas=(1.000, 0.9970), shampoo_beta=0.600) | EMAβ=0.90(lr=0.00386, betas=(0.999, 0.9999), shampoo_beta=0.988)` |
| ill_conditioned_mismatched-nightmare | fp16+ecc | NAdam | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| ill_conditioned_mismatched-trivial | fp16+ecc | NAdam | No | No | ✓ | 108.00s | 8.63e-04 | 7 | 5 | `fp16+ecc-NAdam(lr=0.01246, betas=(1.000, 1.0000), shampoo_beta=0.063) | EMAβ=0.90(lr=0.00207, betas=(0.998, 0.9999), shampoo_beta=0.996)` |
| layer_wise_scale-easy | fp16 | MSAMLaProp | No | No | ✓ | 226.64s | 1.53e-05 | 11 | 5 | `fp16-MSAMLaProp(lr=0.03253, betas=(0.991, 0.9977), shampoo_beta=0.971) | EMAβ=0.90(lr=0.01002, betas=(0.997, 0.9995), shampoo_beta=0.974)` |
| layer_wise_scale-easy | fp16+ecc | UnscaledAdamW | No | No | ✓ | 105.40s | 8.93e-05 | 3 | 2 | `fp16+ecc-UnscaledAdamW(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| layer_wise_scale-extreme | fp16+ecc | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| layer_wise_scale-hard | fp16+ecc | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| layer_wise_scale-medium | fp16+ecc | UnscaledAdamW | No | No | ✗ | 205.95s | 2.38e-04 | 50 | 2 | `fp16+ecc-UnscaledAdamW(lr=25.54165, betas=(0.978, 0.9998), shampoo_beta=0.999) | EMAβ=0.90(lr=61.57255, betas=(0.594, 0.9400), shampoo_beta=0.800)` |
| layer_wise_scale-nightmare | fp16+ecc | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| layer_wise_scale-trivial | fp16 | MSAMLaProp | No | No | ✓ | 256.79s | 8.82e-05 | 16 | 5 | `fp16-MSAMLaProp(lr=0.12169, betas=(0.985, 0.2453), shampoo_beta=1.000) | EMAβ=0.90(lr=0.22648, betas=(0.987, 0.9903), shampoo_beta=1.000)` |
| layer_wise_scale-trivial | fp16+ecc | UnscaledAdamW | No | No | ✓ | 88.94s | 7.12e-05 | 3 | 2 | `fp16+ecc-UnscaledAdamW(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| memory_constrained-easy | fp16+ecc | RMSprop | No | No | ✓ | 86.06s | 6.84e-02 | 1 | 2 | `fp16+ecc-RMSprop(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| memory_constrained-easy | fp32 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| memory_constrained-extreme | fp16+ecc | RMSprop | No | No | ✓ | 34.37s | 8.80e-02 | 1 | 2 | `fp16+ecc-RMSprop(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| memory_constrained-extreme | fp32 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| memory_constrained-hard | fp16+ecc | RMSprop | No | No | ✓ | 38.91s | 7.04e-02 | 1 | 2 | `fp16+ecc-RMSprop(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| memory_constrained-hard | fp32 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| memory_constrained-medium | fp16+ecc | RMSprop | No | No | ✓ | 45.35s | 3.66e-02 | 1 | 2 | `fp16+ecc-RMSprop(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| memory_constrained-medium | fp32 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| memory_constrained-nightmare | fp16+ecc | RMSprop | No | No | ✓ | 138.91s | 9.99e-02 | 1 | 2 | `fp16+ecc-RMSprop(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| memory_constrained-nightmare | fp32 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| memory_constrained-trivial | fp16+ecc | RMSprop | No | No | ✓ | 78.53s | 1.54e-01 | 1 | 2 | `fp16+ecc-RMSprop(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| memory_constrained-trivial | fp32 | LATHER | No | No | ✗ | 51.11s | inf | 8 | 6 | `fp32-LATHER(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| minimax-easy | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| minimax-easy | fp16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| minimax-easy | fp16 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| minimax-easy | fp16 | Scion | No | No | ✓ | 3.10s | 3.77e+00 | 1 | 7 | `fp16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| minimax-easy | fp16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| minimax-extreme | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| minimax-extreme | fp16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| minimax-extreme | fp16 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| minimax-extreme | fp16 | Scion | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| minimax-extreme | fp16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| minimax-hard | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| minimax-hard | fp16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| minimax-hard | fp16 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| minimax-hard | fp16 | Scion | No | No | ✗ | 3.88s | inf | 50 | 7 | `fp16-Scion(lr=nan, betas=(nan, nan), shampoo_beta=nan) | EMAβ=0.90(lr=0.00224, betas=(0.998, 0.9996), shampoo_beta=1.000)` |
| minimax-hard | fp16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| minimax-medium | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| minimax-medium | fp16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| minimax-medium | fp16 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| minimax-medium | fp16 | Scion | No | No | ✓ | 4.03s | 1.11e+02 | 1 | 7 | `fp16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| minimax-medium | fp16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| minimax-nightmare | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| minimax-nightmare | fp16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| minimax-nightmare | fp16 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| minimax-nightmare | fp16 | Scion | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| minimax-nightmare | fp16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| minimax-trivial | bf16 | SAMWrapper | No | No | ✗ | 300.11s | 1.88e+00 | 7 | 6 | `bf16-SAMWrapper(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00108, betas=(0.998, 0.9999), shampoo_beta=0.999)` |
| minimax-trivial | fp16 | AdEMAMix | No | No | ✗ | 300.39s | 2.99e+00 | 22 | 0 | `fp16-AdEMAMix(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00114, betas=(0.991, 0.9998), shampoo_beta=0.999)` |
| minimax-trivial | fp16 | MSAMLaProp | No | No | ✗ | 300.04s | 2.37e+00 | 30 | 4 | `fp16-MSAMLaProp(lr=0.00211, betas=(0.984, 1.0000), shampoo_beta=0.138) | EMAβ=0.90(lr=0.00977, betas=(0.998, 1.0000), shampoo_beta=0.999)` |
| minimax-trivial | fp16 | Scion | No | No | ✓ | 3.31s | 1.02e+00 | 1 | 7 | `fp16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| minimax-trivial | fp16+ecc | SGD | No | No | ✗ | 178.47s | 2.76e+00 | 50 | 1 | `fp16+ecc-SGD(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.16631, betas=(0.986, 0.9991), shampoo_beta=1.000)` |
| momentum_utilization-easy | bf16 | SGD | No | No | ✓ | 46.03s | -5.73e-03 | 3 | 2 | `bf16-SGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| momentum_utilization-extreme | bf16 | SGD | No | No | ✓ | 42.82s | -7.43e-04 | 2 | 2 | `bf16-SGD(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| momentum_utilization-hard | bf16 | SGD | No | No | ✓ | 39.74s | -2.97e-01 | 3 | 2 | `bf16-SGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| momentum_utilization-medium | bf16 | SGD | No | No | ✓ | 39.85s | -7.91e-04 | 2 | 2 | `bf16-SGD(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| momentum_utilization-nightmare | bf16 | SGD | No | No | ✓ | 42.03s | -1.22e+00 | 3 | 2 | `bf16-SGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| momentum_utilization-trivial | bf16 | SGD | No | No | ✓ | 47.19s | -2.78e-04 | 3 | 2 | `bf16-SGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| multi_objective_pareto-easy | bf16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| multi_objective_pareto-easy | bf16+ecc | HyperBallAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| multi_objective_pareto-extreme | bf16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| multi_objective_pareto-extreme | bf16+ecc | HyperBallAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| multi_objective_pareto-hard | bf16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| multi_objective_pareto-hard | bf16+ecc | HyperBallAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| multi_objective_pareto-medium | bf16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| multi_objective_pareto-medium | bf16+ecc | HyperBallAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| multi_objective_pareto-nightmare | bf16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| multi_objective_pareto-nightmare | bf16+ecc | HyperBallAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| multi_objective_pareto-trivial | bf16 | SFAdamW | No | No | ✗ | 301.38s | 3.52e-01 | 4 | 2 | `bf16-SFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| multi_objective_pareto-trivial | bf16+ecc | HyperBallAdamW | No | No | ✗ | 300.35s | 2.78e+00 | 7 | 7 | `bf16+ecc-HyperBallAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00237, betas=(0.999, 0.9999), shampoo_beta=0.999)` |
| noisy_matmul-easy | bf16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| noisy_matmul-easy | bf16+ecc | LaProp | No | No | ✓ | 118.99s | 2.36e-15 | 6 | 5 | `bf16+ecc-LaProp(lr=2.17510, betas=(0.970, 1.0000), shampoo_beta=0.990) | EMAβ=0.90(lr=0.00176, betas=(0.994, 0.9998), shampoo_beta=0.999)` |
| noisy_matmul-easy | fp16 | AdamC | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| noisy_matmul-easy | fp16 | SFAdamW | No | No | ✗ | 300.29s | 1.15e-13 | 12 | 3 | `fp16-SFAdamW(lr=53.50527, betas=(1.000, 1.0000), shampoo_beta=0.992) | EMAβ=0.90(lr=0.55130, betas=(0.999, 1.0000), shampoo_beta=0.997)` |
| noisy_matmul-easy | fp16+ecc | SOLP | No | No | ✓ | 143.40s | 2.25e-16 | 6 | 2 | `fp16+ecc-SOLP(lr=1.33207, betas=(0.978, 0.9997), shampoo_beta=0.985) | EMAβ=0.90(lr=0.00158, betas=(0.995, 0.9997), shampoo_beta=0.999)` |
| noisy_matmul-easy | fp32 | PSGDKron | No | No | ✓ | 245.04s | 4.28e-15 | 9 | 6 | `fp32-PSGDKron(lr=7.34656, betas=(0.689, 0.9985), shampoo_beta=0.999) | EMAβ=0.90(lr=0.06635, betas=(0.977, 0.9945), shampoo_beta=1.000)` |
| noisy_matmul-extreme | bf16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| noisy_matmul-extreme | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| noisy_matmul-extreme | fp16 | AdamC | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| noisy_matmul-extreme | fp16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| noisy_matmul-extreme | fp16+ecc | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| noisy_matmul-extreme | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| noisy_matmul-hard | bf16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| noisy_matmul-hard | bf16+ecc | LaProp | No | No | ✗ | 302.41s | 6.44e-06 | 6 | 5 | `bf16+ecc-LaProp(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00025, betas=(0.996, 0.9997), shampoo_beta=1.000)` |
| noisy_matmul-hard | fp16 | AdamC | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| noisy_matmul-hard | fp16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| noisy_matmul-hard | fp16+ecc | SOLP | No | No | ✗ | 300.67s | 3.10e-07 | 10 | 2 | `fp16+ecc-SOLP(lr=0.00172, betas=(0.999, 1.0000), shampoo_beta=0.778) | EMAβ=0.90(lr=0.00234, betas=(0.999, 1.0000), shampoo_beta=0.992)` |
| noisy_matmul-hard | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| noisy_matmul-medium | bf16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| noisy_matmul-medium | bf16+ecc | LaProp | No | No | ✓ | 223.11s | 5.78e-18 | 7 | 5 | `bf16+ecc-LaProp(lr=0.22530, betas=(0.785, 1.0000), shampoo_beta=0.872) | EMAβ=0.90(lr=0.00383, betas=(0.995, 0.9999), shampoo_beta=0.998)` |
| noisy_matmul-medium | fp16 | AdamC | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| noisy_matmul-medium | fp16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| noisy_matmul-medium | fp16+ecc | SOLP | No | No | ✓ | 292.71s | 3.11e-17 | 10 | 2 | `fp16+ecc-SOLP(lr=2.12092, betas=(0.928, 0.9473), shampoo_beta=1.000) | EMAβ=0.90(lr=0.15866, betas=(0.989, 0.9807), shampoo_beta=1.000)` |
| noisy_matmul-medium | fp32 | PSGDKron | No | No | ✗ | 301.30s | 5.21e-12 | 7 | 6 | `fp32-PSGDKron(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00354, betas=(0.998, 0.9984), shampoo_beta=1.000)` |
| noisy_matmul-nightmare | bf16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| noisy_matmul-nightmare | bf16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| noisy_matmul-nightmare | fp16 | AdamC | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| noisy_matmul-nightmare | fp16 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| noisy_matmul-nightmare | fp16+ecc | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| noisy_matmul-nightmare | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| noisy_matmul-trivial | bf16 | PSGDLRA | No | No | ✗ | 300.21s | 1.87e-14 | 8 | 7 | `bf16-PSGDLRA(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.01190, betas=(0.993, 0.9972), shampoo_beta=1.000)` |
| noisy_matmul-trivial | bf16+ecc | LaProp | No | No | ✓ | 117.67s | 9.74e-16 | 6 | 5 | `bf16+ecc-LaProp(lr=1.59269, betas=(0.985, 1.0000), shampoo_beta=0.677) | EMAβ=0.90(lr=0.00165, betas=(0.995, 0.9998), shampoo_beta=0.999)` |
| noisy_matmul-trivial | fp16 | AdamC | No | No | ✗ | 300.04s | 1.60e-12 | 16 | 4 | `fp16-AdamC(lr=0.05384, betas=(0.928, 0.9991), shampoo_beta=0.166) | EMAβ=0.90(lr=0.02121, betas=(0.961, 0.9997), shampoo_beta=0.918)` |
| noisy_matmul-trivial | fp16 | SFAdamW | No | No | ✓ | 165.04s | 6.83e-17 | 9 | 3 | `fp16-SFAdamW(lr=88.05407, betas=(0.994, 1.0000), shampoo_beta=0.988) | EMAβ=0.90(lr=0.10678, betas=(0.999, 0.9999), shampoo_beta=0.999)` |
| noisy_matmul-trivial | fp16+ecc | SOLP | No | No | ✓ | 131.65s | 2.81e-15 | 6 | 2 | `fp16+ecc-SOLP(lr=1.33207, betas=(0.978, 0.9997), shampoo_beta=0.985) | EMAβ=0.90(lr=0.00158, betas=(0.995, 0.9997), shampoo_beta=0.999)` |
| noisy_matmul-trivial | fp32 | PSGDKron | No | No | ✓ | 201.25s | 0.00e+00 | 9 | 6 | `fp32-PSGDKron(lr=17.55357, betas=(0.658, 0.9343), shampoo_beta=0.995) | EMAβ=0.90(lr=0.05818, betas=(0.983, 0.9892), shampoo_beta=1.000)` |
| non_markovian-easy | bf16+ecc | SGD | No | No | ✓ | 50.97s | 4.52e-04 | 12 | 1 | `bf16+ecc-SGD(lr=0.00030, betas=(0.956, 0.9990), shampoo_beta=0.979) | EMAβ=0.90(lr=0.00025, betas=(1.000, 0.9998), shampoo_beta=0.997)` |
| non_markovian-easy | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| non_markovian-easy | fp16+ecc | HyperBallAdamW | No | No | ✓ | 182.11s | 1.96e-02 | 8 | 5 | `fp16+ecc-HyperBallAdamW(lr=0.00476, betas=(0.990, 1.0000), shampoo_beta=0.785) | EMAβ=0.90(lr=0.00153, betas=(0.990, 1.0000), shampoo_beta=0.993)` |
| non_markovian-extreme | bf16+ecc | SGD | No | No | ✓ | 27.57s | 1.20e+01 | 6 | 1 | `bf16+ecc-SGD(lr=0.07160, betas=(0.948, 1.0000), shampoo_beta=0.839) | EMAβ=0.90(lr=0.00085, betas=(0.993, 0.9999), shampoo_beta=0.999)` |
| non_markovian-extreme | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| non_markovian-extreme | fp16+ecc | HyperBallAdamW | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| non_markovian-hard | bf16+ecc | SGD | No | No | ✓ | 33.62s | 9.26e-01 | 9 | 1 | `bf16+ecc-SGD(lr=0.01945, betas=(0.976, 0.9999), shampoo_beta=0.128) | EMAβ=0.90(lr=0.00302, betas=(0.998, 1.0000), shampoo_beta=0.986)` |
| non_markovian-hard | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| non_markovian-hard | fp16+ecc | HyperBallAdamW | No | No | ✗ | 300.73s | 7.27e-01 | 10 | 5 | `fp16+ecc-HyperBallAdamW(lr=0.01396, betas=(0.954, 1.0000), shampoo_beta=0.131) | EMAβ=0.90(lr=0.01564, betas=(0.997, 1.0000), shampoo_beta=0.984)` |
| non_markovian-medium | bf16+ecc | SGD | No | No | ✓ | 153.16s | 6.24e-02 | 19 | 1 | `bf16+ecc-SGD(lr=0.00053, betas=(0.969, 0.9993), shampoo_beta=0.999) | EMAβ=0.90(lr=0.00314, betas=(0.992, 0.9997), shampoo_beta=0.998)` |
| non_markovian-medium | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| non_markovian-medium | fp16+ecc | HyperBallAdamW | No | No | ✓ | 225.82s | 2.95e-02 | 11 | 5 | `fp16+ecc-HyperBallAdamW(lr=0.00208, betas=(0.986, 1.0000), shampoo_beta=0.725) | EMAβ=0.90(lr=0.01141, betas=(0.995, 1.0000), shampoo_beta=0.982)` |
| non_markovian-nightmare | bf16+ecc | SGD | No | No | ✗ | 123.34s | 4.52e+02 | 50 | 1 | `bf16+ecc-SGD(lr=3.64784, betas=(0.482, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.01180, betas=(0.963, 0.9999), shampoo_beta=1.000)` |
| non_markovian-nightmare | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| non_markovian-nightmare | fp16+ecc | HyperBallAdamW | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| non_markovian-trivial | bf16+ecc | SGD | No | No | ✓ | 292.53s | 2.53e-03 | 18 | 1 | `bf16+ecc-SGD(lr=0.00455, betas=(0.984, 1.0000), shampoo_beta=0.989) | EMAβ=0.90(lr=0.00087, betas=(0.991, 1.0000), shampoo_beta=0.977)` |
| non_markovian-trivial | fp16 | SignLaProp | No | No | ✗ | 300.17s | 1.65e-02 | 17 | 3 | `fp16-SignLaProp(lr=3.90593, betas=(1.000, 0.1655), shampoo_beta=0.338) | EMAβ=0.90(lr=0.29736, betas=(0.999, 0.9928), shampoo_beta=0.896)` |
| non_markovian-trivial | fp16+ecc | HyperBallAdamW | No | No | ✓ | 253.50s | 9.79e-03 | 8 | 5 | `fp16+ecc-HyperBallAdamW(lr=0.00476, betas=(0.990, 1.0000), shampoo_beta=0.785) | EMAβ=0.90(lr=0.00153, betas=(0.990, 1.0000), shampoo_beta=0.993)` |
| parameter_scale-easy | bf16 | Scion | No | No | ✓ | 1.79s | 6.38e-16 | 1 | 4 | `bf16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-easy | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| parameter_scale-easy | fp16+ecc | Scion | No | No | ✓ | 1.89s | 6.38e-16 | 1 | 1 | `fp16+ecc-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-easy | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| parameter_scale-easy | fp32 | SFAdamW | No | No | ✓ | 174.37s | 3.10e-05 | 6 | 0 | `fp32-SFAdamW(lr=1.33207, betas=(0.978, 0.9997), shampoo_beta=0.985) | EMAβ=0.90(lr=0.00158, betas=(0.995, 0.9997), shampoo_beta=0.999)` |
| parameter_scale-extreme | bf16 | Scion | No | No | ✓ | 1.89s | 6.38e-16 | 1 | 4 | `bf16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-extreme | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| parameter_scale-extreme | fp16+ecc | Scion | No | No | ✓ | 1.85s | 6.38e-16 | 1 | 1 | `fp16+ecc-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-extreme | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| parameter_scale-extreme | fp32 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| parameter_scale-hard | bf16 | Scion | No | No | ✓ | 1.82s | 5.11e-16 | 1 | 4 | `bf16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-hard | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| parameter_scale-hard | fp16+ecc | Scion | No | No | ✓ | 1.85s | 5.19e-16 | 1 | 1 | `fp16+ecc-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-hard | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| parameter_scale-hard | fp32 | SFAdamW | No | No | ✗ | 300.76s | 1.93e+08 | 9 | 0 | `fp32-SFAdamW(lr=24.25742, betas=(0.698, 0.3152), shampoo_beta=0.974) | EMAβ=0.90(lr=0.01814, betas=(0.979, 0.9947), shampoo_beta=0.993)` |
| parameter_scale-medium | bf16 | Scion | No | No | ✓ | 1.85s | 6.38e-16 | 1 | 4 | `bf16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-medium | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| parameter_scale-medium | fp16+ecc | Scion | No | No | ✓ | 1.84s | 6.38e-16 | 1 | 1 | `fp16+ecc-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-medium | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| parameter_scale-medium | fp32 | SFAdamW | No | No | ✓ | 292.26s | 9.07e-05 | 11 | 0 | `fp32-SFAdamW(lr=59.87383, betas=(1.000, 0.9415), shampoo_beta=1.000) | EMAβ=0.90(lr=0.63702, betas=(0.999, 0.9783), shampoo_beta=1.000)` |
| parameter_scale-nightmare | bf16 | Scion | No | No | ✓ | 1.88s | 6.38e-16 | 1 | 4 | `bf16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-nightmare | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| parameter_scale-nightmare | fp16+ecc | Scion | No | No | ✓ | 1.85s | 6.38e-16 | 1 | 1 | `fp16+ecc-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-nightmare | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| parameter_scale-nightmare | fp32 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| parameter_scale-trivial | bf16 | Scion | No | No | ✓ | 5.09s | 6.38e-16 | 1 | 4 | `bf16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-trivial | bf16+ecc | SUDSAdamW | No | No | ✗ | 300.45s | 5.47e-01 | 5 | 2 | `bf16+ecc-SUDSAdamW(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| parameter_scale-trivial | fp16+ecc | Scion | No | No | ✓ | 7.43s | 6.38e-16 | 1 | 1 | `fp16+ecc-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-trivial | fp32 | SAMWrapper | No | No | ✗ | 300.61s | 1.90e+10 | 5 | 2 | `fp32-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| parameter_scale-trivial | fp32 | SFAdamW | No | No | ✓ | 273.13s | 9.07e-05 | 11 | 0 | `fp32-SFAdamW(lr=59.87383, betas=(1.000, 0.9415), shampoo_beta=1.000) | EMAβ=0.90(lr=0.63702, betas=(0.999, 0.9783), shampoo_beta=1.000)` |
| plasticity_summod-easy | fp16 | LaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| plasticity_summod-extreme | fp16 | LaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| plasticity_summod-hard | fp16 | LaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| plasticity_summod-medium | fp16 | LaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| plasticity_summod-nightmare | fp16 | LaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| plasticity_summod-trivial | fp16 | LaProp | No | No | ✗ | 150.92s | 1.70e+00 | 50 | 3 | `fp16-LaProp(lr=0.05810, betas=(0.135, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.02063, betas=(0.921, 1.0000), shampoo_beta=1.000)` |
| plasticity_weighted_summod-easy | fp16 | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| plasticity_weighted_summod-easy | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| plasticity_weighted_summod-easy | fp32 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| plasticity_weighted_summod-extreme | fp16 | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| plasticity_weighted_summod-extreme | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| plasticity_weighted_summod-extreme | fp32 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| plasticity_weighted_summod-hard | fp16 | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| plasticity_weighted_summod-hard | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| plasticity_weighted_summod-hard | fp32 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| plasticity_weighted_summod-medium | fp16 | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| plasticity_weighted_summod-medium | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| plasticity_weighted_summod-medium | fp32 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| plasticity_weighted_summod-nightmare | fp16 | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| plasticity_weighted_summod-nightmare | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| plasticity_weighted_summod-nightmare | fp32 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| plasticity_weighted_summod-trivial | fp16 | SUDSAdamW | No | No | ✗ | 303.82s | 1.52e+00 | 38 | 1 | `fp16-SUDSAdamW(lr=0.04339, betas=(0.849, 0.9998), shampoo_beta=1.000) | EMAβ=0.90(lr=0.05186, betas=(0.969, 1.0000), shampoo_beta=1.000)` |
| plasticity_weighted_summod-trivial | fp32 | PSGDKron | No | No | ✗ | 216.00s | 1.17e+00 | 50 | 2 | `fp32-PSGDKron(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=9.73381, betas=(0.999, 0.9973), shampoo_beta=0.984)` |
| plasticity_weighted_summod-trivial | fp32 | UnscaledAdamW | No | No | ✗ | 227.95s | 2.10e+00 | 50 | 5 | `fp32-UnscaledAdamW(lr=1.58470, betas=(0.596, 0.9444), shampoo_beta=1.000) | EMAβ=0.90(lr=30.81919, betas=(0.692, 0.9944), shampoo_beta=1.000)` |
| plateau_navigation-easy | bf16+ecc | PSGDPRO | No | No | ✓ | 73.09s | 0.00e+00 | 3 | 1 | `bf16+ecc-PSGDPRO(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| plateau_navigation-easy | fp16+ecc | LaProp | No | No | ✓ | 103.49s | 7.32e-05 | 2 | 6 | `fp16+ecc-LaProp(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| plateau_navigation-extreme | bf16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| plateau_navigation-extreme | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| plateau_navigation-hard | bf16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| plateau_navigation-hard | fp16+ecc | LaProp | No | No | ✗ | 300.52s | 1.00e+00 | 6 | 6 | `fp16+ecc-LaProp(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00025, betas=(0.996, 0.9997), shampoo_beta=1.000)` |
| plateau_navigation-medium | bf16+ecc | PSGDPRO | No | No | ✗ | 300.61s | 1.00e+00 | 9 | 1 | `bf16+ecc-PSGDPRO(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00236, betas=(0.997, 0.9991), shampoo_beta=1.000)` |
| plateau_navigation-medium | fp16+ecc | LaProp | No | No | ✓ | 118.62s | 3.99e-07 | 4 | 6 | `fp16+ecc-LaProp(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| plateau_navigation-nightmare | bf16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| plateau_navigation-nightmare | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| plateau_navigation-trivial | bf16+ecc | PSGDPRO | No | No | ✓ | 79.89s | 0.00e+00 | 3 | 1 | `bf16+ecc-PSGDPRO(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| plateau_navigation-trivial | fp16+ecc | LaProp | No | No | ✓ | 110.45s | 0.00e+00 | 3 | 6 | `fp16+ecc-LaProp(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| powers-easy | bf16+ecc | MuonLaProp | No | No | ✓ | 154.83s | 4.09e-09 | 8 | 2 | `bf16+ecc-MuonLaProp(lr=0.00121, betas=(0.804, 1.0000), shampoo_beta=0.004) | EMAβ=0.90(lr=0.00221, betas=(0.997, 1.0000), shampoo_beta=0.994)` |
| powers-easy | fp16+ecc | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| powers-extreme | bf16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| powers-extreme | fp16+ecc | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| powers-hard | bf16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| powers-hard | fp16+ecc | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| powers-medium | bf16+ecc | MuonLaProp | No | No | ✗ | 205.23s | 8.19e-03 | 50 | 2 | `bf16+ecc-MuonLaProp(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=4.33203, betas=(1.000, 1.0000), shampoo_beta=0.802)` |
| powers-medium | fp16+ecc | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| powers-nightmare | bf16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| powers-nightmare | fp16+ecc | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| powers-trivial | bf16+ecc | MuonLaProp | No | No | ✓ | 274.05s | 5.95e-09 | 15 | 2 | `bf16+ecc-MuonLaProp(lr=0.01016, betas=(0.825, 1.0000), shampoo_beta=0.969) | EMAβ=0.90(lr=0.00220, betas=(0.991, 1.0000), shampoo_beta=0.970)` |
| powers-trivial | fp16+ecc | RMSprop | No | No | ✗ | 300.09s | 4.84e-05 | 13 | 1 | `fp16+ecc-RMSprop(lr=0.00215, betas=(0.976, 0.9997), shampoo_beta=0.998) | EMAβ=0.90(lr=0.00198, betas=(0.998, 1.0000), shampoo_beta=0.994)` |
| powers_varying_target-easy | bf16 | SOLP | No | No | ✗ | 158.57s | 3.65e-01 | 50 | 2 | `bf16-SOLP(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=8.43409, betas=(1.000, 1.0000), shampoo_beta=0.989)` |
| powers_varying_target-easy | fp32 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| powers_varying_target-extreme | bf16 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| powers_varying_target-extreme | fp32 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| powers_varying_target-hard | bf16 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| powers_varying_target-hard | fp32 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| powers_varying_target-medium | bf16 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| powers_varying_target-medium | fp32 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| powers_varying_target-nightmare | bf16 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| powers_varying_target-nightmare | fp32 | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| powers_varying_target-trivial | bf16 | SOLP | No | No | ✓ | 246.24s | 7.89e-07 | 11 | 2 | `bf16-SOLP(lr=0.04363, betas=(0.064, 1.0000), shampoo_beta=0.345) | EMAβ=0.90(lr=0.01010, betas=(0.994, 1.0000), shampoo_beta=0.971)` |
| powers_varying_target-trivial | fp16 | MSAMLaProp | No | No | ✓ | 178.25s | 8.63e-07 | 12 | 6 | `fp16-MSAMLaProp(lr=7.85171, betas=(0.999, 0.9885), shampoo_beta=0.968) | EMAβ=0.90(lr=0.54474, betas=(0.999, 0.9983), shampoo_beta=0.989)` |
| powers_varying_target-trivial | fp32 | LaPropOrtho | No | No | ✗ | 300.46s | 7.00e-01 | 48 | 3 | `fp32-LaPropOrtho(lr=0.02004, betas=(0.982, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00071, betas=(0.990, 1.0000), shampoo_beta=1.000)` |
| quadratic_coupled-easy | bf16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_coupled-easy | bf16+ecc | Muon | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_coupled-easy | fp16 | SOAP | No | No | ✗ | 300.30s | 4.90e-14 | 16 | 3 | `fp16-SOAP(lr=0.03946, betas=(0.879, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.03669, betas=(0.941, 0.9989), shampoo_beta=1.000)` |
| quadratic_coupled-easy | fp32 | SOLP | No | No | ✗ | 300.42s | 2.15e-07 | 15 | 1 | `fp32-SOLP(lr=0.00471, betas=(0.790, 1.0000), shampoo_beta=0.989) | EMAβ=0.90(lr=0.00750, betas=(0.897, 1.0000), shampoo_beta=0.992)` |
| quadratic_coupled-extreme | bf16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_coupled-extreme | bf16+ecc | Muon | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_coupled-extreme | fp16 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_coupled-extreme | fp32 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| quadratic_coupled-hard | bf16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_coupled-hard | bf16+ecc | Muon | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_coupled-hard | fp16 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_coupled-hard | fp32 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| quadratic_coupled-medium | bf16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_coupled-medium | bf16+ecc | Muon | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_coupled-medium | fp16 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_coupled-medium | fp32 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| quadratic_coupled-nightmare | bf16+ecc | LaPropOrtho | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_coupled-nightmare | bf16+ecc | Muon | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_coupled-nightmare | fp16 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_coupled-nightmare | fp32 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| quadratic_coupled-trivial | bf16+ecc | LaPropOrtho | No | No | ✗ | 300.22s | 1.10e+00 | 17 | 5 | `bf16+ecc-LaPropOrtho(lr=0.00000, betas=(1.000, 0.1023), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.999, 0.9999), shampoo_beta=1.000)` |
| quadratic_coupled-trivial | bf16+ecc | Muon | No | No | ✗ | 300.05s | 1.03e-08 | 21 | 7 | `bf16+ecc-Muon(lr=0.00017, betas=(0.343, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00159, betas=(0.979, 1.0000), shampoo_beta=1.000)` |
| quadratic_coupled-trivial | fp16 | SOAP | No | No | ✓ | 233.84s | 2.70e-19 | 19 | 3 | `fp16-SOAP(lr=0.02796, betas=(0.607, 0.9992), shampoo_beta=1.000) | EMAβ=0.90(lr=0.03874, betas=(0.946, 0.9996), shampoo_beta=1.000)` |
| quadratic_coupled-trivial | fp32 | SOLP | No | No | ✓ | 154.50s | 1.69e-23 | 13 | 1 | `fp32-SOLP(lr=0.02079, betas=(0.977, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.06920, betas=(0.982, 0.9998), shampoo_beta=0.995)` |
| quadratic_flip-easy | bf16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_flip-easy | fp16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_flip-easy | fp16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| quadratic_flip-extreme | bf16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_flip-extreme | fp16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_flip-extreme | fp16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| quadratic_flip-hard | bf16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_flip-hard | fp16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_flip-hard | fp16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| quadratic_flip-medium | bf16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_flip-medium | fp16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_flip-medium | fp16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| quadratic_flip-nightmare | bf16 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_flip-nightmare | fp16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_flip-nightmare | fp16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| quadratic_flip-trivial | bf16 | LATHER | No | No | ✗ | 140.97s | -1.43e+08 | 50 | 0 | `bf16-LATHER(lr=99.57851, betas=(0.352, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=1.04573, betas=(0.972, 0.9998), shampoo_beta=1.000)` |
| quadratic_flip-trivial | fp16 | RMSprop | No | No | ✗ | 110.51s | -1.28e+06 | 50 | 5 | `fp16-RMSprop(lr=73.26260, betas=(0.522, 0.9999), shampoo_beta=1.000) | EMAβ=0.90(lr=13.79643, betas=(0.999, 0.9999), shampoo_beta=1.000)` |
| quadratic_flip-trivial | fp16+ecc | SOAPNAdam | No | No | ✗ | 131.19s | -6.14e+06 | 50 | 4 | `fp16+ecc-SOAPNAdam(lr=45.02022, betas=(0.443, 1.0000), shampoo_beta=0.011) | EMAβ=0.90(lr=27.42913, betas=(0.989, 1.0000), shampoo_beta=0.965)` |
| quadratic_outliers-easy | bf16 | AdamW | No | No | ✗ | 300.08s | 6.97e-14 | 37 | 6 | `bf16-AdamW(lr=0.00573, betas=(0.996, 1.0000), shampoo_beta=0.995) | EMAβ=0.90(lr=0.01343, betas=(0.966, 1.0000), shampoo_beta=1.000)` |
| quadratic_outliers-easy | bf16 | Muon | No | No | ✓ | 45.95s | 1.05e-30 | 3 | 7 | `bf16-Muon(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| quadratic_outliers-easy | bf16+ecc | MuonLaProp | No | No | ✗ | 300.26s | 1.83e-06 | 23 | 5 | `bf16+ecc-MuonLaProp(lr=0.07820, betas=(0.831, 1.0000), shampoo_beta=0.208) | EMAβ=0.90(lr=0.04804, betas=(0.959, 0.9999), shampoo_beta=0.767)` |
| quadratic_outliers-easy | fp16 | MuonAdamW | No | No | ✗ | 300.42s | 3.52e-08 | 18 | 5 | `fp16-MuonAdamW(lr=0.02296, betas=(0.996, 1.0000), shampoo_beta=0.989) | EMAβ=0.90(lr=0.00234, betas=(0.998, 1.0000), shampoo_beta=0.972)` |
| quadratic_outliers-easy | fp16 | PSGDKron | No | No | ✗ | 156.74s | 2.41e-12 | 50 | 0 | `fp16-PSGDKron(lr=4.63838, betas=(0.793, 0.8961), shampoo_beta=1.000) | EMAβ=0.90(lr=35.36863, betas=(0.235, 1.0000), shampoo_beta=1.000)` |
| quadratic_outliers-extreme | bf16 | AdamW | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_outliers-extreme | bf16 | Muon | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_outliers-extreme | bf16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_outliers-extreme | fp16 | MuonAdamW | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_outliers-extreme | fp16 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_outliers-hard | bf16 | AdamW | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_outliers-hard | bf16 | Muon | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_outliers-hard | bf16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_outliers-hard | fp16 | MuonAdamW | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_outliers-hard | fp16 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_outliers-medium | bf16 | AdamW | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_outliers-medium | bf16 | Muon | No | No | ✗ | 135.92s | 2.24e-03 | 50 | 7 | `bf16-Muon(lr=0.25008, betas=(1.000, 0.9997), shampoo_beta=0.988) | EMAβ=0.90(lr=7.29025, betas=(1.000, 1.0000), shampoo_beta=0.973)` |
| quadratic_outliers-medium | bf16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_outliers-medium | fp16 | MuonAdamW | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_outliers-medium | fp16 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_outliers-nightmare | bf16 | AdamW | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_outliers-nightmare | bf16 | Muon | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_outliers-nightmare | bf16+ecc | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_outliers-nightmare | fp16 | MuonAdamW | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_outliers-nightmare | fp16 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_outliers-trivial | bf16 | AdamW | No | No | ✓ | 159.98s | 1.21e-15 | 11 | 6 | `bf16-AdamW(lr=0.00501, betas=(0.985, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00198, betas=(0.998, 1.0000), shampoo_beta=0.999)` |
| quadratic_outliers-trivial | bf16 | Muon | No | No | ✓ | 58.95s | 5.56e-31 | 3 | 7 | `bf16-Muon(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| quadratic_outliers-trivial | bf16+ecc | MuonLaProp | No | No | ✓ | 157.18s | 5.45e-17 | 8 | 5 | `bf16+ecc-MuonLaProp(lr=0.01782, betas=(0.784, 1.0000), shampoo_beta=0.749) | EMAβ=0.90(lr=0.00727, betas=(0.991, 1.0000), shampoo_beta=0.996)` |
| quadratic_outliers-trivial | fp16 | MuonAdamW | No | No | ✓ | 166.36s | 1.11e-20 | 15 | 5 | `fp16-MuonAdamW(lr=0.01260, betas=(0.282, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00338, betas=(0.974, 1.0000), shampoo_beta=0.997)` |
| quadratic_outliers-trivial | fp16 | PSGDKron | No | No | ✓ | 88.09s | 6.12e-18 | 7 | 0 | `fp16-PSGDKron(lr=4.63838, betas=(0.793, 0.8961), shampoo_beta=1.000) | EMAβ=0.90(lr=0.01510, betas=(0.994, 0.9970), shampoo_beta=1.000)` |
| quadratic_quantization_aware_training-easy | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_quantization_aware_training-easy | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_quantization_aware_training-easy | fp16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_quantization_aware_training-easy | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_quantization_aware_training-easy | fp32 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_quantization_aware_training-easy | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantization_aware_training-easy | fp32 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_quantization_aware_training-extreme | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_quantization_aware_training-extreme | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_quantization_aware_training-extreme | fp16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_quantization_aware_training-extreme | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_quantization_aware_training-extreme | fp32 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_quantization_aware_training-extreme | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantization_aware_training-extreme | fp32 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_quantization_aware_training-hard | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_quantization_aware_training-hard | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_quantization_aware_training-hard | fp16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_quantization_aware_training-hard | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_quantization_aware_training-hard | fp32 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_quantization_aware_training-hard | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantization_aware_training-hard | fp32 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_quantization_aware_training-medium | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_quantization_aware_training-medium | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_quantization_aware_training-medium | fp16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_quantization_aware_training-medium | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_quantization_aware_training-medium | fp32 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_quantization_aware_training-medium | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantization_aware_training-medium | fp32 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_quantization_aware_training-nightmare | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_quantization_aware_training-nightmare | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_quantization_aware_training-nightmare | fp16 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_quantization_aware_training-nightmare | fp16 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_quantization_aware_training-nightmare | fp32 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| quadratic_quantization_aware_training-nightmare | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantization_aware_training-nightmare | fp32 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_quantization_aware_training-trivial | bf16 | AdEMAMix | No | No | ✗ | 300.16s | 9.11e-01 | 11 | 7 | `bf16-AdEMAMix(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.09335, betas=(0.996, 0.9997), shampoo_beta=0.999)` |
| quadratic_quantization_aware_training-trivial | bf16+ecc | SUDSAdamW | No | No | ✗ | 300.69s | 8.89e-01 | 9 | 3 | `bf16+ecc-SUDSAdamW(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.01105, betas=(0.997, 0.9995), shampoo_beta=0.998)` |
| quadratic_quantization_aware_training-trivial | fp16 | MuonLaProp | No | No | ✗ | 300.80s | 1.11e+00 | 11 | 5 | `fp16-MuonLaProp(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.09335, betas=(0.996, 0.9997), shampoo_beta=0.999)` |
| quadratic_quantization_aware_training-trivial | fp16 | SignLaProp | No | No | ✗ | 301.20s | 9.03e-01 | 8 | 2 | `fp16-SignLaProp(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00585, betas=(0.996, 0.9998), shampoo_beta=0.998)` |
| quadratic_quantization_aware_training-trivial | fp32 | MuonLaProp | No | No | ✗ | 300.24s | 8.89e-01 | 11 | 3 | `fp32-MuonLaProp(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.09335, betas=(0.996, 0.9997), shampoo_beta=0.999)` |
| quadratic_quantization_aware_training-trivial | fp32 | SAMWrapper | No | No | ✗ | 300.73s | 1.09e+00 | 7 | 0 | `fp32-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00238, betas=(0.995, 0.9992), shampoo_beta=1.000)` |
| quadratic_quantization_aware_training-trivial | fp32 | SOLP | No | No | ✗ | 300.15s | 9.03e-01 | 10 | 2 | `fp32-SOLP(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.04398, betas=(0.998, 0.9998), shampoo_beta=0.997)` |
| quadratic_quantized_gradient-easy | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_quantized_gradient-easy | fp16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 50 | 1 | N/A |
| quadratic_quantized_gradient-easy | fp16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantized_gradient-easy | fp16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_quantized_gradient-easy | fp32 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantized_gradient-extreme | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_quantized_gradient-extreme | fp16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| quadratic_quantized_gradient-extreme | fp16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantized_gradient-extreme | fp16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_quantized_gradient-extreme | fp32 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantized_gradient-hard | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_quantized_gradient-hard | fp16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| quadratic_quantized_gradient-hard | fp16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantized_gradient-hard | fp16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_quantized_gradient-hard | fp32 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantized_gradient-medium | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_quantized_gradient-medium | fp16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| quadratic_quantized_gradient-medium | fp16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantized_gradient-medium | fp16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_quantized_gradient-medium | fp32 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantized_gradient-nightmare | bf16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_quantized_gradient-nightmare | fp16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| quadratic_quantized_gradient-nightmare | fp16+ecc | SOAP | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantized_gradient-nightmare | fp16+ecc | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| quadratic_quantized_gradient-nightmare | fp32 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_quantized_gradient-trivial | bf16 | SAMWrapper | No | No | ✗ | 269.78s | 4.20e-10 | 50 | 6 | `bf16-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.999, 1.0000), shampoo_beta=0.999)` |
| quadratic_quantized_gradient-trivial | fp16+ecc | PSGDPRO | No | No | ✓ | 73.92s | 1.98e-15 | 3 | 1 | `fp16+ecc-PSGDPRO(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| quadratic_quantized_gradient-trivial | fp16+ecc | SOAP | No | No | ✗ | 300.34s | 4.86e-14 | 37 | 0 | `fp16+ecc-SOAP(lr=0.11929, betas=(0.092, 0.9999), shampoo_beta=1.000) | EMAβ=0.90(lr=3.86596, betas=(0.958, 0.9996), shampoo_beta=0.999)` |
| quadratic_quantized_gradient-trivial | fp16+ecc | SOAPAdEMAMix | No | No | ✗ | 300.01s | 4.79e-11 | 7 | 6 | `fp16+ecc-SOAPAdEMAMix(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00117, betas=(0.996, 0.9999), shampoo_beta=0.999)` |
| quadratic_quantized_gradient-trivial | fp32 | LATHER | No | No | ✗ | 300.33s | 1.09e-03 | 12 | 0 | `fp32-LATHER(lr=2.18378, betas=(0.993, 1.0000), shampoo_beta=0.285) | EMAβ=0.90(lr=0.32071, betas=(0.997, 1.0000), shampoo_beta=0.971)` |
| quadratic_sparse-easy | bf16 | PSGDLRA | No | No | ✓ | 266.54s | 8.30e-15 | 16 | 0 | `bf16-PSGDLRA(lr=0.00875, betas=(0.992, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00481, betas=(0.999, 1.0000), shampoo_beta=0.990)` |
| quadratic_sparse-easy | bf16+ecc | AdEMAMix | No | No | ✓ | 97.99s | 1.65e-19 | 12 | 5 | `bf16+ecc-AdEMAMix(lr=0.00361, betas=(0.825, 1.0000), shampoo_beta=0.745) | EMAβ=0.90(lr=0.00652, betas=(0.998, 1.0000), shampoo_beta=0.976)` |
| quadratic_sparse-easy | fp32 | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_sparse-extreme | bf16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_sparse-extreme | bf16+ecc | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_sparse-extreme | fp32 | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_sparse-hard | bf16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_sparse-hard | bf16+ecc | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_sparse-hard | fp32 | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_sparse-medium | bf16 | PSGDLRA | No | No | ✗ | 300.68s | 2.08e-01 | 10 | 0 | `bf16-PSGDLRA(lr=0.00216, betas=(0.999, 0.9998), shampoo_beta=0.698) | EMAβ=0.90(lr=0.00238, betas=(0.998, 0.9999), shampoo_beta=0.993)` |
| quadratic_sparse-medium | bf16+ecc | AdEMAMix | No | No | ✗ | 300.32s | 8.73e-06 | 29 | 5 | `bf16+ecc-AdEMAMix(lr=0.00274, betas=(0.997, 0.9998), shampoo_beta=0.996) | EMAβ=0.90(lr=0.00566, betas=(0.999, 0.9999), shampoo_beta=0.989)` |
| quadratic_sparse-medium | fp32 | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_sparse-nightmare | bf16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_sparse-nightmare | bf16+ecc | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_sparse-nightmare | fp32 | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| quadratic_sparse-trivial | bf16 | PSGDLRA | No | No | ✓ | 122.50s | 3.53e-16 | 3 | 0 | `bf16-PSGDLRA(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| quadratic_sparse-trivial | bf16+ecc | AdEMAMix | No | No | ✓ | 85.22s | 2.17e-15 | 9 | 5 | `bf16+ecc-AdEMAMix(lr=0.01379, betas=(0.996, 1.0000), shampoo_beta=0.976) | EMAβ=0.90(lr=0.00567, betas=(0.999, 1.0000), shampoo_beta=0.994)` |
| quadratic_sparse-trivial | fp32 | SOAPNAdam | No | No | ✗ | 301.05s | 4.28e-06 | 35 | 0 | `fp32-SOAPNAdam(lr=0.07681, betas=(1.000, 0.9994), shampoo_beta=1.000) | EMAβ=0.90(lr=0.02115, betas=(1.000, 0.9998), shampoo_beta=1.000)` |
| quadratic_varying_scale-easy | bf16 | PSGDLRA | No | No | ✓ | 111.49s | 1.00e-18 | 3 | 5 | `bf16-PSGDLRA(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| quadratic_varying_scale-easy | bf16+ecc | SAMWrapper | No | No | ✗ | 266.21s | 4.19e-10 | 50 | 5 | `bf16+ecc-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00490, betas=(0.998, 0.9994), shampoo_beta=0.999)` |
| quadratic_varying_scale-easy | fp32 | MuonLaProp | No | No | ✓ | 261.70s | 0.00e+00 | 24 | 0 | `fp32-MuonLaProp(lr=0.00288, betas=(0.717, 0.9916), shampoo_beta=0.853) | EMAβ=0.90(lr=0.03783, betas=(0.990, 0.9774), shampoo_beta=0.948)` |
| quadratic_varying_scale-extreme | bf16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_varying_scale-extreme | bf16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_varying_scale-extreme | fp32 | MuonLaProp | No | No | ✓ | 189.94s | 1.95e-15 | 12 | 0 | `fp32-MuonLaProp(lr=1.70672, betas=(0.997, 1.0000), shampoo_beta=0.999) | EMAβ=0.90(lr=0.61432, betas=(0.996, 1.0000), shampoo_beta=0.991)` |
| quadratic_varying_scale-hard | bf16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_varying_scale-hard | bf16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_varying_scale-hard | fp32 | MuonLaProp | No | No | ✓ | 190.75s | 1.94e-15 | 12 | 0 | `fp32-MuonLaProp(lr=1.70672, betas=(0.997, 1.0000), shampoo_beta=0.999) | EMAβ=0.90(lr=0.61432, betas=(0.996, 1.0000), shampoo_beta=0.991)` |
| quadratic_varying_scale-medium | bf16 | PSGDLRA | No | No | ✗ | 300.70s | 7.53e-06 | 8 | 5 | `bf16-PSGDLRA(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.01491, betas=(0.990, 0.9974), shampoo_beta=1.000)` |
| quadratic_varying_scale-medium | bf16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_varying_scale-medium | fp32 | MuonLaProp | No | No | ✓ | 76.78s | 3.80e-15 | 6 | 0 | `fp32-MuonLaProp(lr=1.59269, betas=(0.985, 1.0000), shampoo_beta=0.677) | EMAβ=0.90(lr=0.00165, betas=(0.995, 0.9998), shampoo_beta=0.999)` |
| quadratic_varying_scale-nightmare | bf16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_varying_scale-nightmare | bf16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| quadratic_varying_scale-nightmare | fp32 | MuonLaProp | No | No | ✓ | 194.93s | 1.98e-15 | 12 | 0 | `fp32-MuonLaProp(lr=1.70672, betas=(0.997, 1.0000), shampoo_beta=0.999) | EMAβ=0.90(lr=0.61432, betas=(0.996, 1.0000), shampoo_beta=0.991)` |
| quadratic_varying_scale-trivial | bf16 | PSGDLRA | No | No | ✓ | 108.18s | 2.82e-19 | 3 | 5 | `bf16-PSGDLRA(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| quadratic_varying_scale-trivial | bf16+ecc | SAMWrapper | No | No | ✓ | 3.66s | 2.52e-32 | 1 | 5 | `bf16+ecc-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| quadratic_varying_scale-trivial | fp32 | MuonLaProp | No | No | ✓ | 72.48s | 3.29e-16 | 6 | 0 | `fp32-MuonLaProp(lr=2.17510, betas=(0.970, 1.0000), shampoo_beta=0.990) | EMAβ=0.90(lr=0.00176, betas=(0.994, 0.9998), shampoo_beta=0.999)` |
| quadratic_varying_target-easy | bf16+ecc | AdamW | No | No | ✗ | 300.10s | 2.06e-06 | 12 | 7 | `bf16+ecc-AdamW(lr=0.00684, betas=(0.988, 1.0000), shampoo_beta=0.916) | EMAβ=0.90(lr=0.00127, betas=(0.999, 1.0000), shampoo_beta=0.985)` |
| quadratic_varying_target-easy | bf16+ecc | SUDSAdamW | No | No | ✗ | 300.86s | 9.13e-13 | 13 | 2 | `bf16+ecc-SUDSAdamW(lr=0.43421, betas=(0.884, 1.0000), shampoo_beta=0.923) | EMAβ=0.90(lr=0.08747, betas=(0.973, 1.0000), shampoo_beta=0.989)` |
| quadratic_varying_target-easy | fp16+ecc | SOAPNAdam | No | No | ✗ | 300.60s | 6.19e-16 | 20 | 1 | `fp16+ecc-SOAPNAdam(lr=0.01098, betas=(0.521, 1.0000), shampoo_beta=0.999) | EMAβ=0.90(lr=0.02558, betas=(0.943, 1.0000), shampoo_beta=0.999)` |
| quadratic_varying_target-extreme | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_varying_target-extreme | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_varying_target-extreme | fp16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| quadratic_varying_target-hard | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_varying_target-hard | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_varying_target-hard | fp16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| quadratic_varying_target-medium | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_varying_target-medium | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_varying_target-medium | fp16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| quadratic_varying_target-nightmare | bf16+ecc | AdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| quadratic_varying_target-nightmare | bf16+ecc | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| quadratic_varying_target-nightmare | fp16+ecc | SOAPNAdam | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| quadratic_varying_target-trivial | bf16+ecc | AdamW | No | No | ✓ | 141.64s | 5.67e-31 | 7 | 7 | `bf16+ecc-AdamW(lr=0.36845, betas=(0.962, 1.0000), shampoo_beta=0.923) | EMAβ=0.90(lr=0.00384, betas=(0.998, 1.0000), shampoo_beta=0.997)` |
| quadratic_varying_target-trivial | bf16+ecc | SUDSAdamW | No | No | ✓ | 287.86s | 3.95e-21 | 7 | 2 | `bf16+ecc-SUDSAdamW(lr=31.01424, betas=(0.732, 1.0000), shampoo_beta=0.920) | EMAβ=0.90(lr=0.01093, betas=(0.993, 1.0000), shampoo_beta=0.997)` |
| quadratic_varying_target-trivial | fp16+ecc | SOAPNAdam | No | No | ✓ | 211.41s | 3.04e-18 | 14 | 1 | `fp16+ecc-SOAPNAdam(lr=0.00665, betas=(0.690, 1.0000), shampoo_beta=0.441) | EMAβ=0.90(lr=0.01645, betas=(0.982, 1.0000), shampoo_beta=0.899)` |
| rastrigin-easy | bf16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| rastrigin-easy | bf16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rastrigin-easy | fp16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| rastrigin-easy | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rastrigin-extreme | bf16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| rastrigin-extreme | bf16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rastrigin-extreme | fp16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| rastrigin-extreme | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rastrigin-hard | bf16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| rastrigin-hard | bf16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rastrigin-hard | fp16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| rastrigin-hard | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rastrigin-medium | bf16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| rastrigin-medium | bf16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rastrigin-medium | fp16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| rastrigin-medium | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rastrigin-nightmare | bf16 | RMSprop | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| rastrigin-nightmare | bf16+ecc | LATHER | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rastrigin-nightmare | fp16 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| rastrigin-nightmare | fp32 | SGD | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rastrigin-trivial | bf16 | RMSprop | No | No | ✗ | 300.17s | 3.98e+00 | 12 | 7 | `bf16-RMSprop(lr=0.00056, betas=(1.000, 0.9990), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00036, betas=(1.000, 1.0000), shampoo_beta=0.995)` |
| rastrigin-trivial | bf16+ecc | LATHER | No | No | ✓ | 70.95s | 4.22e+00 | 3 | 4 | `bf16+ecc-LATHER(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| rastrigin-trivial | fp16 | PSGDLRA | No | No | ✗ | 300.07s | 3.99e+00 | 12 | 0 | `fp16-PSGDLRA(lr=0.00971, betas=(1.000, 1.0000), shampoo_beta=0.984) | EMAβ=0.90(lr=0.00097, betas=(1.000, 1.0000), shampoo_beta=0.998)` |
| rastrigin-trivial | fp32 | SGD | No | No | ✓ | 32.99s | 4.47e+00 | 4 | 4 | `fp32-SGD(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| rosenbrock-easy | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| rosenbrock-easy | bf16 | Muon | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| rosenbrock-easy | bf16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rosenbrock-easy | fp16 | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| rosenbrock-extreme | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| rosenbrock-extreme | bf16 | Muon | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| rosenbrock-extreme | bf16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rosenbrock-extreme | fp16 | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| rosenbrock-hard | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| rosenbrock-hard | bf16 | Muon | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| rosenbrock-hard | bf16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rosenbrock-hard | fp16 | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| rosenbrock-medium | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| rosenbrock-medium | bf16 | Muon | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| rosenbrock-medium | bf16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rosenbrock-medium | fp16 | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| rosenbrock-nightmare | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| rosenbrock-nightmare | bf16 | Muon | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| rosenbrock-nightmare | bf16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| rosenbrock-nightmare | fp16 | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| rosenbrock-trivial | bf16 | AdEMAMix | No | No | ✓ | 104.55s | 2.79e-12 | 6 | 0 | `bf16-AdEMAMix(lr=0.50881, betas=(0.939, 1.0000), shampoo_beta=0.659) | EMAβ=0.90(lr=0.00129, betas=(0.993, 0.9998), shampoo_beta=0.999)` |
| rosenbrock-trivial | bf16 | Muon | No | No | ✗ | 155.92s | 4.12e-04 | 50 | 2 | `bf16-Muon(lr=0.01453, betas=(1.000, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.12808, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| rosenbrock-trivial | bf16+ecc | AdamC | No | No | ✓ | 111.38s | 2.40e-12 | 7 | 4 | `bf16+ecc-AdamC(lr=1.10327, betas=(0.931, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00546, betas=(0.993, 0.9997), shampoo_beta=1.000)` |
| rosenbrock-trivial | fp16 | ADOPT | No | No | ✓ | 199.06s | 4.59e-11 | 11 | 6 | `fp16-ADOPT(lr=0.01071, betas=(0.990, 1.0000), shampoo_beta=0.999) | EMAβ=0.90(lr=0.00258, betas=(0.994, 0.9998), shampoo_beta=1.000)` |
| saddle_point-easy | bf16 | PSGDPRO | No | No | ✓ | 53.73s | -5.54e+06 | 4 | 4 | `bf16-PSGDPRO(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| saddle_point-easy | fp16+ecc | SOLP | No | No | ✓ | 57.10s | -1.30e+08 | 3 | 3 | `fp16+ecc-SOLP(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| saddle_point-easy | fp32 | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| saddle_point-easy | fp32 | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| saddle_point-extreme | bf16 | PSGDPRO | No | No | ✗ | 50.82s | inf | 50 | 4 | `bf16-PSGDPRO(lr=nan, betas=(nan, nan), shampoo_beta=nan) | EMAβ=0.90(lr=0.00224, betas=(0.998, 0.9996), shampoo_beta=1.000)` |
| saddle_point-extreme | fp16+ecc | SOLP | No | No | ✗ | 151.55s | 1.51e+07 | 50 | 3 | `fp16+ecc-SOLP(lr=0.32192, betas=(0.996, 1.0000), shampoo_beta=0.442) | EMAβ=0.90(lr=3.12621, betas=(1.000, 1.0000), shampoo_beta=0.479)` |
| saddle_point-extreme | fp32 | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| saddle_point-extreme | fp32 | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| saddle_point-hard | bf16 | PSGDPRO | No | No | ✓ | 54.79s | -7.73e+03 | 14 | 4 | `bf16-PSGDPRO(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00151, betas=(0.998, 0.9999), shampoo_beta=1.000)` |
| saddle_point-hard | fp16+ecc | SOLP | No | No | ✓ | 54.16s | -2.04e+14 | 3 | 3 | `fp16+ecc-SOLP(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| saddle_point-hard | fp32 | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| saddle_point-hard | fp32 | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| saddle_point-medium | bf16 | PSGDPRO | No | No | ✓ | 162.30s | -1.16e+00 | 14 | 4 | `bf16-PSGDPRO(lr=0.00003, betas=(1.000, 0.9983), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00245, betas=(0.998, 0.9997), shampoo_beta=1.000)` |
| saddle_point-medium | fp16+ecc | SOLP | No | No | ✓ | 55.06s | -3.51e+07 | 3 | 3 | `fp16+ecc-SOLP(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| saddle_point-medium | fp32 | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| saddle_point-medium | fp32 | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| saddle_point-nightmare | bf16 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| saddle_point-nightmare | fp16+ecc | SOLP | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| saddle_point-nightmare | fp32 | ADOPT | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| saddle_point-nightmare | fp32 | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| saddle_point-trivial | bf16 | PSGDPRO | No | No | ✓ | 62.27s | -8.03e+02 | 4 | 4 | `bf16-PSGDPRO(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| saddle_point-trivial | fp16+ecc | SOLP | No | No | ✓ | 65.18s | -8.34e+05 | 3 | 3 | `fp16+ecc-SOLP(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| saddle_point-trivial | fp32 | ADOPT | No | No | ✗ | 161.32s | 1.00e+01 | 50 | 4 | `fp32-ADOPT(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=25.85368, betas=(0.989, 0.9577), shampoo_beta=0.155)` |
| saddle_point-trivial | fp32 | OrthoLaProp | No | No | ✗ | 300.61s | 1.72e+01 | 17 | 7 | `fp32-OrthoLaProp(lr=0.00006, betas=(0.978, 0.9999), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00002, betas=(0.996, 0.9999), shampoo_beta=1.000)` |
| scale_invariant-easy | bf16 | SUDSAdamW | No | No | ✓ | 111.86s | 7.52e-04 | 6 | 1 | `bf16-SUDSAdamW(lr=0.57652, betas=(0.999, 0.9996), shampoo_beta=0.876) | EMAβ=0.90(lr=0.00132, betas=(0.997, 0.9997), shampoo_beta=0.999)` |
| scale_invariant-easy | bf16+ecc | AdamC | No | No | ✓ | 85.35s | 6.59e-04 | 6 | 6 | `bf16+ecc-AdamC(lr=0.57652, betas=(0.999, 0.9996), shampoo_beta=0.876) | EMAβ=0.90(lr=0.00132, betas=(0.997, 0.9997), shampoo_beta=0.999)` |
| scale_invariant-easy | fp16+ecc | SFAdamW | No | No | ✗ | 300.28s | 3.27e-03 | 11 | 1 | `fp16+ecc-SFAdamW(lr=0.05303, betas=(0.781, 0.0647), shampoo_beta=0.999) | EMAβ=0.90(lr=0.01895, betas=(0.986, 0.9823), shampoo_beta=1.000)` |
| scale_invariant-easy | fp32 | LATHER | No | No | ✓ | 139.82s | 6.50e-04 | 6 | 0 | `fp32-LATHER(lr=0.57652, betas=(0.999, 0.9996), shampoo_beta=0.876) | EMAβ=0.90(lr=0.00132, betas=(0.997, 0.9997), shampoo_beta=0.999)` |
| scale_invariant-easy | fp32 | PSGDLRA | No | No | ✗ | 300.57s | 1.35e-01 | 8 | 4 | `fp32-PSGDLRA(lr=0.38752, betas=(0.999, 1.0000), shampoo_beta=0.994) | EMAβ=0.90(lr=0.00512, betas=(0.997, 0.9999), shampoo_beta=0.998)` |
| scale_invariant-easy | fp32 | SFAdamW | No | No | ✓ | 251.27s | 2.42e-04 | 14 | 7 | `fp32-SFAdamW(lr=0.60187, betas=(0.949, 0.5871), shampoo_beta=1.000) | EMAβ=0.90(lr=0.19614, betas=(0.952, 0.9520), shampoo_beta=1.000)` |
| scale_invariant-extreme | bf16 | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| scale_invariant-extreme | bf16+ecc | AdamC | No | No | ✗ | 300.13s | 1.66e-01 | 18 | 6 | `bf16+ecc-AdamC(lr=14.11863, betas=(0.923, 0.9967), shampoo_beta=1.000) | EMAβ=0.90(lr=1.39564, betas=(0.963, 0.9745), shampoo_beta=1.000)` |
| scale_invariant-extreme | fp16+ecc | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| scale_invariant-extreme | fp32 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| scale_invariant-extreme | fp32 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| scale_invariant-extreme | fp32 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| scale_invariant-hard | bf16 | SUDSAdamW | No | No | ✗ | 300.23s | 1.00e-01 | 15 | 1 | `bf16-SUDSAdamW(lr=47.43117, betas=(0.989, 0.9864), shampoo_beta=0.967) | EMAβ=0.90(lr=1.68695, betas=(0.981, 0.9873), shampoo_beta=0.999)` |
| scale_invariant-hard | bf16+ecc | AdamC | No | No | ✓ | 273.73s | 7.84e-04 | 19 | 6 | `bf16+ecc-AdamC(lr=5.07264, betas=(0.994, 0.9905), shampoo_beta=0.595) | EMAβ=0.90(lr=2.67637, betas=(0.977, 0.9756), shampoo_beta=0.999)` |
| scale_invariant-hard | fp16+ecc | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| scale_invariant-hard | fp32 | LATHER | No | No | ✗ | 300.40s | 1.85e-01 | 13 | 0 | `fp32-LATHER(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.23724, betas=(0.985, 0.9877), shampoo_beta=0.999)` |
| scale_invariant-hard | fp32 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| scale_invariant-hard | fp32 | SFAdamW | No | No | ✗ | 300.38s | 3.13e-02 | 18 | 7 | `fp32-SFAdamW(lr=28.70192, betas=(0.993, 0.0896), shampoo_beta=0.999) | EMAβ=0.90(lr=1.61712, betas=(0.989, 0.8649), shampoo_beta=1.000)` |
| scale_invariant-medium | bf16 | SUDSAdamW | No | No | ✓ | 122.97s | 8.46e-05 | 6 | 1 | `bf16-SUDSAdamW(lr=1.91553, betas=(0.995, 0.9998), shampoo_beta=0.997) | EMAβ=0.90(lr=0.00171, betas=(0.996, 0.9997), shampoo_beta=1.000)` |
| scale_invariant-medium | bf16+ecc | AdamC | No | No | ✓ | 87.10s | 2.70e-04 | 6 | 6 | `bf16+ecc-AdamC(lr=2.53540, betas=(0.996, 0.9992), shampoo_beta=0.956) | EMAβ=0.90(lr=0.00182, betas=(0.996, 0.9996), shampoo_beta=0.999)` |
| scale_invariant-medium | fp16+ecc | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| scale_invariant-medium | fp32 | LATHER | No | No | ✓ | 126.10s | 7.16e-04 | 6 | 0 | `fp32-LATHER(lr=2.53540, betas=(0.996, 0.9992), shampoo_beta=0.956) | EMAβ=0.90(lr=0.00182, betas=(0.996, 0.9996), shampoo_beta=0.999)` |
| scale_invariant-medium | fp32 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| scale_invariant-medium | fp32 | SFAdamW | No | No | ✓ | 98.33s | 9.87e-05 | 7 | 7 | `fp32-SFAdamW(lr=11.20749, betas=(0.998, 0.6199), shampoo_beta=1.000) | EMAβ=0.90(lr=0.01708, betas=(0.996, 0.9977), shampoo_beta=1.000)` |
| scale_invariant-nightmare | bf16 | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| scale_invariant-nightmare | bf16+ecc | AdamC | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| scale_invariant-nightmare | fp16+ecc | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| scale_invariant-nightmare | fp32 | LATHER | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| scale_invariant-nightmare | fp32 | PSGDLRA | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| scale_invariant-nightmare | fp32 | SFAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| scale_invariant-trivial | bf16 | SUDSAdamW | No | No | ✓ | 114.09s | 7.73e-04 | 6 | 1 | `bf16-SUDSAdamW(lr=0.57652, betas=(0.999, 0.9996), shampoo_beta=0.876) | EMAβ=0.90(lr=0.00132, betas=(0.997, 0.9997), shampoo_beta=0.999)` |
| scale_invariant-trivial | bf16+ecc | AdamC | No | No | ✓ | 92.50s | 3.60e-04 | 6 | 6 | `bf16+ecc-AdamC(lr=0.43366, betas=(0.997, 1.0000), shampoo_beta=0.967) | EMAβ=0.90(lr=0.00125, betas=(0.996, 0.9999), shampoo_beta=0.999)` |
| scale_invariant-trivial | fp16+ecc | SFAdamW | No | No | ✓ | 143.88s | 1.90e-36 | 6 | 1 | `fp16+ecc-SFAdamW(lr=1.33207, betas=(0.978, 0.9997), shampoo_beta=0.985) | EMAβ=0.90(lr=0.00158, betas=(0.995, 0.9997), shampoo_beta=0.999)` |
| scale_invariant-trivial | fp32 | LATHER | No | No | ✓ | 140.35s | 6.63e-04 | 6 | 0 | `fp32-LATHER(lr=0.64018, betas=(0.998, 1.0000), shampoo_beta=0.738) | EMAβ=0.90(lr=0.00135, betas=(0.997, 0.9998), shampoo_beta=0.999)` |
| scale_invariant-trivial | fp32 | PSGDLRA | No | No | ✓ | 262.74s | 4.10e-05 | 6 | 4 | `fp32-PSGDLRA(lr=1.33207, betas=(0.978, 0.9997), shampoo_beta=0.985) | EMAβ=0.90(lr=0.00158, betas=(0.995, 0.9997), shampoo_beta=0.999)` |
| scale_invariant-trivial | fp32 | SFAdamW | No | No | ✓ | 108.55s | 1.77e-37 | 6 | 7 | `fp32-SFAdamW(lr=1.33207, betas=(0.978, 0.9997), shampoo_beta=0.985) | EMAβ=0.90(lr=0.00158, betas=(0.995, 0.9997), shampoo_beta=0.999)` |
| sparse_gradient-easy | bf16 | RMSprop | No | No | ✓ | 48.63s | 4.33e-05 | 4 | 1 | `bf16-RMSprop(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| sparse_gradient-easy | bf16 | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| sparse_gradient-easy | fp16+ecc | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| sparse_gradient-extreme | bf16 | RMSprop | No | No | ✓ | 15.80s | 8.11e-05 | 4 | 1 | `bf16-RMSprop(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| sparse_gradient-extreme | bf16 | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| sparse_gradient-extreme | fp16+ecc | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| sparse_gradient-hard | bf16 | RMSprop | No | No | ✓ | 41.54s | 6.76e-05 | 4 | 1 | `bf16-RMSprop(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| sparse_gradient-hard | bf16 | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| sparse_gradient-hard | fp16+ecc | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| sparse_gradient-medium | bf16 | RMSprop | No | No | ✓ | 40.37s | 7.04e-05 | 4 | 1 | `bf16-RMSprop(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| sparse_gradient-medium | bf16 | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| sparse_gradient-medium | fp16+ecc | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| sparse_gradient-nightmare | bf16 | RMSprop | No | No | ✓ | 1.78s | 3.68e-07 | 1 | 1 | `bf16-RMSprop(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| sparse_gradient-nightmare | bf16 | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| sparse_gradient-nightmare | fp16+ecc | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| sparse_gradient-trivial | bf16 | RMSprop | No | No | ✓ | 54.32s | 2.11e-05 | 4 | 1 | `bf16-RMSprop(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| sparse_gradient-trivial | bf16 | SOAPAdEMAMix | No | No | ✗ | 0.87s | inf | 50 | 1 | `bf16-SOAPAdEMAMix(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| sparse_gradient-trivial | fp16+ecc | AdEMAMix | No | No | ✗ | 300.05s | 7.02e-04 | 27 | 6 | `fp16+ecc-AdEMAMix(lr=0.00000, betas=(1.000, 0.8733), shampoo_beta=0.626) | EMAβ=0.90(lr=0.00000, betas=(1.000, 0.9952), shampoo_beta=0.983)` |
| sparse_parameters-easy | bf16 | Scion | No | No | ✓ | 1.62s | 6.38e-16 | 1 | 7 | `bf16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| sparse_parameters-easy | fp32 | PSGDKron | No | No | ✓ | 111.68s | 4.79e-05 | 6 | 7 | `fp32-PSGDKron(lr=3.00416, betas=(1.000, 0.0682), shampoo_beta=0.986) | EMAβ=0.90(lr=0.00188, betas=(0.998, 0.9984), shampoo_beta=0.999)` |
| sparse_parameters-extreme | bf16 | Scion | No | No | ✓ | 1.63s | 6.38e-16 | 1 | 7 | `bf16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| sparse_parameters-extreme | fp32 | PSGDKron | No | No | ✓ | 65.65s | 8.87e-05 | 3 | 7 | `fp32-PSGDKron(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| sparse_parameters-hard | bf16 | Scion | No | No | ✓ | 1.62s | 6.38e-16 | 1 | 7 | `bf16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| sparse_parameters-hard | fp32 | PSGDKron | No | No | ✓ | 76.20s | 1.90e-04 | 4 | 7 | `fp32-PSGDKron(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| sparse_parameters-medium | bf16 | Scion | No | No | ✓ | 1.62s | 6.38e-16 | 1 | 7 | `bf16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| sparse_parameters-medium | fp32 | PSGDKron | No | No | ✓ | 106.88s | 7.71e-05 | 6 | 7 | `fp32-PSGDKron(lr=3.00416, betas=(1.000, 0.0682), shampoo_beta=0.986) | EMAβ=0.90(lr=0.00188, betas=(0.998, 0.9984), shampoo_beta=0.999)` |
| sparse_parameters-nightmare | bf16 | Scion | No | No | ✓ | 1.62s | 6.38e-16 | 1 | 7 | `bf16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| sparse_parameters-nightmare | fp32 | PSGDKron | No | No | ✓ | 1.25s | 6.22e-05 | 1 | 7 | `fp32-PSGDKron(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| sparse_parameters-trivial | bf16 | Scion | No | No | ✓ | 6.53s | 6.38e-16 | 1 | 7 | `bf16-Scion(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| sparse_parameters-trivial | fp32 | PSGDKron | No | No | ✓ | 144.18s | 1.66e-10 | 7 | 7 | `fp32-PSGDKron(lr=15.78945, betas=(1.000, 0.2167), shampoo_beta=1.000) | EMAβ=0.90(lr=0.01064, betas=(0.999, 0.9946), shampoo_beta=1.000)` |
| spiral-easy | bf16 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| spiral-easy | bf16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| spiral-easy | fp16 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| spiral-easy | fp32 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| spiral-easy | fp32 | Scion | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| spiral-extreme | bf16 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| spiral-extreme | bf16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| spiral-extreme | fp16 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| spiral-extreme | fp32 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| spiral-extreme | fp32 | Scion | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| spiral-hard | bf16 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| spiral-hard | bf16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| spiral-hard | fp16 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| spiral-hard | fp32 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| spiral-hard | fp32 | Scion | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| spiral-medium | bf16 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| spiral-medium | bf16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| spiral-medium | fp16 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| spiral-medium | fp32 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| spiral-medium | fp32 | Scion | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| spiral-nightmare | bf16 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| spiral-nightmare | bf16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| spiral-nightmare | fp16 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| spiral-nightmare | fp32 | MuonLaProp | No | No | ✗ | 0.00s | inf | 0 | 6 | `N/A` |
| spiral-nightmare | fp32 | Scion | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| spiral-trivial | bf16 | PSGDPRO | No | No | ✓ | 147.19s | 4.45e-09 | 3 | 2 | `bf16-PSGDPRO(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| spiral-trivial | bf16+ecc | SGD | No | No | ✗ | 301.38s | 5.07e-01 | 16 | 1 | `bf16+ecc-SGD(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(1.000, 0.9995), shampoo_beta=1.000)` |
| spiral-trivial | fp16 | PSGDKron | No | No | ✓ | 162.20s | 5.04e-10 | 3 | 5 | `fp16-PSGDKron(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| spiral-trivial | fp32 | MuonLaProp | No | No | ✗ | 301.27s | 6.20e-05 | 10 | 6 | `fp32-MuonLaProp(lr=0.17612, betas=(0.968, 0.1490), shampoo_beta=1.000) | EMAβ=0.90(lr=0.03438, betas=(0.989, 0.9841), shampoo_beta=1.000)` |
| spiral-trivial | fp32 | Scion | No | No | ✓ | 273.45s | 4.04e-09 | 9 | 0 | `fp32-Scion(lr=0.00008, betas=(0.663, 0.9986), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00006, betas=(0.993, 0.9995), shampoo_beta=1.000)` |
| transfer_domain_shift-easy | bf16 | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| transfer_domain_shift-easy | fp16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| transfer_domain_shift-extreme | bf16 | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| transfer_domain_shift-extreme | fp16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| transfer_domain_shift-hard | bf16 | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| transfer_domain_shift-hard | fp16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| transfer_domain_shift-medium | bf16 | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| transfer_domain_shift-medium | fp16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| transfer_domain_shift-nightmare | bf16 | SOAPAdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| transfer_domain_shift-nightmare | fp16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| transfer_domain_shift-trivial | bf16 | SOAPAdEMAMix | No | No | ✗ | 303.73s | 1.37e+00 | 2 | 3 | `bf16-SOAPAdEMAMix(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| transfer_domain_shift-trivial | fp16+ecc | PSGDPRO | No | No | ✗ | 305.71s | inf | 1 | 2 | `fp16+ecc-PSGDPRO(lr=nan, betas=(nan, nan), shampoo_beta=nan)` |
| vanishing_gradient-easy | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| vanishing_gradient-easy | fp32 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| vanishing_gradient-extreme | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| vanishing_gradient-extreme | fp32 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| vanishing_gradient-hard | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| vanishing_gradient-hard | fp32 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| vanishing_gradient-medium | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| vanishing_gradient-medium | fp32 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| vanishing_gradient-nightmare | fp16+ecc | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| vanishing_gradient-nightmare | fp32 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| vanishing_gradient-trivial | fp16+ecc | PSGDKron | No | No | ✗ | 300.04s | 0.00e+00 | 11 | 5 | `fp16+ecc-PSGDKron(lr=0.32192, betas=(0.996, 1.0000), shampoo_beta=0.442) | EMAβ=0.90(lr=0.07929, betas=(0.994, 0.9999), shampoo_beta=0.998)` |
| vanishing_gradient-trivial | fp32 | MSAMLaProp | No | No | ✗ | 300.72s | 6.79e-278 | 12 | 1 | `fp32-MSAMLaProp(lr=95.90648, betas=(1.000, 1.0000), shampoo_beta=0.994) | EMAβ=0.90(lr=0.70309, betas=(0.999, 0.9987), shampoo_beta=0.998)` |
| wide_linear-easy | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| wide_linear-easy | bf16 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| wide_linear-easy | bf16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| wide_linear-easy | fp16 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| wide_linear-easy | fp16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| wide_linear-easy | fp32 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| wide_linear-easy | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| wide_linear-extreme | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| wide_linear-extreme | bf16 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| wide_linear-extreme | bf16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| wide_linear-extreme | fp16 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| wide_linear-extreme | fp16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| wide_linear-extreme | fp32 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| wide_linear-extreme | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| wide_linear-hard | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| wide_linear-hard | bf16 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| wide_linear-hard | bf16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| wide_linear-hard | fp16 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| wide_linear-hard | fp16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| wide_linear-hard | fp32 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| wide_linear-hard | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| wide_linear-medium | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| wide_linear-medium | bf16 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| wide_linear-medium | bf16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| wide_linear-medium | fp16 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| wide_linear-medium | fp16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| wide_linear-medium | fp32 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| wide_linear-medium | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| wide_linear-nightmare | bf16 | AdEMAMix | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| wide_linear-nightmare | bf16 | SOAP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| wide_linear-nightmare | bf16+ecc | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| wide_linear-nightmare | fp16 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| wide_linear-nightmare | fp16 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| wide_linear-nightmare | fp32 | PSGDPRO | No | No | ✗ | 0.00s | inf | 0 | 0 | `N/A` |
| wide_linear-nightmare | fp32 | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| wide_linear-trivial | bf16 | AdEMAMix | No | No | ✗ | 300.09s | 2.78e-16 | 27 | 5 | `bf16-AdEMAMix(lr=0.01974, betas=(0.262, 1.0000), shampoo_beta=0.902) | EMAβ=0.90(lr=0.00419, betas=(0.861, 1.0000), shampoo_beta=0.911)` |
| wide_linear-trivial | bf16 | SOAP | No | No | ✗ | 269.20s | 1.20e-08 | 50 | 2 | `bf16-SOAP(lr=0.00115, betas=(0.963, 0.9930), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00032, betas=(0.998, 0.9987), shampoo_beta=1.000)` |
| wide_linear-trivial | bf16+ecc | PSGDPRO | No | No | ✗ | 284.39s | 1.22e-15 | 50 | 3 | `bf16+ecc-PSGDPRO(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=66.81465, betas=(0.799, 0.8927), shampoo_beta=0.968)` |
| wide_linear-trivial | fp16 | PSGDPRO | No | No | ✗ | 175.76s | 2.35e-15 | 50 | 0 | `fp16-PSGDPRO(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=85.14379, betas=(0.999, 0.7087), shampoo_beta=1.000)` |
| wide_linear-trivial | fp16 | SAMWrapper | No | No | ✗ | 301.30s | 2.23e-09 | 28 | 3 | `fp16-SAMWrapper(lr=8.96574, betas=(0.983, 0.9720), shampoo_beta=0.995) | EMAβ=0.90(lr=2.26727, betas=(0.984, 0.9903), shampoo_beta=0.992)` |
| wide_linear-trivial | fp32 | PSGDPRO | No | No | ✗ | 254.07s | 2.44e-15 | 50 | 0 | `fp32-PSGDPRO(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=1.000) | EMAβ=0.90(lr=66.81465, betas=(0.799, 0.8927), shampoo_beta=0.968)` |
| wide_linear-trivial | fp32 | SAMWrapper | No | No | ✗ | 300.60s | 7.83e-08 | 39 | 5 | `fp32-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 0.9999), shampoo_beta=1.000)` |
| xor_digit-easy | bf16+ecc | SAMWrapper | No | No | ✓ | 14.27s | 4.21e-04 | 1 | 7 | `bf16+ecc-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| xor_digit-easy | fp16 | MSAMLaProp | No | No | ✓ | 158.48s | 7.26e-04 | 2 | 1 | `fp16-MSAMLaProp(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| xor_digit-easy | fp16 | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| xor_digit-easy | fp16+ecc | SAMWrapper | No | No | ✓ | 14.46s | 4.21e-04 | 1 | 1 | `fp16+ecc-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| xor_digit-extreme | bf16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| xor_digit-extreme | fp16 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_digit-extreme | fp16 | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| xor_digit-extreme | fp16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_digit-hard | bf16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| xor_digit-hard | fp16 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_digit-hard | fp16 | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| xor_digit-hard | fp16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_digit-medium | bf16+ecc | SAMWrapper | No | No | ✗ | 303.70s | 6.92e-01 | 3 | 7 | `bf16+ecc-SAMWrapper(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| xor_digit-medium | fp16 | MSAMLaProp | No | No | ✗ | 302.12s | 5.80e-01 | 6 | 1 | `fp16-MSAMLaProp(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00025, betas=(0.996, 0.9997), shampoo_beta=1.000)` |
| xor_digit-medium | fp16 | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| xor_digit-medium | fp16+ecc | SAMWrapper | No | No | ✗ | 303.62s | 6.93e-01 | 3 | 1 | `fp16+ecc-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| xor_digit-nightmare | bf16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| xor_digit-nightmare | fp16 | MSAMLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_digit-nightmare | fp16 | SUDSAdamW | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| xor_digit-nightmare | fp16+ecc | SAMWrapper | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_digit-trivial | bf16+ecc | SAMWrapper | No | No | ✓ | 14.99s | 6.41e-04 | 1 | 7 | `bf16+ecc-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| xor_digit-trivial | fp16 | MSAMLaProp | No | No | ✓ | 115.44s | 7.29e-04 | 2 | 1 | `fp16-MSAMLaProp(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| xor_digit-trivial | fp16 | SUDSAdamW | No | No | ✗ | 300.20s | 2.84e-02 | 5 | 2 | `fp16-SUDSAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_digit-trivial | fp16+ecc | SAMWrapper | No | No | ✓ | 15.35s | 6.41e-04 | 1 | 1 | `fp16+ecc-SAMWrapper(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| xor_digit_rnn-easy | fp16 | SOLP | No | No | ✗ | 300.55s | 4.97e-01 | 5 | 2 | `fp16-SOLP(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_digit_rnn-easy | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| xor_digit_rnn-easy | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| xor_digit_rnn-extreme | fp16 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| xor_digit_rnn-extreme | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| xor_digit_rnn-extreme | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| xor_digit_rnn-hard | fp16 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| xor_digit_rnn-hard | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| xor_digit_rnn-hard | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| xor_digit_rnn-medium | fp16 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| xor_digit_rnn-medium | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| xor_digit_rnn-medium | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| xor_digit_rnn-nightmare | fp16 | SOLP | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| xor_digit_rnn-nightmare | fp16+ecc | LaProp | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| xor_digit_rnn-nightmare | fp32 | PSGDKron | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| xor_digit_rnn-trivial | fp16 | SOLP | No | No | ✓ | 200.03s | 2.22e-06 | 4 | 2 | `fp16-SOLP(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_digit_rnn-trivial | fp16+ecc | LaProp | No | No | ✗ | 301.68s | 6.91e-01 | 2 | 4 | `fp16+ecc-LaProp(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| xor_digit_rnn-trivial | fp32 | PSGDKron | No | No | ✗ | 300.87s | 7.23e-01 | 2 | 4 | `fp32-PSGDKron(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| xor_sequence_rnn-easy | bf16+ecc | SignLaProp | No | No | ✗ | 300.11s | 2.43e-01 | 7 | 1 | `bf16+ecc-SignLaProp(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00237, betas=(0.999, 0.9999), shampoo_beta=0.999)` |
| xor_sequence_rnn-easy | fp16+ecc | NAdam | No | No | ✓ | 192.77s | 4.61e-04 | 4 | 4 | `fp16+ecc-NAdam(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_sequence_rnn-easy | fp16+ecc | SGD | No | No | ✓ | 159.09s | 9.06e-03 | 2 | 7 | `fp16+ecc-SGD(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| xor_sequence_rnn-easy | fp32 | AdamC | No | No | ✓ | 258.09s | 3.28e-03 | 7 | 5 | `fp32-AdamC(lr=0.00319, betas=(0.999, 1.0000), shampoo_beta=0.555) | EMAβ=0.90(lr=0.00251, betas=(0.999, 1.0000), shampoo_beta=0.998)` |
| xor_sequence_rnn-extreme | bf16+ecc | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_sequence_rnn-extreme | fp16+ecc | NAdam | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| xor_sequence_rnn-extreme | fp16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| xor_sequence_rnn-extreme | fp32 | AdamC | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| xor_sequence_rnn-hard | bf16+ecc | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_sequence_rnn-hard | fp16+ecc | NAdam | No | No | ✗ | 301.09s | 6.64e-01 | 5 | 4 | `fp16+ecc-NAdam(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_sequence_rnn-hard | fp16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| xor_sequence_rnn-hard | fp32 | AdamC | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| xor_sequence_rnn-medium | bf16+ecc | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_sequence_rnn-medium | fp16+ecc | NAdam | No | No | ✓ | 194.31s | 3.36e-03 | 4 | 4 | `fp16+ecc-NAdam(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_sequence_rnn-medium | fp16+ecc | SGD | No | No | ✗ | 300.50s | 6.93e-01 | 6 | 7 | `fp16+ecc-SGD(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00025, betas=(0.996, 0.9997), shampoo_beta=1.000)` |
| xor_sequence_rnn-medium | fp32 | AdamC | No | No | ✗ | 301.63s | 6.73e-01 | 5 | 5 | `fp32-AdamC(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_sequence_rnn-nightmare | bf16+ecc | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_sequence_rnn-nightmare | fp16+ecc | NAdam | No | No | ✗ | 0.00s | inf | 0 | 4 | `N/A` |
| xor_sequence_rnn-nightmare | fp16+ecc | SGD | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| xor_sequence_rnn-nightmare | fp32 | AdamC | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| xor_sequence_rnn-trivial | bf16+ecc | SignLaProp | No | No | ✓ | 224.06s | 9.08e-08 | 4 | 1 | `bf16+ecc-SignLaProp(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_sequence_rnn-trivial | fp16+ecc | NAdam | No | No | ✓ | 228.48s | 1.97e-04 | 4 | 4 | `fp16+ecc-NAdam(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_sequence_rnn-trivial | fp16+ecc | SGD | No | No | ✓ | 117.09s | 9.98e-03 | 2 | 7 | `fp16+ecc-SGD(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| xor_sequence_rnn-trivial | fp32 | AdamC | No | No | ✓ | 189.50s | 1.09e-10 | 4 | 5 | `fp32-AdamC(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_spot-easy | fp16+ecc | HyperBallAdamW | No | No | ✗ | 301.63s | 7.04e-01 | 2 | 1 | `fp16+ecc-HyperBallAdamW(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| xor_spot-easy | fp32 | SignLaProp | No | No | ✗ | 301.84s | 6.95e-01 | 4 | 3 | `fp32-SignLaProp(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000) | EMAβ=0.90(lr=0.00069, betas=(0.999, 0.9989), shampoo_beta=1.000)` |
| xor_spot-easy | fp32 | UnscaledAdamW | No | No | ✗ | 302.21s | 6.92e-01 | 5 | 7 | `fp32-UnscaledAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_spot-extreme | fp16+ecc | HyperBallAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_spot-extreme | fp32 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| xor_spot-extreme | fp32 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| xor_spot-hard | fp16+ecc | HyperBallAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_spot-hard | fp32 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| xor_spot-hard | fp32 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| xor_spot-medium | fp16+ecc | HyperBallAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_spot-medium | fp32 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| xor_spot-medium | fp32 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| xor_spot-nightmare | fp16+ecc | HyperBallAdamW | No | No | ✗ | 0.00s | inf | 0 | 1 | `N/A` |
| xor_spot-nightmare | fp32 | SignLaProp | No | No | ✗ | 0.00s | inf | 0 | 3 | `N/A` |
| xor_spot-nightmare | fp32 | UnscaledAdamW | No | No | ✗ | 0.00s | inf | 0 | 7 | `N/A` |
| xor_spot-trivial | fp16+ecc | HyperBallAdamW | No | No | ✓ | 299.31s | 1.52e-04 | 4 | 1 | `fp16+ecc-HyperBallAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_spot-trivial | fp32 | SignLaProp | No | No | ✓ | 208.18s | 9.40e-03 | 2 | 3 | `fp32-SignLaProp(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00000, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| xor_spot-trivial | fp32 | UnscaledAdamW | No | No | ✓ | 220.18s | 2.42e-03 | 4 | 7 | `fp32-UnscaledAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_spot_rnn-easy | bf16+ecc | OrthoLaProp | No | No | ✗ | 300.09s | 6.94e-01 | 8 | 2 | `bf16+ecc-OrthoLaProp(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00018, betas=(0.994, 1.0000), shampoo_beta=1.000)` |
| xor_spot_rnn-easy | fp32 | Muon | No | No | ✓ | 197.10s | 1.38e-04 | 4 | 5 | `fp32-Muon(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_spot_rnn-extreme | bf16+ecc | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| xor_spot_rnn-extreme | fp32 | Muon | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| xor_spot_rnn-hard | bf16+ecc | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| xor_spot_rnn-hard | fp32 | Muon | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| xor_spot_rnn-medium | bf16+ecc | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| xor_spot_rnn-medium | fp32 | Muon | No | No | ✗ | 300.56s | 6.94e-01 | 5 | 5 | `fp32-Muon(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.993) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_spot_rnn-nightmare | bf16+ecc | OrthoLaProp | No | No | ✗ | 0.00s | inf | 0 | 2 | `N/A` |
| xor_spot_rnn-nightmare | fp32 | Muon | No | No | ✗ | 0.00s | inf | 0 | 5 | `N/A` |
| xor_spot_rnn-trivial | bf16+ecc | OrthoLaProp | No | No | ✓ | 176.56s | 4.24e-04 | 4 | 2 | `bf16+ecc-OrthoLaProp(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |
| xor_spot_rnn-trivial | fp32 | Muon | No | No | ✓ | 203.13s | 9.62e-07 | 4 | 5 | `fp32-Muon(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.520) | EMAβ=0.90(lr=0.00215, betas=(0.999, 0.9998), shampoo_beta=0.998)` |

## Errors


### CIFAR100-easy - MSAMLaProp (bf16)
```
Gated after CIFAR100-trivial failure
```

### CIFAR100-easy - PSGDKron (bf16+ecc)
```
Gated after CIFAR100-trivial failure
```

### CIFAR100-extreme - MSAMLaProp (bf16)
```
Gated after CIFAR100-trivial failure
```

### CIFAR100-extreme - PSGDKron (bf16+ecc)
```
Gated after CIFAR100-trivial failure
```

### CIFAR100-hard - MSAMLaProp (bf16)
```
Gated after CIFAR100-trivial failure
```

### CIFAR100-hard - PSGDKron (bf16+ecc)
```
Gated after CIFAR100-trivial failure
```

### CIFAR100-medium - MSAMLaProp (bf16)
```
Gated after CIFAR100-trivial failure
```

### CIFAR100-medium - PSGDKron (bf16+ecc)
```
Gated after CIFAR100-trivial failure
```

### CIFAR100-nightmare - MSAMLaProp (bf16)
```
Gated after CIFAR100-trivial failure
```

### CIFAR100-nightmare - PSGDKron (bf16+ecc)
```
Gated after CIFAR100-trivial failure
```

### CIFAR10_wide-easy - LaPropOrtho (bf16)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-easy - AdamC (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-easy - NAdam (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-easy - PSGDKron (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-easy - LATHER (fp16)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-easy - PSGDKron (fp16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-easy - SOAP (fp16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-extreme - LaPropOrtho (bf16)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-extreme - AdamC (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-extreme - NAdam (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-extreme - PSGDKron (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-extreme - LATHER (fp16)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-extreme - PSGDKron (fp16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-extreme - SOAP (fp16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-hard - LaPropOrtho (bf16)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-hard - AdamC (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-hard - NAdam (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-hard - PSGDKron (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-hard - LATHER (fp16)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-hard - PSGDKron (fp16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-hard - SOAP (fp16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-medium - LaPropOrtho (bf16)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-medium - AdamC (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-medium - NAdam (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-medium - PSGDKron (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-medium - LATHER (fp16)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-medium - PSGDKron (fp16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-medium - SOAP (fp16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-nightmare - LaPropOrtho (bf16)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-nightmare - AdamC (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-nightmare - NAdam (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-nightmare - PSGDKron (bf16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-nightmare - LATHER (fp16)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-nightmare - PSGDKron (fp16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### CIFAR10_wide-nightmare - SOAP (fp16+ecc)
```
Gated after CIFAR10_wide-trivial failure
```

### MNIST-easy - PSGDLRA (bf16+ecc)
```
Gated after MNIST-trivial failure
```

### MNIST-easy - SOAPAdEMAMix (bf16+ecc)
```
Gated after MNIST-trivial failure
```

### MNIST-extreme - PSGDLRA (bf16+ecc)
```
Gated after MNIST-trivial failure
```

### MNIST-extreme - SOAPAdEMAMix (bf16+ecc)
```
Gated after MNIST-trivial failure
```

### MNIST-hard - PSGDLRA (bf16+ecc)
```
Gated after MNIST-trivial failure
```

### MNIST-hard - SOAPAdEMAMix (bf16+ecc)
```
Gated after MNIST-trivial failure
```

### MNIST-medium - PSGDLRA (bf16+ecc)
```
Gated after MNIST-trivial failure
```

### MNIST-medium - SOAPAdEMAMix (bf16+ecc)
```
Gated after MNIST-trivial failure
```

### MNIST-nightmare - PSGDLRA (bf16+ecc)
```
Gated after MNIST-trivial failure
```

### MNIST-nightmare - SOAPAdEMAMix (bf16+ecc)
```
Gated after MNIST-trivial failure
```

### SVHN-easy - UnscaledAdamW (fp16)
```
Gated after SVHN-trivial failure
```

### SVHN-easy - SOAPNAdam (fp32)
```
Gated after SVHN-trivial failure
```

### SVHN-extreme - UnscaledAdamW (fp16)
```
Gated after SVHN-trivial failure
```

### SVHN-extreme - SOAPNAdam (fp32)
```
Gated after SVHN-trivial failure
```

### SVHN-hard - UnscaledAdamW (fp16)
```
Gated after SVHN-trivial failure
```

### SVHN-hard - SOAPNAdam (fp32)
```
Gated after SVHN-trivial failure
```

### SVHN-medium - UnscaledAdamW (fp16)
```
Gated after SVHN-trivial failure
```

### SVHN-medium - SOAPNAdam (fp32)
```
Gated after SVHN-trivial failure
```

### SVHN-nightmare - UnscaledAdamW (fp16)
```
Gated after SVHN-trivial failure
```

### SVHN-nightmare - SOAPNAdam (fp32)
```
Gated after SVHN-trivial failure
```

### Tolstoi_RNN-easy - AdamC (bf16)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-easy - ADOPT (bf16+ecc)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-easy - LaProp (bf16+ecc)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-easy - LaPropOrtho (fp16)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-extreme - AdamC (bf16)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-extreme - ADOPT (bf16+ecc)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-extreme - LaProp (bf16+ecc)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-extreme - LaPropOrtho (fp16)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-hard - AdamC (bf16)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-hard - ADOPT (bf16+ecc)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-hard - LaProp (bf16+ecc)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-hard - LaPropOrtho (fp16)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-medium - AdamC (bf16)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-medium - ADOPT (bf16+ecc)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-medium - LaProp (bf16+ecc)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-medium - LaPropOrtho (fp16)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-nightmare - AdamC (bf16)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-nightmare - ADOPT (bf16+ecc)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-nightmare - LaProp (bf16+ecc)
```
Gated after Tolstoi_RNN-trivial failure
```

### Tolstoi_RNN-nightmare - LaPropOrtho (fp16)
```
Gated after Tolstoi_RNN-trivial failure
```

### absolute_varying_scale-easy - MuonLaProp (bf16)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-easy - SignLaProp (bf16)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-easy - AdamC (fp16+ecc)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-easy - LaProp (fp16+ecc)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-easy - AdEMAMix (fp32)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-extreme - MuonLaProp (bf16)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-extreme - SignLaProp (bf16)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-extreme - AdamC (fp16+ecc)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-extreme - LaProp (fp16+ecc)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-extreme - AdEMAMix (fp32)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-hard - MuonLaProp (bf16)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-hard - SignLaProp (bf16)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-hard - AdamC (fp16+ecc)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-hard - LaProp (fp16+ecc)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-hard - AdEMAMix (fp32)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-medium - MuonLaProp (bf16)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-medium - SignLaProp (bf16)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-medium - AdamC (fp16+ecc)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-medium - LaProp (fp16+ecc)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-medium - AdEMAMix (fp32)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-nightmare - MuonLaProp (bf16)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-nightmare - SignLaProp (bf16)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-nightmare - AdamC (fp16+ecc)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-nightmare - LaProp (fp16+ecc)
```
Gated after absolute_varying_scale-trivial failure
```

### absolute_varying_scale-nightmare - AdEMAMix (fp32)
```
Gated after absolute_varying_scale-trivial failure
```

### adversarial_gradient-easy - SOLP (bf16)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-easy - AdamW (bf16+ecc)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-easy - RMSprop (fp16+ecc)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-extreme - SOLP (bf16)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-extreme - AdamW (bf16+ecc)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-extreme - RMSprop (fp16+ecc)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-hard - SOLP (bf16)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-hard - AdamW (bf16+ecc)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-hard - RMSprop (fp16+ecc)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-medium - SOLP (bf16)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-medium - AdamW (bf16+ecc)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-medium - RMSprop (fp16+ecc)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-nightmare - SOLP (bf16)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-nightmare - AdamW (bf16+ecc)
```
Gated after adversarial_gradient-trivial failure
```

### adversarial_gradient-nightmare - RMSprop (fp16+ecc)
```
Gated after adversarial_gradient-trivial failure
```

### batch_size_scaling-extreme - SUDSAdamW (bf16+ecc)
```
Gated after batch_size_scaling-medium failure
```

### batch_size_scaling-hard - SUDSAdamW (bf16+ecc)
```
Gated after batch_size_scaling-medium failure
```

### batch_size_scaling-nightmare - SUDSAdamW (bf16+ecc)
```
Gated after batch_size_scaling-medium failure
```

### beale-easy - AdamW (bf16)
```
Gated after beale-trivial failure
```

### beale-easy - NAdam (bf16)
```
Gated after beale-trivial failure
```

### beale-easy - OrthoLaProp (bf16)
```
Gated after beale-trivial failure
```

### beale-easy - SAMWrapper (bf16)
```
Gated after beale-trivial failure
```

### beale-easy - Scion (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### beale-easy - AdamW (bf16+ecc)
```
Gated after beale-trivial failure
```

### beale-easy - SAMWrapper (bf16+ecc)
```
Gated after beale-trivial failure
```

### beale-easy - SAMWrapper (fp16)
```
Gated after beale-trivial failure
```

### beale-easy - SFAdamW (fp16)
```
Gated after beale-trivial failure
```

### beale-easy - NAdam (fp32)
```
Gated after beale-trivial failure
```

### beale-easy - PSGDKron (fp32)
```
Gated after beale-trivial failure
```

### beale-easy - SOAP (fp32)
```
Gated after beale-trivial failure
```

### beale-extreme - AdamW (bf16)
```
Gated after beale-trivial failure
```

### beale-extreme - NAdam (bf16)
```
Gated after beale-trivial failure
```

### beale-extreme - OrthoLaProp (bf16)
```
Gated after beale-trivial failure
```

### beale-extreme - SAMWrapper (bf16)
```
Gated after beale-trivial failure
```

### beale-extreme - Scion (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### beale-extreme - AdamW (bf16+ecc)
```
Gated after beale-trivial failure
```

### beale-extreme - SAMWrapper (bf16+ecc)
```
Gated after beale-trivial failure
```

### beale-extreme - SAMWrapper (fp16)
```
Gated after beale-trivial failure
```

### beale-extreme - SFAdamW (fp16)
```
Gated after beale-trivial failure
```

### beale-extreme - NAdam (fp32)
```
Gated after beale-trivial failure
```

### beale-extreme - PSGDKron (fp32)
```
Gated after beale-trivial failure
```

### beale-extreme - SOAP (fp32)
```
Gated after beale-trivial failure
```

### beale-hard - AdamW (bf16)
```
Gated after beale-trivial failure
```

### beale-hard - NAdam (bf16)
```
Gated after beale-trivial failure
```

### beale-hard - OrthoLaProp (bf16)
```
Gated after beale-trivial failure
```

### beale-hard - SAMWrapper (bf16)
```
Gated after beale-trivial failure
```

### beale-hard - Scion (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### beale-hard - AdamW (bf16+ecc)
```
Gated after beale-trivial failure
```

### beale-hard - SAMWrapper (bf16+ecc)
```
Gated after beale-trivial failure
```

### beale-hard - SAMWrapper (fp16)
```
Gated after beale-trivial failure
```

### beale-hard - SFAdamW (fp16)
```
Gated after beale-trivial failure
```

### beale-hard - NAdam (fp32)
```
Gated after beale-trivial failure
```

### beale-hard - PSGDKron (fp32)
```
Gated after beale-trivial failure
```

### beale-hard - SOAP (fp32)
```
Gated after beale-trivial failure
```

### beale-medium - AdamW (bf16)
```
Gated after beale-trivial failure
```

### beale-medium - NAdam (bf16)
```
Gated after beale-trivial failure
```

### beale-medium - OrthoLaProp (bf16)
```
Gated after beale-trivial failure
```

### beale-medium - SAMWrapper (bf16)
```
Gated after beale-trivial failure
```

### beale-medium - Scion (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### beale-medium - AdamW (bf16+ecc)
```
Gated after beale-trivial failure
```

### beale-medium - SAMWrapper (bf16+ecc)
```
Gated after beale-trivial failure
```

### beale-medium - SAMWrapper (fp16)
```
Gated after beale-trivial failure
```

### beale-medium - SFAdamW (fp16)
```
Gated after beale-trivial failure
```

### beale-medium - NAdam (fp32)
```
Gated after beale-trivial failure
```

### beale-medium - PSGDKron (fp32)
```
Gated after beale-trivial failure
```

### beale-medium - SOAP (fp32)
```
Gated after beale-trivial failure
```

### beale-nightmare - AdamW (bf16)
```
Gated after beale-trivial failure
```

### beale-nightmare - NAdam (bf16)
```
Gated after beale-trivial failure
```

### beale-nightmare - OrthoLaProp (bf16)
```
Gated after beale-trivial failure
```

### beale-nightmare - SAMWrapper (bf16)
```
Gated after beale-trivial failure
```

### beale-nightmare - Scion (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### beale-nightmare - AdamW (bf16+ecc)
```
Gated after beale-trivial failure
```

### beale-nightmare - SAMWrapper (bf16+ecc)
```
Gated after beale-trivial failure
```

### beale-nightmare - SAMWrapper (fp16)
```
Gated after beale-trivial failure
```

### beale-nightmare - SFAdamW (fp16)
```
Gated after beale-trivial failure
```

### beale-nightmare - NAdam (fp32)
```
Gated after beale-trivial failure
```

### beale-nightmare - PSGDKron (fp32)
```
Gated after beale-trivial failure
```

### beale-nightmare - SOAP (fp32)
```
Gated after beale-trivial failure
```

### constrained_optimization_log-easy - SAMWrapper (fp16+ecc)
```
Gated after constrained_optimization_log-trivial failure
```

### constrained_optimization_log-extreme - SAMWrapper (fp16+ecc)
```
Gated after constrained_optimization_log-trivial failure
```

### constrained_optimization_log-hard - SAMWrapper (fp16+ecc)
```
Gated after constrained_optimization_log-trivial failure
```

### constrained_optimization_log-medium - SAMWrapper (fp16+ecc)
```
Gated after constrained_optimization_log-trivial failure
```

### constrained_optimization_log-nightmare - SAMWrapper (fp16+ecc)
```
Gated after constrained_optimization_log-trivial failure
```

### constrained_optimization_relu-easy - SOAP (bf16+ecc)
```
Gated after constrained_optimization_relu-trivial failure
```

### constrained_optimization_relu-extreme - SGD (bf16)
```
Gated after constrained_optimization_relu-easy failure
```

### constrained_optimization_relu-extreme - SignLaProp (bf16)
```
Gated after constrained_optimization_relu-medium failure
```

### constrained_optimization_relu-extreme - SOAP (bf16+ecc)
```
Gated after constrained_optimization_relu-trivial failure
```

### constrained_optimization_relu-extreme - SignLaProp (fp16)
```
Gated after constrained_optimization_relu-hard failure
```

### constrained_optimization_relu-extreme - AdEMAMix (fp32)
```
Gated after constrained_optimization_relu-hard failure
```

### constrained_optimization_relu-hard - SGD (bf16)
```
Gated after constrained_optimization_relu-easy failure
```

### constrained_optimization_relu-hard - SignLaProp (bf16)
```
Gated after constrained_optimization_relu-medium failure
```

### constrained_optimization_relu-hard - SOAP (bf16+ecc)
```
Gated after constrained_optimization_relu-trivial failure
```

### constrained_optimization_relu-medium - SGD (bf16)
```
Gated after constrained_optimization_relu-easy failure
```

### constrained_optimization_relu-medium - SOAP (bf16+ecc)
```
Gated after constrained_optimization_relu-trivial failure
```

### constrained_optimization_relu-nightmare - SGD (bf16)
```
Gated after constrained_optimization_relu-easy failure
```

### constrained_optimization_relu-nightmare - SignLaProp (bf16)
```
Gated after constrained_optimization_relu-medium failure
```

### constrained_optimization_relu-nightmare - SOAP (bf16+ecc)
```
Gated after constrained_optimization_relu-trivial failure
```

### constrained_optimization_relu-nightmare - SignLaProp (fp16)
```
Gated after constrained_optimization_relu-hard failure
```

### constrained_optimization_relu-nightmare - AdEMAMix (fp32)
```
Gated after constrained_optimization_relu-hard failure
```

### constrained_optimization_relu-nightmare - HyperBallAdamW (fp32)
```
Gated after constrained_optimization_relu-extreme failure
```

### discontinuous_gradient-easy - SGD (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-easy - AdamW (fp16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-easy - SGD (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-extreme - SGD (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-extreme - AdamW (fp16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-extreme - SGD (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-hard - SGD (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-hard - AdamW (fp16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-hard - SGD (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-medium - SGD (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-medium - AdamW (fp16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-medium - SGD (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-nightmare - SGD (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-nightmare - AdamW (fp16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### discontinuous_gradient-nightmare - SGD (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### dynamic_landscape-extreme - RMSprop (fp16)
```
Gated after dynamic_landscape-medium failure
```

### dynamic_landscape-extreme - SGD (fp32)
```
Gated after dynamic_landscape-medium failure
```

### dynamic_landscape-hard - RMSprop (fp16)
```
Gated after dynamic_landscape-medium failure
```

### dynamic_landscape-hard - SGD (fp32)
```
Gated after dynamic_landscape-medium failure
```

### dynamic_landscape-nightmare - RMSprop (fp16)
```
Gated after dynamic_landscape-medium failure
```

### dynamic_landscape-nightmare - SGD (fp32)
```
Gated after dynamic_landscape-medium failure
```

### exploding_gradient-easy - LaProp (bf16+ecc)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-easy - SOLP (bf16+ecc)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-easy - RMSprop (fp16)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-easy - UnscaledAdamW (fp16)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-extreme - LaProp (bf16+ecc)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-extreme - SOLP (bf16+ecc)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-extreme - RMSprop (fp16)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-extreme - UnscaledAdamW (fp16)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-hard - LaProp (bf16+ecc)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-hard - SOLP (bf16+ecc)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-hard - RMSprop (fp16)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-hard - UnscaledAdamW (fp16)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-medium - LaProp (bf16+ecc)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-medium - SOLP (bf16+ecc)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-medium - RMSprop (fp16)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-medium - UnscaledAdamW (fp16)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-nightmare - LaProp (bf16+ecc)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-nightmare - SOLP (bf16+ecc)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-nightmare - RMSprop (fp16)
```
Gated after exploding_gradient-trivial failure
```

### exploding_gradient-nightmare - UnscaledAdamW (fp16)
```
Gated after exploding_gradient-trivial failure
```

### generalization_summod-easy - LaPropOrtho (fp16+ecc)
```
Gated after generalization_summod-trivial failure
```

### generalization_summod-extreme - LaPropOrtho (fp16+ecc)
```
Gated after generalization_summod-trivial failure
```

### generalization_summod-hard - LaPropOrtho (fp16+ecc)
```
Gated after generalization_summod-trivial failure
```

### generalization_summod-medium - LaPropOrtho (fp16+ecc)
```
Gated after generalization_summod-trivial failure
```

### generalization_summod-nightmare - LaPropOrtho (fp16+ecc)
```
Gated after generalization_summod-trivial failure
```

### generalization_weighted_summod-easy - SOAPNAdam (bf16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-easy - LATHER (fp16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-easy - LaProp (fp16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-extreme - SOAPNAdam (bf16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-extreme - LATHER (fp16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-extreme - LaProp (fp16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-hard - SOAPNAdam (bf16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-hard - LATHER (fp16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-hard - LaProp (fp16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-medium - SOAPNAdam (bf16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-medium - LATHER (fp16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-medium - LaProp (fp16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-nightmare - SOAPNAdam (bf16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-nightmare - LATHER (fp16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### generalization_weighted_summod-nightmare - LaProp (fp16+ecc)
```
Gated after generalization_weighted_summod-trivial failure
```

### gradient_delay-easy - LaPropOrtho (bf16+ecc)
```
Gated after gradient_delay-trivial failure
```

### gradient_delay-extreme - AdEMAMix (bf16)
```
Gated after gradient_delay-medium failure
```

### gradient_delay-extreme - PSGDKron (bf16)
```
Gated after gradient_delay-medium failure
```

### gradient_delay-extreme - LaPropOrtho (bf16+ecc)
```
Gated after gradient_delay-trivial failure
```

### gradient_delay-extreme - OrthoLaProp (bf16+ecc)
```
Gated after gradient_delay-medium failure
```

### gradient_delay-extreme - SOAPAdEMAMix (fp32)
```
Gated after gradient_delay-medium failure
```

### gradient_delay-hard - AdEMAMix (bf16)
```
Gated after gradient_delay-medium failure
```

### gradient_delay-hard - PSGDKron (bf16)
```
Gated after gradient_delay-medium failure
```

### gradient_delay-hard - LaPropOrtho (bf16+ecc)
```
Gated after gradient_delay-trivial failure
```

### gradient_delay-hard - OrthoLaProp (bf16+ecc)
```
Gated after gradient_delay-medium failure
```

### gradient_delay-hard - SOAPAdEMAMix (fp32)
```
Gated after gradient_delay-medium failure
```

### gradient_delay-medium - LaPropOrtho (bf16+ecc)
```
Gated after gradient_delay-trivial failure
```

### gradient_delay-nightmare - AdEMAMix (bf16)
```
Gated after gradient_delay-medium failure
```

### gradient_delay-nightmare - PSGDKron (bf16)
```
Gated after gradient_delay-medium failure
```

### gradient_delay-nightmare - LaPropOrtho (bf16+ecc)
```
Gated after gradient_delay-trivial failure
```

### gradient_delay-nightmare - OrthoLaProp (bf16+ecc)
```
Gated after gradient_delay-medium failure
```

### gradient_delay-nightmare - SOAPAdEMAMix (fp32)
```
Gated after gradient_delay-medium failure
```

### gradient_noise_scale_heavy_tailed-easy - AdamW (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-easy - LATHER (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-easy - PSGDKron (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-easy - PSGDKron (fp16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-extreme - AdamW (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-extreme - LATHER (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-extreme - PSGDKron (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-extreme - PSGDKron (fp16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-hard - AdamW (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-hard - LATHER (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-hard - PSGDKron (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-hard - PSGDKron (fp16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-medium - AdamW (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-medium - LATHER (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-medium - PSGDKron (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-medium - PSGDKron (fp16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-nightmare - AdamW (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-nightmare - LATHER (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-nightmare - PSGDKron (bf16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_heavy_tailed-nightmare - PSGDKron (fp16+ecc)
```
Gated after gradient_noise_scale_heavy_tailed-trivial failure
```

### gradient_noise_scale_instantaneous_brown-easy - PSGDLRA (bf16+ecc)
```
Gated after gradient_noise_scale_instantaneous_brown-trivial failure
```

### gradient_noise_scale_instantaneous_brown-extreme - PSGDLRA (bf16+ecc)
```
Gated after gradient_noise_scale_instantaneous_brown-trivial failure
```

### gradient_noise_scale_instantaneous_brown-extreme - MSAMLaProp (fp32)
```
Gated after gradient_noise_scale_instantaneous_brown-easy failure
```

### gradient_noise_scale_instantaneous_brown-hard - PSGDLRA (bf16+ecc)
```
Gated after gradient_noise_scale_instantaneous_brown-trivial failure
```

### gradient_noise_scale_instantaneous_brown-hard - MSAMLaProp (fp32)
```
Gated after gradient_noise_scale_instantaneous_brown-easy failure
```

### gradient_noise_scale_instantaneous_brown-medium - PSGDLRA (bf16+ecc)
```
Gated after gradient_noise_scale_instantaneous_brown-trivial failure
```

### gradient_noise_scale_instantaneous_brown-medium - MSAMLaProp (fp32)
```
Gated after gradient_noise_scale_instantaneous_brown-easy failure
```

### gradient_noise_scale_instantaneous_brown-nightmare - PSGDLRA (bf16+ecc)
```
Gated after gradient_noise_scale_instantaneous_brown-trivial failure
```

### gradient_noise_scale_instantaneous_brown-nightmare - MSAMLaProp (fp32)
```
Gated after gradient_noise_scale_instantaneous_brown-easy failure
```

### gradient_noise_scale_temporal_brown-easy - MuonLaProp (fp16+ecc)
```
Gated after gradient_noise_scale_temporal_brown-trivial failure
```

### gradient_noise_scale_temporal_brown-easy - SGD (fp32)
```
Gated after gradient_noise_scale_temporal_brown-trivial failure
```

### gradient_noise_scale_temporal_brown-extreme - MuonLaProp (fp16+ecc)
```
Gated after gradient_noise_scale_temporal_brown-trivial failure
```

### gradient_noise_scale_temporal_brown-extreme - SGD (fp32)
```
Gated after gradient_noise_scale_temporal_brown-trivial failure
```

### gradient_noise_scale_temporal_brown-hard - MuonLaProp (fp16+ecc)
```
Gated after gradient_noise_scale_temporal_brown-trivial failure
```

### gradient_noise_scale_temporal_brown-hard - SGD (fp32)
```
Gated after gradient_noise_scale_temporal_brown-trivial failure
```

### gradient_noise_scale_temporal_brown-medium - MuonLaProp (fp16+ecc)
```
Gated after gradient_noise_scale_temporal_brown-trivial failure
```

### gradient_noise_scale_temporal_brown-medium - SGD (fp32)
```
Gated after gradient_noise_scale_temporal_brown-trivial failure
```

### gradient_noise_scale_temporal_brown-nightmare - MuonLaProp (fp16+ecc)
```
Gated after gradient_noise_scale_temporal_brown-trivial failure
```

### gradient_noise_scale_temporal_brown-nightmare - SGD (fp32)
```
Gated after gradient_noise_scale_temporal_brown-trivial failure
```

### gradient_noise_scale_white-easy - MuonLaProp (bf16)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-easy - MuonLaProp (fp16+ecc)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-easy - NAdam (fp32)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-extreme - MuonLaProp (bf16)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-extreme - SFAdamW (bf16)
```
Gated after gradient_noise_scale_white-medium failure
```

### gradient_noise_scale_white-extreme - MuonLaProp (fp16+ecc)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-extreme - NAdam (fp32)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-hard - MuonLaProp (bf16)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-hard - SFAdamW (bf16)
```
Gated after gradient_noise_scale_white-medium failure
```

### gradient_noise_scale_white-hard - MuonLaProp (fp16+ecc)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-hard - NAdam (fp32)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-medium - MuonLaProp (bf16)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-medium - MuonLaProp (fp16+ecc)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-medium - NAdam (fp32)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-nightmare - MuonLaProp (bf16)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-nightmare - SFAdamW (bf16)
```
Gated after gradient_noise_scale_white-medium failure
```

### gradient_noise_scale_white-nightmare - MuonLaProp (fp16+ecc)
```
Gated after gradient_noise_scale_white-trivial failure
```

### gradient_noise_scale_white-nightmare - NAdam (fp32)
```
Gated after gradient_noise_scale_white-trivial failure
```

### grokking-easy - MuonAdamW (bf16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-easy - NAdam (bf16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-easy - AdamW (bf16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-easy - LaProp (bf16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-easy - LATHER (fp16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-easy - SignLaProp (fp16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-easy - MuonLaProp (fp16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-easy - SOAPAdEMAMix (fp16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-easy - Scion (fp32)
```
Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-extreme - MuonAdamW (bf16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-extreme - NAdam (bf16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-extreme - AdamW (bf16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-extreme - LaProp (bf16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-extreme - LATHER (fp16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-extreme - SignLaProp (fp16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-extreme - MuonLaProp (fp16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-extreme - SOAPAdEMAMix (fp16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-extreme - Scion (fp32)
```
Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-hard - MuonAdamW (bf16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-hard - NAdam (bf16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-hard - AdamW (bf16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-hard - LaProp (bf16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-hard - LATHER (fp16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-hard - SignLaProp (fp16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-hard - MuonLaProp (fp16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-hard - SOAPAdEMAMix (fp16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-hard - Scion (fp32)
```
Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-medium - MuonAdamW (bf16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-medium - NAdam (bf16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-medium - AdamW (bf16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-medium - LaProp (bf16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-medium - LATHER (fp16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-medium - SignLaProp (fp16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-medium - MuonLaProp (fp16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-medium - SOAPAdEMAMix (fp16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-medium - Scion (fp32)
```
Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-nightmare - MuonAdamW (bf16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-nightmare - NAdam (bf16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-nightmare - AdamW (bf16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-nightmare - LaProp (bf16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-nightmare - LATHER (fp16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-nightmare - SignLaProp (fp16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-nightmare - MuonLaProp (fp16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-nightmare - SOAPAdEMAMix (fp16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-nightmare - Scion (fp32)
```
Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-trivial - MuonAdamW (bf16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-trivial - NAdam (bf16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-trivial - AdamW (bf16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-trivial - LaProp (bf16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-trivial - LATHER (fp16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-trivial - SignLaProp (fp16)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-trivial - MuonLaProp (fp16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-trivial - SOAPAdEMAMix (fp16+ecc)
```
/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py:433: UserWarning: Got pickle error when attempting to start a worker Process. This might be because the worker Process arguments are not picklable. Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms to 'forkserver', which requires the worker Process arguments to be picklable. You can also try multiprocessing.set_start_method('fork').
  return _MultiProcessingDataLoaderIter(self)

Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### grokking-trivial - Scion (fp32)
```
Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/grokking.py", line 169, in main
    test_loader = list(test_loader)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 500, in __iter__
    return self._get_iterator()
           ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/venv/main/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/venv/main/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/venv/main/lib/python3.12/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>.<lambda>'

```

### ill_conditioned-extreme - ADOPT (bf16)
```
Gated after ill_conditioned-medium failure
```

### ill_conditioned-extreme - PSGDKron (fp32)
```
Gated after ill_conditioned-hard failure
```

### ill_conditioned-hard - ADOPT (bf16)
```
Gated after ill_conditioned-medium failure
```

### ill_conditioned-nightmare - ADOPT (bf16)
```
Gated after ill_conditioned-medium failure
```

### ill_conditioned-nightmare - SGD (fp16)
```
Gated after ill_conditioned-extreme failure
```

### ill_conditioned-nightmare - PSGDKron (fp32)
```
Gated after ill_conditioned-hard failure
```

### ill_conditioned_mismatched-nightmare - NAdam (fp16+ecc)
```
Gated after ill_conditioned_mismatched-extreme failure
```

### layer_wise_scale-extreme - UnscaledAdamW (fp16+ecc)
```
Gated after layer_wise_scale-medium failure
```

### layer_wise_scale-hard - UnscaledAdamW (fp16+ecc)
```
Gated after layer_wise_scale-medium failure
```

### layer_wise_scale-nightmare - UnscaledAdamW (fp16+ecc)
```
Gated after layer_wise_scale-medium failure
```

### memory_constrained-easy - LATHER (fp32)
```
Gated after memory_constrained-trivial failure
```

### memory_constrained-extreme - LATHER (fp32)
```
Gated after memory_constrained-trivial failure
```

### memory_constrained-hard - LATHER (fp32)
```
Gated after memory_constrained-trivial failure
```

### memory_constrained-medium - LATHER (fp32)
```
Gated after memory_constrained-trivial failure
```

### memory_constrained-nightmare - LATHER (fp32)
```
Gated after memory_constrained-trivial failure
```

### minimax-easy - SAMWrapper (bf16)
```
Gated after minimax-trivial failure
```

### minimax-easy - AdEMAMix (fp16)
```
Gated after minimax-trivial failure
```

### minimax-easy - MSAMLaProp (fp16)
```
Gated after minimax-trivial failure
```

### minimax-easy - SGD (fp16+ecc)
```
Gated after minimax-trivial failure
```

### minimax-extreme - SAMWrapper (bf16)
```
Gated after minimax-trivial failure
```

### minimax-extreme - AdEMAMix (fp16)
```
Gated after minimax-trivial failure
```

### minimax-extreme - MSAMLaProp (fp16)
```
Gated after minimax-trivial failure
```

### minimax-extreme - Scion (fp16)
```
Gated after minimax-hard failure
```

### minimax-extreme - SGD (fp16+ecc)
```
Gated after minimax-trivial failure
```

### minimax-hard - SAMWrapper (bf16)
```
Gated after minimax-trivial failure
```

### minimax-hard - AdEMAMix (fp16)
```
Gated after minimax-trivial failure
```

### minimax-hard - MSAMLaProp (fp16)
```
Gated after minimax-trivial failure
```

### minimax-hard - SGD (fp16+ecc)
```
Gated after minimax-trivial failure
```

### minimax-medium - SAMWrapper (bf16)
```
Gated after minimax-trivial failure
```

### minimax-medium - AdEMAMix (fp16)
```
Gated after minimax-trivial failure
```

### minimax-medium - MSAMLaProp (fp16)
```
Gated after minimax-trivial failure
```

### minimax-medium - SGD (fp16+ecc)
```
Gated after minimax-trivial failure
```

### minimax-nightmare - SAMWrapper (bf16)
```
Gated after minimax-trivial failure
```

### minimax-nightmare - AdEMAMix (fp16)
```
Gated after minimax-trivial failure
```

### minimax-nightmare - MSAMLaProp (fp16)
```
Gated after minimax-trivial failure
```

### minimax-nightmare - Scion (fp16)
```
Gated after minimax-hard failure
```

### minimax-nightmare - SGD (fp16+ecc)
```
Gated after minimax-trivial failure
```

### multi_objective_pareto-easy - SFAdamW (bf16)
```
Gated after multi_objective_pareto-trivial failure
```

### multi_objective_pareto-easy - HyperBallAdamW (bf16+ecc)
```
Gated after multi_objective_pareto-trivial failure
```

### multi_objective_pareto-extreme - SFAdamW (bf16)
```
Gated after multi_objective_pareto-trivial failure
```

### multi_objective_pareto-extreme - HyperBallAdamW (bf16+ecc)
```
Gated after multi_objective_pareto-trivial failure
```

### multi_objective_pareto-hard - SFAdamW (bf16)
```
Gated after multi_objective_pareto-trivial failure
```

### multi_objective_pareto-hard - HyperBallAdamW (bf16+ecc)
```
Gated after multi_objective_pareto-trivial failure
```

### multi_objective_pareto-medium - SFAdamW (bf16)
```
Gated after multi_objective_pareto-trivial failure
```

### multi_objective_pareto-medium - HyperBallAdamW (bf16+ecc)
```
Gated after multi_objective_pareto-trivial failure
```

### multi_objective_pareto-nightmare - SFAdamW (bf16)
```
Gated after multi_objective_pareto-trivial failure
```

### multi_objective_pareto-nightmare - HyperBallAdamW (bf16+ecc)
```
Gated after multi_objective_pareto-trivial failure
```

### noisy_matmul-easy - PSGDLRA (bf16)
```
Gated after noisy_matmul-trivial failure
```

### noisy_matmul-easy - AdamC (fp16)
```
Gated after noisy_matmul-trivial failure
```

### noisy_matmul-extreme - PSGDLRA (bf16)
```
Gated after noisy_matmul-trivial failure
```

### noisy_matmul-extreme - LaProp (bf16+ecc)
```
Gated after noisy_matmul-hard failure
```

### noisy_matmul-extreme - AdamC (fp16)
```
Gated after noisy_matmul-trivial failure
```

### noisy_matmul-extreme - SFAdamW (fp16)
```
Gated after noisy_matmul-easy failure
```

### noisy_matmul-extreme - SOLP (fp16+ecc)
```
Gated after noisy_matmul-hard failure
```

### noisy_matmul-extreme - PSGDKron (fp32)
```
Gated after noisy_matmul-medium failure
```

### noisy_matmul-hard - PSGDLRA (bf16)
```
Gated after noisy_matmul-trivial failure
```

### noisy_matmul-hard - AdamC (fp16)
```
Gated after noisy_matmul-trivial failure
```

### noisy_matmul-hard - SFAdamW (fp16)
```
Gated after noisy_matmul-easy failure
```

### noisy_matmul-hard - PSGDKron (fp32)
```
Gated after noisy_matmul-medium failure
```

### noisy_matmul-medium - PSGDLRA (bf16)
```
Gated after noisy_matmul-trivial failure
```

### noisy_matmul-medium - AdamC (fp16)
```
Gated after noisy_matmul-trivial failure
```

### noisy_matmul-medium - SFAdamW (fp16)
```
Gated after noisy_matmul-easy failure
```

### noisy_matmul-nightmare - PSGDLRA (bf16)
```
Gated after noisy_matmul-trivial failure
```

### noisy_matmul-nightmare - LaProp (bf16+ecc)
```
Gated after noisy_matmul-hard failure
```

### noisy_matmul-nightmare - AdamC (fp16)
```
Gated after noisy_matmul-trivial failure
```

### noisy_matmul-nightmare - SFAdamW (fp16)
```
Gated after noisy_matmul-easy failure
```

### noisy_matmul-nightmare - SOLP (fp16+ecc)
```
Gated after noisy_matmul-hard failure
```

### noisy_matmul-nightmare - PSGDKron (fp32)
```
Gated after noisy_matmul-medium failure
```

### non_markovian-easy - SignLaProp (fp16)
```
Gated after non_markovian-trivial failure
```

### non_markovian-extreme - SignLaProp (fp16)
```
Gated after non_markovian-trivial failure
```

### non_markovian-extreme - HyperBallAdamW (fp16+ecc)
```
Gated after non_markovian-hard failure
```

### non_markovian-hard - SignLaProp (fp16)
```
Gated after non_markovian-trivial failure
```

### non_markovian-medium - SignLaProp (fp16)
```
Gated after non_markovian-trivial failure
```

### non_markovian-nightmare - SignLaProp (fp16)
```
Gated after non_markovian-trivial failure
```

### non_markovian-nightmare - HyperBallAdamW (fp16+ecc)
```
Gated after non_markovian-hard failure
```

### parameter_scale-easy - SUDSAdamW (bf16+ecc)
```
Gated after parameter_scale-trivial failure
```

### parameter_scale-easy - SAMWrapper (fp32)
```
Gated after parameter_scale-trivial failure
```

### parameter_scale-extreme - SUDSAdamW (bf16+ecc)
```
Gated after parameter_scale-trivial failure
```

### parameter_scale-extreme - SAMWrapper (fp32)
```
Gated after parameter_scale-trivial failure
```

### parameter_scale-extreme - SFAdamW (fp32)
```
Gated after parameter_scale-hard failure
```

### parameter_scale-hard - SUDSAdamW (bf16+ecc)
```
Gated after parameter_scale-trivial failure
```

### parameter_scale-hard - SAMWrapper (fp32)
```
Gated after parameter_scale-trivial failure
```

### parameter_scale-medium - SUDSAdamW (bf16+ecc)
```
Gated after parameter_scale-trivial failure
```

### parameter_scale-medium - SAMWrapper (fp32)
```
Gated after parameter_scale-trivial failure
```

### parameter_scale-nightmare - SUDSAdamW (bf16+ecc)
```
Gated after parameter_scale-trivial failure
```

### parameter_scale-nightmare - SAMWrapper (fp32)
```
Gated after parameter_scale-trivial failure
```

### parameter_scale-nightmare - SFAdamW (fp32)
```
Gated after parameter_scale-hard failure
```

### plasticity_summod-easy - LaProp (fp16)
```
Gated after plasticity_summod-trivial failure
```

### plasticity_summod-extreme - LaProp (fp16)
```
Gated after plasticity_summod-trivial failure
```

### plasticity_summod-hard - LaProp (fp16)
```
Gated after plasticity_summod-trivial failure
```

### plasticity_summod-medium - LaProp (fp16)
```
Gated after plasticity_summod-trivial failure
```

### plasticity_summod-nightmare - LaProp (fp16)
```
Gated after plasticity_summod-trivial failure
```

### plasticity_weighted_summod-easy - SUDSAdamW (fp16)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-easy - PSGDKron (fp32)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-easy - UnscaledAdamW (fp32)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-extreme - SUDSAdamW (fp16)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-extreme - PSGDKron (fp32)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-extreme - UnscaledAdamW (fp32)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-hard - SUDSAdamW (fp16)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-hard - PSGDKron (fp32)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-hard - UnscaledAdamW (fp32)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-medium - SUDSAdamW (fp16)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-medium - PSGDKron (fp32)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-medium - UnscaledAdamW (fp32)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-nightmare - SUDSAdamW (fp16)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-nightmare - PSGDKron (fp32)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plasticity_weighted_summod-nightmare - UnscaledAdamW (fp32)
```
Gated after plasticity_weighted_summod-trivial failure
```

### plateau_navigation-extreme - PSGDPRO (bf16+ecc)
```
Gated after plateau_navigation-medium failure
```

### plateau_navigation-extreme - LaProp (fp16+ecc)
```
Gated after plateau_navigation-hard failure
```

### plateau_navigation-hard - PSGDPRO (bf16+ecc)
```
Gated after plateau_navigation-medium failure
```

### plateau_navigation-nightmare - PSGDPRO (bf16+ecc)
```
Gated after plateau_navigation-medium failure
```

### plateau_navigation-nightmare - LaProp (fp16+ecc)
```
Gated after plateau_navigation-hard failure
```

### powers-easy - RMSprop (fp16+ecc)
```
Gated after powers-trivial failure
```

### powers-extreme - MuonLaProp (bf16+ecc)
```
Gated after powers-medium failure
```

### powers-extreme - RMSprop (fp16+ecc)
```
Gated after powers-trivial failure
```

### powers-hard - MuonLaProp (bf16+ecc)
```
Gated after powers-medium failure
```

### powers-hard - RMSprop (fp16+ecc)
```
Gated after powers-trivial failure
```

### powers-medium - RMSprop (fp16+ecc)
```
Gated after powers-trivial failure
```

### powers-nightmare - MuonLaProp (bf16+ecc)
```
Gated after powers-medium failure
```

### powers-nightmare - RMSprop (fp16+ecc)
```
Gated after powers-trivial failure
```

### powers_varying_target-easy - LaPropOrtho (fp32)
```
Gated after powers_varying_target-trivial failure
```

### powers_varying_target-extreme - SOLP (bf16)
```
Gated after powers_varying_target-easy failure
```

### powers_varying_target-extreme - LaPropOrtho (fp32)
```
Gated after powers_varying_target-trivial failure
```

### powers_varying_target-hard - SOLP (bf16)
```
Gated after powers_varying_target-easy failure
```

### powers_varying_target-hard - LaPropOrtho (fp32)
```
Gated after powers_varying_target-trivial failure
```

### powers_varying_target-medium - SOLP (bf16)
```
Gated after powers_varying_target-easy failure
```

### powers_varying_target-medium - LaPropOrtho (fp32)
```
Gated after powers_varying_target-trivial failure
```

### powers_varying_target-nightmare - SOLP (bf16)
```
Gated after powers_varying_target-easy failure
```

### powers_varying_target-nightmare - LaPropOrtho (fp32)
```
Gated after powers_varying_target-trivial failure
```

### quadratic_coupled-easy - LaPropOrtho (bf16+ecc)
```
Gated after quadratic_coupled-trivial failure
```

### quadratic_coupled-easy - Muon (bf16+ecc)
```
Gated after quadratic_coupled-trivial failure
```

### quadratic_coupled-extreme - LaPropOrtho (bf16+ecc)
```
Gated after quadratic_coupled-trivial failure
```

### quadratic_coupled-extreme - Muon (bf16+ecc)
```
Gated after quadratic_coupled-trivial failure
```

### quadratic_coupled-extreme - SOAP (fp16)
```
Gated after quadratic_coupled-easy failure
```

### quadratic_coupled-extreme - SOLP (fp32)
```
Gated after quadratic_coupled-easy failure
```

### quadratic_coupled-hard - LaPropOrtho (bf16+ecc)
```
Gated after quadratic_coupled-trivial failure
```

### quadratic_coupled-hard - Muon (bf16+ecc)
```
Gated after quadratic_coupled-trivial failure
```

### quadratic_coupled-hard - SOAP (fp16)
```
Gated after quadratic_coupled-easy failure
```

### quadratic_coupled-hard - SOLP (fp32)
```
Gated after quadratic_coupled-easy failure
```

### quadratic_coupled-medium - LaPropOrtho (bf16+ecc)
```
Gated after quadratic_coupled-trivial failure
```

### quadratic_coupled-medium - Muon (bf16+ecc)
```
Gated after quadratic_coupled-trivial failure
```

### quadratic_coupled-medium - SOAP (fp16)
```
Gated after quadratic_coupled-easy failure
```

### quadratic_coupled-medium - SOLP (fp32)
```
Gated after quadratic_coupled-easy failure
```

### quadratic_coupled-nightmare - LaPropOrtho (bf16+ecc)
```
Gated after quadratic_coupled-trivial failure
```

### quadratic_coupled-nightmare - Muon (bf16+ecc)
```
Gated after quadratic_coupled-trivial failure
```

### quadratic_coupled-nightmare - SOAP (fp16)
```
Gated after quadratic_coupled-easy failure
```

### quadratic_coupled-nightmare - SOLP (fp32)
```
Gated after quadratic_coupled-easy failure
```

### quadratic_flip-easy - LATHER (bf16)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-easy - RMSprop (fp16)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-easy - SOAPNAdam (fp16+ecc)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-extreme - LATHER (bf16)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-extreme - RMSprop (fp16)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-extreme - SOAPNAdam (fp16+ecc)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-hard - LATHER (bf16)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-hard - RMSprop (fp16)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-hard - SOAPNAdam (fp16+ecc)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-medium - LATHER (bf16)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-medium - RMSprop (fp16)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-medium - SOAPNAdam (fp16+ecc)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-nightmare - LATHER (bf16)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-nightmare - RMSprop (fp16)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_flip-nightmare - SOAPNAdam (fp16+ecc)
```
Gated after quadratic_flip-trivial failure
```

### quadratic_outliers-extreme - AdamW (bf16)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-extreme - Muon (bf16)
```
Gated after quadratic_outliers-medium failure
```

### quadratic_outliers-extreme - MuonLaProp (bf16+ecc)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-extreme - MuonAdamW (fp16)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-extreme - PSGDKron (fp16)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-hard - AdamW (bf16)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-hard - Muon (bf16)
```
Gated after quadratic_outliers-medium failure
```

### quadratic_outliers-hard - MuonLaProp (bf16+ecc)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-hard - MuonAdamW (fp16)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-hard - PSGDKron (fp16)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-medium - AdamW (bf16)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-medium - MuonLaProp (bf16+ecc)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-medium - MuonAdamW (fp16)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-medium - PSGDKron (fp16)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-nightmare - AdamW (bf16)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-nightmare - Muon (bf16)
```
Gated after quadratic_outliers-medium failure
```

### quadratic_outliers-nightmare - MuonLaProp (bf16+ecc)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-nightmare - MuonAdamW (fp16)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_outliers-nightmare - PSGDKron (fp16)
```
Gated after quadratic_outliers-easy failure
```

### quadratic_quantization_aware_training-easy - AdEMAMix (bf16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-easy - SUDSAdamW (bf16+ecc)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-easy - MuonLaProp (fp16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-easy - SignLaProp (fp16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-easy - MuonLaProp (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-easy - SAMWrapper (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-easy - SOLP (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-extreme - AdEMAMix (bf16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-extreme - SUDSAdamW (bf16+ecc)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-extreme - MuonLaProp (fp16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-extreme - SignLaProp (fp16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-extreme - MuonLaProp (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-extreme - SAMWrapper (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-extreme - SOLP (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-hard - AdEMAMix (bf16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-hard - SUDSAdamW (bf16+ecc)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-hard - MuonLaProp (fp16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-hard - SignLaProp (fp16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-hard - MuonLaProp (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-hard - SAMWrapper (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-hard - SOLP (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-medium - AdEMAMix (bf16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-medium - SUDSAdamW (bf16+ecc)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-medium - MuonLaProp (fp16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-medium - SignLaProp (fp16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-medium - MuonLaProp (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-medium - SAMWrapper (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-medium - SOLP (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-nightmare - AdEMAMix (bf16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-nightmare - SUDSAdamW (bf16+ecc)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-nightmare - MuonLaProp (fp16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-nightmare - SignLaProp (fp16)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-nightmare - MuonLaProp (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-nightmare - SAMWrapper (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantization_aware_training-nightmare - SOLP (fp32)
```
Gated after quadratic_quantization_aware_training-trivial failure
```

### quadratic_quantized_gradient-easy - SAMWrapper (bf16)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-easy - PSGDPRO (fp16+ecc)
```
Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/quadratic_quantized_gradient.py", line 54, in main
    model = Model(**kwargs).cuda()
            ^^^^^^^^^^^^^^^
TypeError: Model.__init__() missing 1 required positional argument: 'bins'

```

### quadratic_quantized_gradient-easy - SOAP (fp16+ecc)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-easy - SOAPAdEMAMix (fp16+ecc)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-easy - LATHER (fp32)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-extreme - SAMWrapper (bf16)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-extreme - PSGDPRO (fp16+ecc)
```
Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/quadratic_quantized_gradient.py", line 54, in main
    model = Model(**kwargs).cuda()
            ^^^^^^^^^^^^^^^
TypeError: Model.__init__() missing 1 required positional argument: 'bins'

```

### quadratic_quantized_gradient-extreme - SOAP (fp16+ecc)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-extreme - SOAPAdEMAMix (fp16+ecc)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-extreme - LATHER (fp32)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-hard - SAMWrapper (bf16)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-hard - PSGDPRO (fp16+ecc)
```
Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/quadratic_quantized_gradient.py", line 54, in main
    model = Model(**kwargs).cuda()
            ^^^^^^^^^^^^^^^
TypeError: Model.__init__() missing 1 required positional argument: 'bins'

```

### quadratic_quantized_gradient-hard - SOAP (fp16+ecc)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-hard - SOAPAdEMAMix (fp16+ecc)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-hard - LATHER (fp32)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-medium - SAMWrapper (bf16)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-medium - PSGDPRO (fp16+ecc)
```
Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/quadratic_quantized_gradient.py", line 54, in main
    model = Model(**kwargs).cuda()
            ^^^^^^^^^^^^^^^
TypeError: Model.__init__() missing 1 required positional argument: 'bins'

```

### quadratic_quantized_gradient-medium - SOAP (fp16+ecc)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-medium - SOAPAdEMAMix (fp16+ecc)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-medium - LATHER (fp32)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-nightmare - SAMWrapper (bf16)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-nightmare - PSGDPRO (fp16+ecc)
```
Traceback (most recent call last):
  File "/root/LightBench/lightbench/run_all_benchmarks.py", line 131, in run_benchmark
    module.main(**arguments)
  File "/root/LightBench/lightbench/quadratic_quantized_gradient.py", line 54, in main
    model = Model(**kwargs).cuda()
            ^^^^^^^^^^^^^^^
TypeError: Model.__init__() missing 1 required positional argument: 'bins'

```

### quadratic_quantized_gradient-nightmare - SOAP (fp16+ecc)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-nightmare - SOAPAdEMAMix (fp16+ecc)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_quantized_gradient-nightmare - LATHER (fp32)
```
Gated after quadratic_quantized_gradient-trivial failure
```

### quadratic_sparse-easy - SOAPNAdam (fp32)
```
Gated after quadratic_sparse-trivial failure
```

### quadratic_sparse-extreme - PSGDLRA (bf16)
```
Gated after quadratic_sparse-medium failure
```

### quadratic_sparse-extreme - AdEMAMix (bf16+ecc)
```
Gated after quadratic_sparse-medium failure
```

### quadratic_sparse-extreme - SOAPNAdam (fp32)
```
Gated after quadratic_sparse-trivial failure
```

### quadratic_sparse-hard - PSGDLRA (bf16)
```
Gated after quadratic_sparse-medium failure
```

### quadratic_sparse-hard - AdEMAMix (bf16+ecc)
```
Gated after quadratic_sparse-medium failure
```

### quadratic_sparse-hard - SOAPNAdam (fp32)
```
Gated after quadratic_sparse-trivial failure
```

### quadratic_sparse-medium - SOAPNAdam (fp32)
```
Gated after quadratic_sparse-trivial failure
```

### quadratic_sparse-nightmare - PSGDLRA (bf16)
```
Gated after quadratic_sparse-medium failure
```

### quadratic_sparse-nightmare - AdEMAMix (bf16+ecc)
```
Gated after quadratic_sparse-medium failure
```

### quadratic_sparse-nightmare - SOAPNAdam (fp32)
```
Gated after quadratic_sparse-trivial failure
```

### quadratic_varying_scale-extreme - PSGDLRA (bf16)
```
Gated after quadratic_varying_scale-medium failure
```

### quadratic_varying_scale-extreme - SAMWrapper (bf16+ecc)
```
Gated after quadratic_varying_scale-easy failure
```

### quadratic_varying_scale-hard - PSGDLRA (bf16)
```
Gated after quadratic_varying_scale-medium failure
```

### quadratic_varying_scale-hard - SAMWrapper (bf16+ecc)
```
Gated after quadratic_varying_scale-easy failure
```

### quadratic_varying_scale-medium - SAMWrapper (bf16+ecc)
```
Gated after quadratic_varying_scale-easy failure
```

### quadratic_varying_scale-nightmare - PSGDLRA (bf16)
```
Gated after quadratic_varying_scale-medium failure
```

### quadratic_varying_scale-nightmare - SAMWrapper (bf16+ecc)
```
Gated after quadratic_varying_scale-easy failure
```

### quadratic_varying_target-extreme - AdamW (bf16+ecc)
```
Gated after quadratic_varying_target-easy failure
```

### quadratic_varying_target-extreme - SUDSAdamW (bf16+ecc)
```
Gated after quadratic_varying_target-easy failure
```

### quadratic_varying_target-extreme - SOAPNAdam (fp16+ecc)
```
Gated after quadratic_varying_target-easy failure
```

### quadratic_varying_target-hard - AdamW (bf16+ecc)
```
Gated after quadratic_varying_target-easy failure
```

### quadratic_varying_target-hard - SUDSAdamW (bf16+ecc)
```
Gated after quadratic_varying_target-easy failure
```

### quadratic_varying_target-hard - SOAPNAdam (fp16+ecc)
```
Gated after quadratic_varying_target-easy failure
```

### quadratic_varying_target-medium - AdamW (bf16+ecc)
```
Gated after quadratic_varying_target-easy failure
```

### quadratic_varying_target-medium - SUDSAdamW (bf16+ecc)
```
Gated after quadratic_varying_target-easy failure
```

### quadratic_varying_target-medium - SOAPNAdam (fp16+ecc)
```
Gated after quadratic_varying_target-easy failure
```

### quadratic_varying_target-nightmare - AdamW (bf16+ecc)
```
Gated after quadratic_varying_target-easy failure
```

### quadratic_varying_target-nightmare - SUDSAdamW (bf16+ecc)
```
Gated after quadratic_varying_target-easy failure
```

### quadratic_varying_target-nightmare - SOAPNAdam (fp16+ecc)
```
Gated after quadratic_varying_target-easy failure
```

### rastrigin-easy - RMSprop (bf16)
```
Gated after rastrigin-trivial failure
```

### rastrigin-easy - LATHER (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### rastrigin-easy - PSGDLRA (fp16)
```
Gated after rastrigin-trivial failure
```

### rastrigin-easy - SGD (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### rastrigin-extreme - RMSprop (bf16)
```
Gated after rastrigin-trivial failure
```

### rastrigin-extreme - LATHER (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### rastrigin-extreme - PSGDLRA (fp16)
```
Gated after rastrigin-trivial failure
```

### rastrigin-extreme - SGD (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### rastrigin-hard - RMSprop (bf16)
```
Gated after rastrigin-trivial failure
```

### rastrigin-hard - LATHER (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### rastrigin-hard - PSGDLRA (fp16)
```
Gated after rastrigin-trivial failure
```

### rastrigin-hard - SGD (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### rastrigin-medium - RMSprop (bf16)
```
Gated after rastrigin-trivial failure
```

### rastrigin-medium - LATHER (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### rastrigin-medium - PSGDLRA (fp16)
```
Gated after rastrigin-trivial failure
```

### rastrigin-medium - SGD (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### rastrigin-nightmare - RMSprop (bf16)
```
Gated after rastrigin-trivial failure
```

### rastrigin-nightmare - LATHER (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### rastrigin-nightmare - PSGDLRA (fp16)
```
Gated after rastrigin-trivial failure
```

### rastrigin-nightmare - SGD (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-easy - AdEMAMix (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-easy - Muon (bf16)
```
Gated after rosenbrock-trivial failure
```

### rosenbrock-easy - AdamC (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-easy - ADOPT (fp16)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-extreme - AdEMAMix (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-extreme - Muon (bf16)
```
Gated after rosenbrock-trivial failure
```

### rosenbrock-extreme - AdamC (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-extreme - ADOPT (fp16)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-hard - AdEMAMix (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-hard - Muon (bf16)
```
Gated after rosenbrock-trivial failure
```

### rosenbrock-hard - AdamC (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-hard - ADOPT (fp16)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-medium - AdEMAMix (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-medium - Muon (bf16)
```
Gated after rosenbrock-trivial failure
```

### rosenbrock-medium - AdamC (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-medium - ADOPT (fp16)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-nightmare - AdEMAMix (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-nightmare - Muon (bf16)
```
Gated after rosenbrock-trivial failure
```

### rosenbrock-nightmare - AdamC (bf16+ecc)
```
Skipped config: 'config' must be 'trivial'.
```

### rosenbrock-nightmare - ADOPT (fp16)
```
Skipped config: 'config' must be 'trivial'.
```

### saddle_point-easy - ADOPT (fp32)
```
Gated after saddle_point-trivial failure
```

### saddle_point-easy - OrthoLaProp (fp32)
```
Gated after saddle_point-trivial failure
```

### saddle_point-extreme - ADOPT (fp32)
```
Gated after saddle_point-trivial failure
```

### saddle_point-extreme - OrthoLaProp (fp32)
```
Gated after saddle_point-trivial failure
```

### saddle_point-hard - ADOPT (fp32)
```
Gated after saddle_point-trivial failure
```

### saddle_point-hard - OrthoLaProp (fp32)
```
Gated after saddle_point-trivial failure
```

### saddle_point-medium - ADOPT (fp32)
```
Gated after saddle_point-trivial failure
```

### saddle_point-medium - OrthoLaProp (fp32)
```
Gated after saddle_point-trivial failure
```

### saddle_point-nightmare - PSGDPRO (bf16)
```
Gated after saddle_point-extreme failure
```

### saddle_point-nightmare - SOLP (fp16+ecc)
```
Gated after saddle_point-extreme failure
```

### saddle_point-nightmare - ADOPT (fp32)
```
Gated after saddle_point-trivial failure
```

### saddle_point-nightmare - OrthoLaProp (fp32)
```
Gated after saddle_point-trivial failure
```

### scale_invariant-extreme - SUDSAdamW (bf16)
```
Gated after scale_invariant-hard failure
```

### scale_invariant-extreme - SFAdamW (fp16+ecc)
```
Gated after scale_invariant-easy failure
```

### scale_invariant-extreme - LATHER (fp32)
```
Gated after scale_invariant-hard failure
```

### scale_invariant-extreme - PSGDLRA (fp32)
```
Gated after scale_invariant-easy failure
```

### scale_invariant-extreme - SFAdamW (fp32)
```
Gated after scale_invariant-hard failure
```

### scale_invariant-hard - SFAdamW (fp16+ecc)
```
Gated after scale_invariant-easy failure
```

### scale_invariant-hard - PSGDLRA (fp32)
```
Gated after scale_invariant-easy failure
```

### scale_invariant-medium - SFAdamW (fp16+ecc)
```
Gated after scale_invariant-easy failure
```

### scale_invariant-medium - PSGDLRA (fp32)
```
Gated after scale_invariant-easy failure
```

### scale_invariant-nightmare - SUDSAdamW (bf16)
```
Gated after scale_invariant-hard failure
```

### scale_invariant-nightmare - AdamC (bf16+ecc)
```
Gated after scale_invariant-extreme failure
```

### scale_invariant-nightmare - SFAdamW (fp16+ecc)
```
Gated after scale_invariant-easy failure
```

### scale_invariant-nightmare - LATHER (fp32)
```
Gated after scale_invariant-hard failure
```

### scale_invariant-nightmare - PSGDLRA (fp32)
```
Gated after scale_invariant-easy failure
```

### scale_invariant-nightmare - SFAdamW (fp32)
```
Gated after scale_invariant-hard failure
```

### sparse_gradient-easy - SOAPAdEMAMix (bf16)
```
Gated after sparse_gradient-trivial failure
```

### sparse_gradient-easy - AdEMAMix (fp16+ecc)
```
Gated after sparse_gradient-trivial failure
```

### sparse_gradient-extreme - SOAPAdEMAMix (bf16)
```
Gated after sparse_gradient-trivial failure
```

### sparse_gradient-extreme - AdEMAMix (fp16+ecc)
```
Gated after sparse_gradient-trivial failure
```

### sparse_gradient-hard - SOAPAdEMAMix (bf16)
```
Gated after sparse_gradient-trivial failure
```

### sparse_gradient-hard - AdEMAMix (fp16+ecc)
```
Gated after sparse_gradient-trivial failure
```

### sparse_gradient-medium - SOAPAdEMAMix (bf16)
```
Gated after sparse_gradient-trivial failure
```

### sparse_gradient-medium - AdEMAMix (fp16+ecc)
```
Gated after sparse_gradient-trivial failure
```

### sparse_gradient-nightmare - SOAPAdEMAMix (bf16)
```
Gated after sparse_gradient-trivial failure
```

### sparse_gradient-nightmare - AdEMAMix (fp16+ecc)
```
Gated after sparse_gradient-trivial failure
```

### spiral-easy - PSGDPRO (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-easy - SGD (bf16+ecc)
```
Gated after spiral-trivial failure
```

### spiral-easy - PSGDKron (fp16)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-easy - MuonLaProp (fp32)
```
Gated after spiral-trivial failure
```

### spiral-easy - Scion (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-extreme - PSGDPRO (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-extreme - SGD (bf16+ecc)
```
Gated after spiral-trivial failure
```

### spiral-extreme - PSGDKron (fp16)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-extreme - MuonLaProp (fp32)
```
Gated after spiral-trivial failure
```

### spiral-extreme - Scion (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-hard - PSGDPRO (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-hard - SGD (bf16+ecc)
```
Gated after spiral-trivial failure
```

### spiral-hard - PSGDKron (fp16)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-hard - MuonLaProp (fp32)
```
Gated after spiral-trivial failure
```

### spiral-hard - Scion (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-medium - PSGDPRO (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-medium - SGD (bf16+ecc)
```
Gated after spiral-trivial failure
```

### spiral-medium - PSGDKron (fp16)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-medium - MuonLaProp (fp32)
```
Gated after spiral-trivial failure
```

### spiral-medium - Scion (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-nightmare - PSGDPRO (bf16)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-nightmare - SGD (bf16+ecc)
```
Gated after spiral-trivial failure
```

### spiral-nightmare - PSGDKron (fp16)
```
Skipped config: 'config' must be 'trivial'.
```

### spiral-nightmare - MuonLaProp (fp32)
```
Gated after spiral-trivial failure
```

### spiral-nightmare - Scion (fp32)
```
Skipped config: 'config' must be 'trivial'.
```

### transfer_domain_shift-easy - SOAPAdEMAMix (bf16)
```
Gated after transfer_domain_shift-trivial failure
```

### transfer_domain_shift-easy - PSGDPRO (fp16+ecc)
```
Gated after transfer_domain_shift-trivial failure
```

### transfer_domain_shift-extreme - SOAPAdEMAMix (bf16)
```
Gated after transfer_domain_shift-trivial failure
```

### transfer_domain_shift-extreme - PSGDPRO (fp16+ecc)
```
Gated after transfer_domain_shift-trivial failure
```

### transfer_domain_shift-hard - SOAPAdEMAMix (bf16)
```
Gated after transfer_domain_shift-trivial failure
```

### transfer_domain_shift-hard - PSGDPRO (fp16+ecc)
```
Gated after transfer_domain_shift-trivial failure
```

### transfer_domain_shift-medium - SOAPAdEMAMix (bf16)
```
Gated after transfer_domain_shift-trivial failure
```

### transfer_domain_shift-medium - PSGDPRO (fp16+ecc)
```
Gated after transfer_domain_shift-trivial failure
```

### transfer_domain_shift-nightmare - SOAPAdEMAMix (bf16)
```
Gated after transfer_domain_shift-trivial failure
```

### transfer_domain_shift-nightmare - PSGDPRO (fp16+ecc)
```
Gated after transfer_domain_shift-trivial failure
```

### vanishing_gradient-easy - PSGDKron (fp16+ecc)
```
Gated after vanishing_gradient-trivial failure
```

### vanishing_gradient-easy - MSAMLaProp (fp32)
```
Gated after vanishing_gradient-trivial failure
```

### vanishing_gradient-extreme - PSGDKron (fp16+ecc)
```
Gated after vanishing_gradient-trivial failure
```

### vanishing_gradient-extreme - MSAMLaProp (fp32)
```
Gated after vanishing_gradient-trivial failure
```

### vanishing_gradient-hard - PSGDKron (fp16+ecc)
```
Gated after vanishing_gradient-trivial failure
```

### vanishing_gradient-hard - MSAMLaProp (fp32)
```
Gated after vanishing_gradient-trivial failure
```

### vanishing_gradient-medium - PSGDKron (fp16+ecc)
```
Gated after vanishing_gradient-trivial failure
```

### vanishing_gradient-medium - MSAMLaProp (fp32)
```
Gated after vanishing_gradient-trivial failure
```

### vanishing_gradient-nightmare - PSGDKron (fp16+ecc)
```
Gated after vanishing_gradient-trivial failure
```

### vanishing_gradient-nightmare - MSAMLaProp (fp32)
```
Gated after vanishing_gradient-trivial failure
```

### wide_linear-easy - AdEMAMix (bf16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-easy - SOAP (bf16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-easy - PSGDPRO (bf16+ecc)
```
Gated after wide_linear-trivial failure
```

### wide_linear-easy - PSGDPRO (fp16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-easy - SAMWrapper (fp16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-easy - PSGDPRO (fp32)
```
Gated after wide_linear-trivial failure
```

### wide_linear-easy - SAMWrapper (fp32)
```
Gated after wide_linear-trivial failure
```

### wide_linear-extreme - AdEMAMix (bf16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-extreme - SOAP (bf16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-extreme - PSGDPRO (bf16+ecc)
```
Gated after wide_linear-trivial failure
```

### wide_linear-extreme - PSGDPRO (fp16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-extreme - SAMWrapper (fp16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-extreme - PSGDPRO (fp32)
```
Gated after wide_linear-trivial failure
```

### wide_linear-extreme - SAMWrapper (fp32)
```
Gated after wide_linear-trivial failure
```

### wide_linear-hard - AdEMAMix (bf16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-hard - SOAP (bf16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-hard - PSGDPRO (bf16+ecc)
```
Gated after wide_linear-trivial failure
```

### wide_linear-hard - PSGDPRO (fp16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-hard - SAMWrapper (fp16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-hard - PSGDPRO (fp32)
```
Gated after wide_linear-trivial failure
```

### wide_linear-hard - SAMWrapper (fp32)
```
Gated after wide_linear-trivial failure
```

### wide_linear-medium - AdEMAMix (bf16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-medium - SOAP (bf16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-medium - PSGDPRO (bf16+ecc)
```
Gated after wide_linear-trivial failure
```

### wide_linear-medium - PSGDPRO (fp16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-medium - SAMWrapper (fp16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-medium - PSGDPRO (fp32)
```
Gated after wide_linear-trivial failure
```

### wide_linear-medium - SAMWrapper (fp32)
```
Gated after wide_linear-trivial failure
```

### wide_linear-nightmare - AdEMAMix (bf16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-nightmare - SOAP (bf16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-nightmare - PSGDPRO (bf16+ecc)
```
Gated after wide_linear-trivial failure
```

### wide_linear-nightmare - PSGDPRO (fp16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-nightmare - SAMWrapper (fp16)
```
Gated after wide_linear-trivial failure
```

### wide_linear-nightmare - PSGDPRO (fp32)
```
Gated after wide_linear-trivial failure
```

### wide_linear-nightmare - SAMWrapper (fp32)
```
Gated after wide_linear-trivial failure
```

### xor_digit-easy - SUDSAdamW (fp16)
```
Gated after xor_digit-trivial failure
```

### xor_digit-extreme - SAMWrapper (bf16+ecc)
```
Gated after xor_digit-medium failure
```

### xor_digit-extreme - MSAMLaProp (fp16)
```
Gated after xor_digit-medium failure
```

### xor_digit-extreme - SUDSAdamW (fp16)
```
Gated after xor_digit-trivial failure
```

### xor_digit-extreme - SAMWrapper (fp16+ecc)
```
Gated after xor_digit-medium failure
```

### xor_digit-hard - SAMWrapper (bf16+ecc)
```
Gated after xor_digit-medium failure
```

### xor_digit-hard - MSAMLaProp (fp16)
```
Gated after xor_digit-medium failure
```

### xor_digit-hard - SUDSAdamW (fp16)
```
Gated after xor_digit-trivial failure
```

### xor_digit-hard - SAMWrapper (fp16+ecc)
```
Gated after xor_digit-medium failure
```

### xor_digit-medium - SUDSAdamW (fp16)
```
Gated after xor_digit-trivial failure
```

### xor_digit-nightmare - SAMWrapper (bf16+ecc)
```
Gated after xor_digit-medium failure
```

### xor_digit-nightmare - MSAMLaProp (fp16)
```
Gated after xor_digit-medium failure
```

### xor_digit-nightmare - SUDSAdamW (fp16)
```
Gated after xor_digit-trivial failure
```

### xor_digit-nightmare - SAMWrapper (fp16+ecc)
```
Gated after xor_digit-medium failure
```

### xor_digit_rnn-easy - LaProp (fp16+ecc)
```
Gated after xor_digit_rnn-trivial failure
```

### xor_digit_rnn-easy - PSGDKron (fp32)
```
Gated after xor_digit_rnn-trivial failure
```

### xor_digit_rnn-extreme - SOLP (fp16)
```
Gated after xor_digit_rnn-easy failure
```

### xor_digit_rnn-extreme - LaProp (fp16+ecc)
```
Gated after xor_digit_rnn-trivial failure
```

### xor_digit_rnn-extreme - PSGDKron (fp32)
```
Gated after xor_digit_rnn-trivial failure
```

### xor_digit_rnn-hard - SOLP (fp16)
```
Gated after xor_digit_rnn-easy failure
```

### xor_digit_rnn-hard - LaProp (fp16+ecc)
```
Gated after xor_digit_rnn-trivial failure
```

### xor_digit_rnn-hard - PSGDKron (fp32)
```
Gated after xor_digit_rnn-trivial failure
```

### xor_digit_rnn-medium - SOLP (fp16)
```
Gated after xor_digit_rnn-easy failure
```

### xor_digit_rnn-medium - LaProp (fp16+ecc)
```
Gated after xor_digit_rnn-trivial failure
```

### xor_digit_rnn-medium - PSGDKron (fp32)
```
Gated after xor_digit_rnn-trivial failure
```

### xor_digit_rnn-nightmare - SOLP (fp16)
```
Gated after xor_digit_rnn-easy failure
```

### xor_digit_rnn-nightmare - LaProp (fp16+ecc)
```
Gated after xor_digit_rnn-trivial failure
```

### xor_digit_rnn-nightmare - PSGDKron (fp32)
```
Gated after xor_digit_rnn-trivial failure
```

### xor_sequence_rnn-extreme - SignLaProp (bf16+ecc)
```
Gated after xor_sequence_rnn-easy failure
```

### xor_sequence_rnn-extreme - NAdam (fp16+ecc)
```
Gated after xor_sequence_rnn-hard failure
```

### xor_sequence_rnn-extreme - SGD (fp16+ecc)
```
Gated after xor_sequence_rnn-medium failure
```

### xor_sequence_rnn-extreme - AdamC (fp32)
```
Gated after xor_sequence_rnn-medium failure
```

### xor_sequence_rnn-hard - SignLaProp (bf16+ecc)
```
Gated after xor_sequence_rnn-easy failure
```

### xor_sequence_rnn-hard - SGD (fp16+ecc)
```
Gated after xor_sequence_rnn-medium failure
```

### xor_sequence_rnn-hard - AdamC (fp32)
```
Gated after xor_sequence_rnn-medium failure
```

### xor_sequence_rnn-medium - SignLaProp (bf16+ecc)
```
Gated after xor_sequence_rnn-easy failure
```

### xor_sequence_rnn-nightmare - SignLaProp (bf16+ecc)
```
Gated after xor_sequence_rnn-easy failure
```

### xor_sequence_rnn-nightmare - NAdam (fp16+ecc)
```
Gated after xor_sequence_rnn-hard failure
```

### xor_sequence_rnn-nightmare - SGD (fp16+ecc)
```
Gated after xor_sequence_rnn-medium failure
```

### xor_sequence_rnn-nightmare - AdamC (fp32)
```
Gated after xor_sequence_rnn-medium failure
```

### xor_spot-extreme - HyperBallAdamW (fp16+ecc)
```
Gated after xor_spot-easy failure
```

### xor_spot-extreme - SignLaProp (fp32)
```
Gated after xor_spot-easy failure
```

### xor_spot-extreme - UnscaledAdamW (fp32)
```
Gated after xor_spot-easy failure
```

### xor_spot-hard - HyperBallAdamW (fp16+ecc)
```
Gated after xor_spot-easy failure
```

### xor_spot-hard - SignLaProp (fp32)
```
Gated after xor_spot-easy failure
```

### xor_spot-hard - UnscaledAdamW (fp32)
```
Gated after xor_spot-easy failure
```

### xor_spot-medium - HyperBallAdamW (fp16+ecc)
```
Gated after xor_spot-easy failure
```

### xor_spot-medium - SignLaProp (fp32)
```
Gated after xor_spot-easy failure
```

### xor_spot-medium - UnscaledAdamW (fp32)
```
Gated after xor_spot-easy failure
```

### xor_spot-nightmare - HyperBallAdamW (fp16+ecc)
```
Gated after xor_spot-easy failure
```

### xor_spot-nightmare - SignLaProp (fp32)
```
Gated after xor_spot-easy failure
```

### xor_spot-nightmare - UnscaledAdamW (fp32)
```
Gated after xor_spot-easy failure
```

### xor_spot_rnn-extreme - OrthoLaProp (bf16+ecc)
```
Gated after xor_spot_rnn-easy failure
```

### xor_spot_rnn-extreme - Muon (fp32)
```
Gated after xor_spot_rnn-medium failure
```

### xor_spot_rnn-hard - OrthoLaProp (bf16+ecc)
```
Gated after xor_spot_rnn-easy failure
```

### xor_spot_rnn-hard - Muon (fp32)
```
Gated after xor_spot_rnn-medium failure
```

### xor_spot_rnn-medium - OrthoLaProp (bf16+ecc)
```
Gated after xor_spot_rnn-easy failure
```

### xor_spot_rnn-nightmare - OrthoLaProp (bf16+ecc)
```
Gated after xor_spot_rnn-easy failure
```

### xor_spot_rnn-nightmare - Muon (fp32)
```
Gated after xor_spot_rnn-medium failure
```
