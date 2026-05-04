# NeurIPS Core Public-Baseline Summary

Scope: CIFAR-100/ResNet18, non-i.i.d., 20 clients, 20% malicious clients, 200 rounds, attacks ALIE/FangAttack/MinMax/MinSum.
Excluded: TriGuardFL and all stress-test settings.

Seed coverage:
- CARAT: final_locked_protocol, n=3, seeds=42,43,44
- FLTrust: final_locked_protocol, n=3, seeds=42,43,44
- TrimmedMean: final_locked_protocol, n=3, seeds=42,43,44
- FLDetector: legacy_public_baseline, n=5, seeds=42,43,44,45,46
- Mean: legacy_public_baseline, n=5, seeds=42,43,44,45,46
- MultiKrum: legacy_public_baseline, n=5, seeds=42,43,44,45,46
- NormClipping: legacy_public_baseline, n=5, seeds=42,43,44,45,46

Main-text accuracy observations:
- alpha=1: best average rank in the main-text subset is CARAT; CARAT main-text average rank=1.0.
- alpha=0.5: best average rank in the main-text subset is CARAT; CARAT main-text average rank=1.0.

Complete appendix accuracy observations:
- alpha=1: best average rank is TrimmedMean; CARAT average rank=2.0, mean final accuracy=53.67%.
- alpha=0.5: best average rank is TrimmedMean; CARAT average rank=2.0, mean final accuracy=52.42%.
