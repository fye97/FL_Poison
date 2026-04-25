# Concise Responses to Editorial Questions for TIFS

## 1. Scope
This paper is in scope because it addresses an information security problem: protecting federated learning from Byzantine/model-poisoning clients. It contributes a defense for detecting malicious updates, separating benign from adversarial participants, and preserving model integrity under attack. This aligns with TIFS interests in information security, trust, and forensics for adversarial systems.

## 2. Significance
The work is significant because it targets strong poisoning attacks under realistic non-IID data, where many defenses weaken. TriGuardFL combines suspicious-client screening, class-wise behavioral validation, and reputation-aware aggregation. Its impact is to improve the security and deployability of federated learning in high-stakes domains such as healthcare, finance, mobile/edge intelligence, and other adversarial collaborative systems.

## 3. Closest Papers
Closest works are FLTrust (Cao et al., NDSS 2021), FLDetector (Zhang et al., KDD 2022, DOI: 10.1145/3534678.3539231), and RFVIR (Wang et al., Information Fusion 2024, DOI: 10.1016/j.inffus.2024.102251). FLTrust URL: https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/

## 4. Novelty
Unlike FLTrust, this paper adds class-wise statistical validation and cross-round reputation-aware aggregation. Unlike FLDetector, it does not mainly rely on temporal inconsistency, but uses server-side behavioral detection against a benign reference. Unlike RFVIR, it unifies three explicit stages: coarse screening, class-wise statistical detection, and long-term reputation-guided robust aggregation.
