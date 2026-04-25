# Responses to Editorial Questions for TIFS

## 1. Why is this manuscript within the scope of IEEE Transactions on Information Forensics \& Security?

This manuscript is within the scope of *IEEE Transactions on Information Forensics and Security (TIFS)* because it addresses a core information security problem: preserving the integrity of federated learning against malicious model-poisoning participants. The contribution is not a generic machine learning method; rather, it is a security mechanism for detecting and mitigating Byzantine/adversarial clients in a distributed collaborative learning system. In federated learning, malicious users can upload manipulated model updates to corrupt the global model, which is fundamentally an attack on the integrity and trustworthiness of the system. Our work studies how to identify such malicious behavior, distinguish benign from adversarial clients, and robustly aggregate updates under attack. These problems fall directly within information security and information forensics, since the server must analyze suspicious evidence in client updates and make trust decisions under adversarial conditions. The proposed method, TriGuardFL, contributes a three-stage defense pipeline: suspicious-client screening, class-wise statistical malicious-client detection, and reputation-aware robust aggregation. Therefore, the paper fits TIFS because it advances security and forensics for distributed AI systems rather than merely improving machine learning accuracy.

## 2. Why is the contribution significant (What impact will it have)?

The contribution is significant because it addresses a practically important and difficult problem: defending federated learning against strong model-poisoning attacks under non-IID data, where many existing defenses degrade. The proposed method is impactful in three respects. First, it improves robustness against strong attacks by combining multiple complementary signals rather than relying on a single brittle rule. Second, it introduces a class-wise behavioral validation stage that improves interpretability and strengthens malicious-client detection. Third, it incorporates cross-round reputation into aggregation, which provides a realistic trust-management mechanism for long-running federated systems. The expected impact is to improve the security, reliability, and deployability of federated learning in sensitive applications such as healthcare, finance, mobile intelligence, and edge systems, where poisoned global models can cause serious downstream harm.

## 3. What are the three papers in the published literature most closely related to this paper?

1. Xiaoyu Cao, Minghong Fang, Jia Liu, and Neil Zhenqiang Gong, “FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping,” in *Proceedings of the Network and Distributed System Security Symposium (NDSS)*, 2021.  
   URL: https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/  
   DOI: not listed on the NDSS publication page.

2. Zaixi Zhang, Xiaoyu Cao, Jinyuan Jia, and Neil Zhenqiang Gong, “FLDetector: Defending Federated Learning Against Model Poisoning Attacks via Detecting Malicious Clients,” in *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’22)*, pp. 2545–2555, 2022.  
   DOI: https://doi.org/10.1145/3534678.3539231

3. Yongkang Wang, Di-Hua Zhai, and Yuanqing Xia, “RFVIR: A Robust Federated Algorithm Defending Against Byzantine Attacks,” *Information Fusion*, vol. 105, article 102251, 2024.  
   DOI: https://doi.org/10.1016/j.inffus.2024.102251

## 4. What is distinctive/new about the current paper relative to these previously published works?

Relative to FLTrust, the current paper goes beyond trust bootstrapping from a server reference and introduces a class-wise statistical validation stage plus cross-round reputation-aware aggregation. Relative to FLDetector, the current paper does not rely mainly on temporal inconsistency in client updates; instead, it performs server-side behavioral detection against a benign reference model and integrates this with robust aggregation. Relative to RFVIR, the current paper is distinctive in explicitly structuring the defense into three coupled stages: coarse suspicious-client screening, class-wise statistical detection, and reputation-aware aggregation over time. The novelty of the current paper is therefore the unified integration of geometric screening, behavioral validation, and long-term trust management into a single defense framework for poisoning-resilient federated learning under heterogeneous non-IID data.
