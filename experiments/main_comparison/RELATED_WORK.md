# Related work and scope of the termination control

- [DARTH: official repository](https://github.com/MChatzakis/DARTH) integrates recall prediction and declarative early termination into Faiss, with HNSW and IVF drivers. Its published speedups are not DiskANN SSD request measurements.
- [Learned Adaptive Early Termination: official repository](https://github.com/efficient/faiss-learned-termination) is based on Faiss and LightGBM; its documented implementation modifies HNSW and IVF. Its published performance is not inserted into this experiment's numerical comparison.

This experiment implements a comparable learned termination strategy **inside the same DiskANN search loop**: a 17-feature MLP estimates the current exact-scored prefix's GT Recall@10; the full search stops at a beam boundary once predicted recall reaches a tunable threshold, and only after at least K exact candidates exist. Ground truth is used only to create training labels and score returned results, never as inference input. The estimator is trained on mixed-L baseline trajectories from training-only queries.

This is a DiskANN-native learned recall termination control inspired by the related strategy class, **not** a faithful reproduction or performance claim for DARTH or LAET. It uses neither external HNSW/IVF numbers nor an oracle stopping condition. ET+pruning runs both real models on live changed trajectories and includes both inference costs.

The generic ET control predicts current recall; it differs from the previously available L100 safe-stop classifier, whose target was completion of the full baseline's final top10. New model versions and training manifests are stored separately from all previous experiments.
