# Implementation and recovery notes

- Existing DiskANN sources already contain local changes. E1 builds a separate library and collector under native/; existing source and executable files were preserved.
- Initial isolated compilation required copying online_termination.h alongside the isolated online_gate.h to avoid duplicate class definitions, declaring the reader as shared_ptr<AlignedFileReader>, and linking libatomic. Failed compiler output remains in build.log.
- matplotlib was not installed and network installation was unavailable. Charts use existing native Cairo bindings and export PNG, SVG and PDF without new dependencies.
- The original validation cutoff99.603% selected tau0 for MLP. The cutoff was amended to discrete99.60% based on validation only; original pilot run artifacts remain in runs/ and pilot_*.json. Main artifacts use runs_main/.
- A pilot profile check exposed a gap between two clock reads that could include preemption. The collector now records the profile total from the same endpoint as its exclusive category sum. Main closure errors are saved in verification.json.
- The first analysis encountered an empty far-distance intersection. Its report formatter initially failed on the undefined metric; this was corrected to N/A, with zero-overlap and zero-Skip denominator rates retained. No missing values were replaced with zero agreement.
- Source changes and recovery do not alter frozen main search thresholds or collected main search outputs. Trace/profile/timed output parity and native/Python scores are verified.
