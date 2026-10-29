# Decision Log

## D001: Separate Research and Source Repositories
The Gate repository stores research coordination documents and project-level files.
The DiskANN fork stores the modified DiskANN source code.

## D002: Keep Large Experiment Artifacts Out of Git by Default
Datasets, indexes, large traces, model binaries, and generated outputs remain local
unless there is an explicit reason to version a particular artifact.

## D003: Separate Implementation from Validation
A successful Git commit does not prove build correctness, algorithm correctness,
or performance improvement. Each claim requires corresponding evidence.
