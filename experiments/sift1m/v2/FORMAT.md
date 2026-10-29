# LGTRACE2 fields and timing

Enable with `DISKANN_TRACE_VERSION=2` and `DISKANN_TRACE_PREFIX=<prefix>`. Without version=2, LGTRACE1 remains the default. Original v1 trace files are unchanged. This experiment uses unfiltered float L2, no disk PQ, no reorder, no skip masks.

File: magic `LGTRACE2` (8 bytes); repeated blocks: 5 little-endian uint32 [query_id,L,final_visited_count,n_expansions,n_edges], n_expansions x 24-byte v1 expansion records, n_edges x 88-byte edge records. Exact field order and dtype are in `read_trace.py`. Every field is explicitly serialized; query and step IDs are repeated on every edge. Blocks can appear in arbitrary query order due to OpenMP.

| Field | Meaning |
|---|---|
| query_id | Zero-based query index |
| step_id | Zero-based expansion processing order within query (cached neighborhoods are processed before disk neighborhoods in each beam) |
| beam_step | Zero-based original search-loop/beam iteration |
| current_node_id / neighbor_id | Expanded parent and scanned neighbor |
| current_distance_exact | Exact squared L2(query,current_node), observed when current node is scored |
| current_distance_pq | PQ approximate squared L2(query,current_node) |
| neighbor_distance_pq | PQ approximate squared L2(query,neighbor), actually used by search |
| neighbor_distance_exact | Exact squared L2(query,neighbor), annotated offline from original vectors for EVERY edge; never used to alter search |
| current_best_distance | Minimum PQ distance in candidate queue BEFORE this edge's visited check/insertion |
| current_best_exact_distance | Best exact scored-node distance so far, INCLUDING current node |
| neighbor_rank | Zero-based rank among this parent's full neighborhood by (PQ distance, neighbor ID, adjacency position); includes already visited neighbors |
| neighbor_position | Zero-based original adjacency-list position |
| queue_size | Candidate queue size BEFORE this edge, including candidates already marked expanded |
| visited_count | Distinct visited count BEFORE this edge, including seed medoid |
| flags | Bit0=first visit, bit1=admitted |
| neighbor_was_inserted | Whether THIS edge actually admitted its neighbor to bounded candidate queue |
| neighbor_was_expanded | Whether this neighbor node was scored/expanded anywhere in this query; post-query node label, not necessarily after this edge |
| neighbor_in_final_topk | Whether this neighbor ID belongs to actual returned top10; true for repeated edges too |
| neighbor_improved_best_pq | THIS edge was admitted and strictly reduced best PQ distance (ties are not improvements) |
| neighbor_improved_best_exact | THIS edge first discovered a node that later strictly improved the running best exact scored distance when expanded |
| neighbor_led_to_improvement | THIS edge first discovered a node whose own expansion or first-discovery descendants later strictly improved the running best exact distance |

The initial medoid establishing an incumbent from infinity is excluded from improvement labels. Improvements concern minimum exact distance, not changes in top-k cutoff, Recall@10, or a guaranteed causal effect of skipping an edge. Descendant attribution uses the actual first-discovery tree; duplicate later edges get no discovery credit. Both direct exact and descendant labels are filled after query completion in the logger. Exact neighbor distances are filled by `run_v2.py` after raw capture and are ready only when `COMPLETE.json` reports completion. Raw exact-neighbor slots are NaN before annotation. Future labels and offline exact neighbor distances are not online gate inputs.

Snapshots are taken before each edge. The current node has already been read and scored, and neighborhood PQ distances have been computed. A gate BEFORE neighborhood read cannot directly use these fields without paying for their computation.

Run: `../../../.venv/bin/python run_v2.py` from this directory. Resume annotation/validation without repeating searches: same command with `--analyze-only`. Export all rows of a query: `../../../.venv/bin/python read_trace.py --trace trace_100.trace --query-id 0 --output query0_L100.csv`.

Validation: complete 10000-query coverage for every L; exact baseline parity with logging enabled AND disabled; degree/edge totals; per-step ranks; pre-edge visited and queue size sequences; monotonic best PQ/exact snapshots; independent first-discovery improvement and final/expanded labels; exact distances checked against original vectors; bounded-queue admission replay for 32 fixed queries per L. `edges_sample.csv` contains 200 edges from query 0 per budget; complete rows are in binary trace, exportable with the reader.
