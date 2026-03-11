.PHONY: install test test-fast test-full bench-ann bench-sweep bench-ingest bench-mixed soak-mixed lint clean

DEEPDATA_PORT ?= 8080

install:
	pip install -e .

# Fast CI: correctness + tiny ANN (< 5 min)
test-fast:
	DEEPDATA_TESTS=1 DEEPDATA_PORT=$(DEEPDATA_PORT) pytest tests/ -q --timeout=120

# Full test suite including slow tests
test-full:
	DEEPDATA_TESTS=1 DEEPDATA_PORT=$(DEEPDATA_PORT) pytest tests/ -v --timeout=300

# ANN recall benchmark
bench-ann:
	python bench/ann_recall.py \
		--dataset-name sift-10k \
		--base datasets/cache/sift/sift_base.fvecs \
		--queries datasets/cache/sift/sift_query.fvecs \
		--gt datasets/cache/sift/sift_gt100.npy \
		--dim 128 --n-base 10000 \
		--port $(DEEPDATA_PORT) \
		--json artifacts/runs/latest_ann.json

# HNSW parameter sweep
bench-sweep:
	python bench/ann_param_sweep.py \
		--dataset-name sift-10k \
		--base datasets/cache/sift/sift_base.fvecs \
		--queries datasets/cache/sift/sift_query.fvecs \
		--gt datasets/cache/sift/sift_gt100.npy \
		--dim 128 --n-base 10000 \
		--port $(DEEPDATA_PORT) \
		--json artifacts/runs/latest_sweep.json

# Ingest benchmark
bench-ingest:
	python bench/ingest_bench.py \
		--port $(DEEPDATA_PORT) \
		--json artifacts/runs/latest_ingest.json

# Mixed workload benchmark
bench-mixed:
	python bench/mixed_workload_bench.py \
		--port $(DEEPDATA_PORT) \
		--duration 60 \
		--json artifacts/runs/latest_mixed.json

# 30-min soak test
soak-mixed:
	python soak/soak_mixed_workload.py \
		--port $(DEEPDATA_PORT) \
		--duration 1800 \
		--json artifacts/runs/latest_soak.json

# Download standard datasets
datasets:
	python datasets/scripts/download_datasets.py

lint:
	ruff check .

clean:
	rm -rf artifacts/runs/*
	rm -rf datasets/cache/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
