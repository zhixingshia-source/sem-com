#!/usr/bin/env bash
set -e
python -u -m src.pipelines.semantic_comm_pipeline \
  --exp configs/experiment.yaml
