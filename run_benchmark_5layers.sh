#!/bin/bash
# 5-layer benchmark, one inference per VTune run.
cd "$(dirname "$0")"
exec bash run_vtune_sweep.sh wide5
