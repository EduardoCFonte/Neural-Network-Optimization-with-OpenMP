#!/bin/bash
# 8-layer benchmark, one inference per VTune run.
cd "$(dirname "$0")"
exec bash run_vtune_sweep.sh standard
