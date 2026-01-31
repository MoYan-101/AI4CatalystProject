#!/usr/bin/env bash
set -e
python train.py
python inference.py
python visualization.py
