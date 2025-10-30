#!/bin/bash
set -a
source ./.env
set +a

cd "$(dirname "$0")"
~/miniconda3/bin/python ./main.py
