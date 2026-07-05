#!/bin/bash

# RK3588 农业病虫害识别系统 — 一键启动脚本(不建议使用,想要使用要更改环境启动)
# bash命令：   ./start.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 设置显示环境
export QT_QPA_PLATFORM=xcb
export DISPLAY=${DISPLAY:-:0}
export PYTHONUNBUFFERED=1

# 启动
conda run -n pest-env python3 main.py
