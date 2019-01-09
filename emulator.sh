#!/bin/bash
make
mv bench_pq testdata/
aoc -march=emulator pred_quant.cl -o testdata/pred_quant.aocx
cd testdata
echo "Start Emulation..."
CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bench_pq
