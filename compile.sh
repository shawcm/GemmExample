#!/bin/bash

# the unified memory hints for A, B, C
um_A="RM"
um_B="RM"
um_C="UM"

# problem size
NI="20480"

[ ! -e "bin" ] && mkdir "bin"

# compile for 1 to 4 GPUs
for n in {1..4}; do
    GEMM_SRC=$(mktemp -u 'gemm_tmp_XXXXXXXXX.cu')
    cp gemm_multi_gpu.cu "${GEMM_SRC}"

    sed -i -e "s/@n@/$n/" $GEMM_SRC
    sed -i -e "s/@um_A@/$um_A/" $GEMM_SRC
    sed -i -e "s/@um_B@/$um_B/" $GEMM_SRC
    sed -i -e "s/@um_C@/$um_C/" $GEMM_SRC

    sed -i -e "s/@NI@/$NI/" $GEMM_SRC
    sed -i -e "s/@NJ@/$NI/" $GEMM_SRC
    sed -i -e "s/@NK@/$NI/" $GEMM_SRC

    nvcc -O3 $GEMM_SRC -o "bin/gemm-${n}-${NI}-${um_A}-${um_B}-${um_C}"
    echo "generated bin/gemm-${n}-${NI}-${um_A}-${um_B}-${um_C}"

    rm $GEMM_SRC
done
