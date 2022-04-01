#!/bin/bash

set -eo pipefail

VITIS_HOME="${XILINX_VITIS:-/opt/Xilinx/Vitis/2021.2}"
# shellcheck source=/dev/null
source "${VITIS_HOME}"/settings64.sh

PATH=$(dirname "$(readlink -e "$0")")
cd "$PATH" || true
cd ..
BASEDIR=$(pwd)

cd ../../
INCLUDEDIR=$(pwd)

cd "$BASEDIR"

# create platform
xsct << EOF
setws
platform create -name {design_1_wrapper} -hw {design_1_wrapper.xsa} -out .;platform write
domain create -name {freertos10_xilinx_ps7_cortexa9_0} -display-name {freertos10_xilinx_ps7_cortexa9_0} -os {freertos10_xilinx} -proc {ps7_cortexa9_0} -runtime {cpp} -arch {32-bit} -support-app {freertos_hello_world}
platform write
platform active {design_1_wrapper}
domain active {zynq_fsbl}
domain active {freertos10_xilinx_ps7_cortexa9_0}
bsp setlib -name xilffs
platform generate
app create -name test -platform design_1_wrapper -domain {freertos10_xilinx_ps7_cortexa9_0} -sysproj {test_system} -template {Empty Application(C)}
EOF

cp -r gen/. test/src

# increase stack and heap memory size
sed -i 's/0x2000/0x1E8480/g' test/src/lscript.ld

xsct << EOF
setws
app config -name test include-path "$BASEDIR"/include
app config -name test include-path "$INCLUDEDIR"/include
app config -name test libraries m
app build -name test
EOF
