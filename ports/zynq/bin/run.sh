#!/bin/bash

set -eo pipefail

VITIS_HOME="${XILINX_VITIS:-/opt/Xilinx/Vitis/2021.2}"
# shellcheck source=/dev/null
source "${VITIS_HOME}/settings64.sh"

xsct << EOF
setws
connect
source design_1_wrapper/hw/ps7_init.tcl
targets -set -nocase -filter {name =~ "Arm*#0"}
stop
ps7_init
rst -processor
dow test/Debug/test.elf
source design_1_wrapper/hw/ps7_init.tcl
targets -set -nocase -filter {name =~ "Arm*#0"}
stop
ps7_init
rst -processor
dow test/Debug/test.elf
con
disconnect
EOF
