# Board
- Zynq SoC zybo z7-20
# Host
- Ubuntu 20.04 LTS

# Prerequisite
- Install Vitis IDE(2021.2) and XRT Library. https://www.xilinx.com/products/design-tools/vitis/vitis-platform.html#gettingStarted (Step 1, 2)
- Install minicom
- Create a Hardware Design using Vivado Tool (XSA file)
- micro SD card
    - Rename files under the examples directory. ex) 0_1.data -> 0_1.dat, 0.text -> 0.txt, model.connx -> model.cnx
    - Copy files to the root of your sdcard.
    - Rename test_data_set_X/input_0.data to input_0.dat.
    - Copy input_0.dat file to the root of the sdcard.
    - Insert the sdcard into the zynq board.

# minicom configure
- Connect board and host PC
- Install minicom `sudo apt install minicom`
- `dmesg | grep tty`<br>output: [348931.221965] usb 3-4: FTDI USB Serial Device converter now attached to **ttyUSB1**
- Configure minicom `sudo minicom -s`
    - Serial port setup : /dev/ttyUSB1
    - Screen and keyboard : Line Wrap->Yes, Add carriage return->Yes
    - Save setup as dfl
    - Exit
- `minicom`

# RUN
1. `make gen`
2. `source /opt/Xilinx/Vitis/2021.2/settings64.sh`
3. `xsct` - activate xsct shell
- ### create platform and application
    ```shell
    setws
    platform create -name {design_1_wrapper} -hw {[.xsa file path]} -out .;platform write
    domain create -name {freertos10_xilinx_ps7_cortexa9_0} -display-name {freertos10_xilinx_ps7_cortexa9_0} -os {freertos10_xilinx} -proc {ps7_cortexa9_0} -runtime {cpp} -arch {32-bit} -support-app {freertos_hello_world}
    platform write
    platform active {design_1_wrapper}
    domain active {zynq_fsbl}
    domain active {freertos10_xilinx_ps7_cortexa9_0}
    bsp setlib -name xilffs
    platform generate


    app create -name test -platform design_1_wrapper -domain {freertos10_xilinx_ps7_cortexa9_0} -sysproj {test_system} -template {Empty Application(C)}

    cp -r gen/. test/src

    app config -name test include-path [connx include ]/include
    app config -name test include-path include
    app config -name test libraries m
    ```
- ### resize stack and heap size
    - open **test/src/lscript.ld** file
    - change _STACK_SIZE : 0x2000; to _STACK_SIZE : 0x1E8480;
    - change _HEAP_SIZE : 0x2000; to _HEAP_SIZE : 0x1E8480;
- ### build and run
    ```shell
    app build -name test

    connect
    source design_1_wrapper/hw/ps7_init.tcl
    targets -set -nocase -filter {name =~ "Arm*#0"}
    stop
    ps7_init
    rst -processor
    dow test/Debug/test.elf

    con

    disconnect

    ```