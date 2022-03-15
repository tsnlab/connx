## Board
- Zynq SoC zybo z7-20
## Host
- Ubuntu 20.04 LTS

## Prerequisite
- Install Vitis IDE(2021.2) and XRT Library. https://www.xilinx.com/products/design-tools/vitis/vitis-platform.html#gettingStarted (Step 1, 2)
    - You should install Vitis under /opt like /opt/Xilinx/Vitis.
- Install minicom
- Create a Hardware Design using Vivado Tool (XSA file)
    - Copy XSA file to current directory, ports/zynq/.
    - XSA file name should be design_1_wrapper.xsa.
- micro SD card
    - Rename files under the examples directory. ex) 0_1.data -> 0_1.dat, 0.text -> 0.txt, model.connx -> model.cnx
    - Copy files to the root of your sdcard.
    - Rename test_data_set_X/input_0.data to input_0.dat.
    - Copy input_0.dat file to the root of the sdcard.
    - Insert the sdcard into the zynq board.

## minicom configure
- Connect board and host PC
- Install minicom `sudo apt install minicom`
- `dmesg | grep tty`<br>output: [348931.221965] usb 3-4: FTDI USB Serial Device converter now attached to **ttyUSB1**
- Configure minicom `sudo minicom -s`
    - Serial port setup : /dev/ttyUSB1
    - Screen and keyboard : Line Wrap->Yes, Add carriage return->Yes
    - Save setup as dfl
    - Exit
- `minicom`

## Build
- `make build`
## RUN
- `make run`
