# Prerequirements
## esp-idf must be installed
Ref: https://github.com/espressif/esp-idf

## Activate esp-idf environment
```sh
source $(ESP\_IDF\_HOME)/export.sh
```

# build
Below command will build and flash the image to ESP32

```sh
make
```

# monitor
```sh
make run
```

To exit the monitor, Ctrl + ]
If you have any problem with I/O permission, please check you have enough permission to access /dev/ttyUSB0
