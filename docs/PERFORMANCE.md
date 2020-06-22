# H/W Configurations
  * Intel Core i7
    * CPU: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz x 8
    * Memory: 32GB
    * O/S: Ubuntu 20.04 64-bit
  * Raspberry Pi 3 Model B Rev 1.2
    * CPU: ARMv7 Processor rev 4 x 4
    * Memory: 1GB
    * O/S: Raspberry Pi OS 32-bit

# Test cases
  * Case i7-connx
    * H/W: Intel Core i7
    * S/W: connx-d7d6ce82
  * Case pi3-connx
    * H/W: Raspberry Pi 3 Model B Rev 1.2
    * S/W: connx-d7d6ce82
  * Case i7-tf1
    * H/W: Intel Core i7
    * S/W: TensorFlow 1.14
  * Case pi3-tf1
    * H/W: Raspberry Pi 3 Model B Rev 1.2
    * S/W: TensorFlow 1.13

# Test scenario
Compile: RELEASE=1 option is used
Test process
  1. Collect 12 samples
  2. Drop max/min 2 samples
  3. Get average/std dev/min/max

# MNIST
## Case i7-connx
Average:    923,493 
Std dev:      5,351 
Minimum:    918,786
Maximum:    938,195
Samples
  1.  919,279
  2.  946,669
  3.  918,786
  4.  938,195
  5.  919,168
  6.  925,071
  7.  922,393
  8.  922,614
  9.  924,347
  10. 921,027
  11. 924,051
  12. 918,297

## Case i7-tf1
Average: 15,550,941 
Std dev:     50,509 
Minimum: 15,493,306 
Maximum: 15,649,594 
Samples
  1.  15,612,313
  2.  15,586,623
  3.  15,649,594
  4.  15,495,432
  5.  15,568,938
  6.  15,518,133
  7.  15,554,518
  8.  15,493,306
  9.  15,529,050
  10. 15,664,591
  11. 15,443,263
  12. 15,501,506

## Case pi3-connx
Average: 27,562,269 
Std dev:    741,266 
Minimum: 26,353,508			
Maximum: 28,103,074			
Samples
  1.  28,101,448
  2.  26,353,508
  3.  28,025,985
  4.  28,043,865
  5.  28,021,502
  6.  28,111,397
  7.  27,995,843
  8.  26,357,453
  9.  28,026,161
  10. 28,103,074
  11. 26,593,851
  12. 26,332,273

## Case pi3-tf1
Average: 354,055,684 
Std dev:     993,996 
Minimum: 351,825,697 
Maximum: 355,282,075 
Samples
  1.  355,224,623
  2.  346,797,951
  3.  355,644,413
  4.  354,967,219
  5.  354,022,266
  6.  353,094,584
  7.  355,282,075
  8.  354,075,028
  9.  353,802,718
  10. 351,825,697
  11. 354,451,829
  12. 353,810,797
