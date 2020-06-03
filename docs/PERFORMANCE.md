# Test machines
 * Case 1: Intel Core i7
   * CPU: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz x 8
   * Memory: 32GB
   * O/S: Ubuntu 20.04 64-bit
 * Case 2: Raspberry Pi 3 Model B Rev 1.2
   * CPU: ARMv7 Processor rev 4 x 4
   * Memory: 1GB
   * O/S: Raspberry Pi OS 32-bit

# Result
Compile: RELEASE=1 option is used
Test scenario
 1. 12 samples are collected
 2. Max/Min 2 samples are removed
 3. Get average/min/max and std dev

## MNIST
d7d6ce82 version is used

### Case 1
Average:    923,493 
Minimum:    918,786
Maximum:    938,195
Std dev:      5,351 
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

### Case 2
Average: 27,562,269 
Minimum: 26,353,508			
Maximum: 28,103,074			
Std dev:    741,266 
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
