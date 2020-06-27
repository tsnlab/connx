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
  * Case i7-connx-d7d6ce82 MNIST
    * H/W: Intel Core i7
    * S/W: connx-d7d6ce82
	* Date: 2020-06-03
  * Case pi3-connx-d7d6ce82 MNIST
    * H/W: Raspberry Pi 3 Model B Rev 1.2
    * S/W: connx-d7d6ce82
	* Date: 2020-06-03
  * Case i7-tf1 MNIST
    * H/W: Intel Core i7
    * S/W: TensorFlow 1.14
	* Date: 2020-06-03
  * Case pi3-tf1 MNIST
    * H/W: Raspberry Pi 3 Model B Rev 1.2
    * S/W: TensorFlow 1.13
	* Date: 2020-06-03
  * Case i7-connx-32062fa3 MNIST
    * H/W: Intel Core i7
    * S/W: connx-32062fa3
	* Date: 2020-06-27
  * Case i7-connx-32062fa3 Mobilenet2
    * H/W: Intel Core i7
    * S/W: connx-32062fa3
	* Date: 2020-06-27
  * Case i7-onnxruntime MNIST
    * H/W: Intel Core i7
    * S/W: ONNX Runtime 1.3.0
	* Date: 2020-06-27
  * Case i7-onnxruntime Mobilenet2
    * H/W: Intel Core i7
    * S/W: ONNX Runtime 1.3.0
	* Date: 2020-06-27

# Test scenario
Compile: RELEASE=1 option is used

## MNIST Test process (2020.06.03)
  1. Run MNIST 1,000 times
  2. Collect 12 samples
  3. Drop max/min 2 samples
  4. Get average/std dev/min/max

## MNIST Test process (2020.06.27)
  1. Run MNIST 1,000 times
  2. Collect 15 samples
  3. Get average/std dev/min/max

## Mobilenet v2 Test process (2020.06.27)
  1. Run Mobilenet2 10 times
  2. Collect 15 samples
  3. Get average/std dev/min/max

# Test
## Case i7-connx-d7d6ce82 MNIST (2020.06.03)
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

## Case i7-tf1 MNIST (2020.06.03)
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

## Case pi3-connx-d7d6ce82 MNIST (2020.06.03)
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

## Case pi3-tf1 MNIST (2020.06.03)
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

## Case i7-connx-32062fa3 MNIST (2020.06.27)
Average: 728,537
Std dev:   2,240
Minimum: 724,989
Maximum: 733,501
Samples
  1.  728,928
  2.  726,118
  3.  728,791
  4.  728,256
  5.  729,408
  6.  728,953
  7.  733,501
  8.  730,946
  9.  727,810
  10. 727,639
  11. 727,965
  12. 724,989
  13. 726,959
  14. 726,087
  15. 731,700

## Case i7-connx-32062fa3 Mobilenet2 (2020.06.27)
Average: 11,376,103
Std dev:      6,332
Minimum: 11,363,036
Maximum: 11,389,392
Samples
  1.  11,373,458
  2.  11,374,640
  3.  11,374,753
  4.  11,370,673
  5.  11,377,014
  6.  11,385,356
  7.  11,389,392
  8.  11,363,036
  9.  11,372,624
  10. 11,377,129
  11. 11,372,077
  12. 11,379,496
  13. 11,382,486
  14. 11,375,607
  15. 11,373,806

## Case i7-onnxruntime MNIST (2020.06.27)
Average: 20,598
Std dev:    959
Minimum: 19,452
Maximum: 22,674
Samples
  1.  21,043
  2.  20,604
  3.  20,232
  4.  19,452
  5.  20,978
  6.  22,470
  7.  20,200
  8.  22,674
  9.  20,041
  10. 19,928
  11. 21,380
  12. 19,905
  13. 19,658
  14. 20,033
  15. 20,365

## Case i7-onnxruntime Mobilenet2 (2020.06.27)
Average: 25,149
Std dev:  4,308
Minimum: 22,711
Maximum: 34,853
Samples
  1.  23,120
  2.  23,059
  3.  24,426
  4.  22,862
  5.  23,100
  6.  22,876
  7.  23,214
  8.  30,986
  9.  23,091
  10. 22,711
  11. 34,071
  12. 22,822
  13. 23,034
  14. 23,015
  15. 34,853
