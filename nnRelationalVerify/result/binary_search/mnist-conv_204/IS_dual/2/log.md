## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.78041487804
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.8020515, 2.8020515)
1: (-9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142)
2: (-9.9503212, -6.9502754, -9.9503212, -6.9502754, -3.0000458, 3.0000458)
3: (-10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.5673351, 2.5673351)
4: (-5.5582318, -3.5118723, -5.5582318, -3.5118723, -2.0463595, 2.0463595)
5: (-8.8875761, -6.1918221, -8.8875761, -6.1918221, -2.4845166, 2.4845166)
6: (-12.9723425, -9.7499943, -12.9723425, -9.7499943, -3.1591969, 3.1591969)
7: (0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.4368451, 2.4368451)
8: (-3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.7339888, 2.7339888)
9: (0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423)

## BASE Result
execution time: IAR + LP analysis = 15.09 + 32.28 = 47.38 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.62 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.2286510467529297
rel_dist={7: [-1.3704629636566878, 1.370462874585793]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.0185980796813965
rel_dist={7: [-1.0634246550232, 1.0634268200891506]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.8785624504089355
rel_dist={7: [-0.7927072900922123, 0.7927031607168487]}

## Binary Search Result
Binary search time: 148.04 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 3404.59 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 6181

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3704541, upper bound: 1.3603419
time: 3.83 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3704541, upper bound: 1.3704539
time: 3.91 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.91 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 7.91
Output dim: 7, lower bound: -1.3704541, upper bound: 1.3603419
IS_B2, status: Status.UNKNOWN, split count: 1, time: 7.91
Output dim: 7, lower bound: -1.3704541, upper bound: 1.3704539

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -8.1611385, -5.3684554, -8.1321526, -5.3759112, -2.3172426, 2.3022795
1: -9.2237587, -6.2050738, -9.2126160, -6.2120185, -3.0117402, 3.0075421
2: -9.9485035, -6.9602008, -9.9340887, -6.9920025, -2.6079078, 2.6226640
3: -10.8303308, -8.2697258, -10.8188086, -8.2860575, -2.1637526, 2.1705296
4: -5.5563612, -3.5183458, -5.5464420, -3.5399156, -1.8507237, 1.8611650
5: -8.8813400, -6.1926470, -8.8613005, -6.2009053, -1.8919773, 1.8811326
6: -12.9708843, -9.7512064, -12.9650011, -9.7570105, -2.4267941, 2.4262562
7: 0.4160919, 2.8416376, 0.4514637, 2.8336329, -2.2085896, 2.1825786
8: -3.7183290, -0.9940157, -3.7049961, -1.0193210, -2.5891223, 2.5953140
9: 0.1569617, 2.2647943, 0.1637008, 2.2586241, -2.1016624, 2.1010935

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3147528, upper bound: 1.2914280
time: 3.09 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3479616, upper bound: 1.3378970
time: 3.68 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -8.1700583, -5.3680067, -8.1700544, -5.3680086, -2.3390398, 2.3221960
1: -9.2260056, -6.2035913, -9.2260036, -6.2035923, -3.0224133, 3.0224123
2: -9.9503212, -6.9502754, -9.9503212, -6.9502831, -2.6256390, 2.6494493
3: -10.8334827, -8.2661476, -10.8334780, -8.2661514, -2.1944652, 2.1891730
4: -5.5582318, -3.5118723, -5.5582314, -3.5118756, -1.8637280, 1.8800120
5: -8.8875761, -6.1918221, -8.8875713, -6.1918206, -1.9080458, 1.8949378
6: -12.9723425, -9.7499943, -12.9723415, -9.7499933, -2.4407964, 2.4364743
7: 0.4052801, 2.8421252, 0.4052863, 2.8421242, -2.2286515, 2.2031617
8: -3.7202172, -0.9862285, -3.7202139, -0.9862318, -2.6133041, 2.6218419
9: 0.1555150, 2.2660573, 0.1555148, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3147528, upper bound: 1.3007508
time: 3.21 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3479616, upper bound: 1.3479611
time: 3.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.47 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 21.47
Output dim: 7, lower bound: -1.3147528, upper bound: 1.2914280
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 21.47
Output dim: 7, lower bound: -1.3479616, upper bound: 1.3378970
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 21.47
Output dim: 7, lower bound: -1.3147528, upper bound: 1.3007508
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 21.47
Output dim: 7, lower bound: -1.3479616, upper bound: 1.3479611

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -8.1060734, -5.3335080, -8.1169834, -5.3761945, -2.2532911, 2.3109403
1: -9.1123877, -6.2008524, -9.1863155, -6.2166510, -2.8957367, 2.9854631
2: -9.9065113, -6.9990969, -9.9244270, -7.0008945, -2.5693140, 2.5858622
3: -10.8716583, -8.3538561, -10.8141279, -8.3104239, -2.1359539, 2.0542681
4: -5.5100656, -3.5153117, -5.5358391, -3.5422475, -1.8158922, 1.8725405
5: -8.8761806, -6.1876626, -8.8579979, -6.2011814, -1.8842750, 1.8778641
6: -12.9068270, -9.7121391, -12.9497137, -9.7572651, -2.3503342, 2.4264472
7: 0.3516245, 2.7343678, 0.4549294, 2.8082418, -2.1871061, 2.0328341
8: -3.6516080, -0.9422121, -3.6880112, -1.0243864, -2.4471059, 2.5462761
9: 0.2631596, 2.2642236, 0.1884406, 2.2560494, -1.9928898, 2.0757830

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2794832, upper bound: 1.2596606
time: 3.37 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2912191, upper bound: 1.2646407
time: 3.11 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -8.1494045, -5.3684616, -8.1321526, -5.3759112, -2.2776122, 2.3022370
1: -9.2154179, -6.2061758, -9.2126160, -6.2120185, -2.9521127, 3.0064402
2: -9.9465389, -6.9635944, -9.9340887, -6.9920025, -2.6059928, 2.6128798
3: -10.8288841, -8.2815189, -10.8188086, -8.2860575, -2.1624975, 2.0632586
4: -5.5510807, -3.5192468, -5.5464420, -3.5399156, -1.8656869, 1.8598905
5: -8.8805809, -6.1929984, -8.8613005, -6.2009053, -1.8902893, 1.8830180
6: -12.9631948, -9.7513294, -12.9650011, -9.7570105, -2.4047503, 2.4226923
7: 0.4175200, 2.8213553, 0.4514637, 2.8336329, -2.2076135, 2.0594046
8: -3.6994205, -0.9950175, -3.7049961, -1.0193210, -2.4832840, 2.5945144
9: 0.1645412, 2.2641039, 0.1637008, 2.2586241, -2.0940828, 2.1004031

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3051362
time: 3.54 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3378985
time: 3.47 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -8.1148891, -5.3330436, -8.1548538, -5.3682842, -2.2750926, 2.3307896
1: -9.1146488, -6.1993046, -9.1998711, -6.2081742, -2.9064746, 3.0005665
2: -9.9084711, -6.9891744, -9.9408169, -6.9591765, -2.5872016, 2.6127486
3: -10.8748178, -8.3504839, -10.8287354, -8.2905121, -2.1667795, 2.0725679
4: -5.5119309, -3.5088503, -5.5476270, -3.5142245, -1.8288298, 1.8913255
5: -8.8824062, -6.1868315, -8.8842621, -6.1920948, -1.9003181, 1.8916473
6: -12.9083214, -9.7108288, -12.9571075, -9.7502499, -2.3640041, 2.4369009
7: 0.3405662, 2.7348566, 0.4087830, 2.8167381, -2.2074528, 2.0533333
8: -3.6534634, -0.9343410, -3.7030644, -0.9913211, -2.4712143, 2.5729401
9: 0.2617211, 2.2655323, 0.1802540, 2.2635264, -2.0018053, 2.0852783

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2794832, upper bound: 1.2689856
time: 3.61 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2912191, upper bound: 1.2739639
time: 3.02 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -8.1583176, -5.3680158, -8.1700544, -5.3680086, -2.2994385, 2.3221512
1: -9.2176466, -6.2046900, -9.2260036, -6.2035923, -2.9909048, 3.0213137
2: -9.9483614, -6.9536695, -9.9503212, -6.9502831, -2.6237283, 2.6397161
3: -10.8320341, -8.2779436, -10.8334780, -8.2661514, -2.1932092, 2.0817318
4: -5.5529528, -3.5127759, -5.5582314, -3.5118756, -1.8790331, 1.8787279
5: -8.8868170, -6.1921725, -8.8875713, -6.1918206, -1.9063506, 1.8968737
6: -12.9646549, -9.7501202, -12.9723415, -9.7499933, -2.4202437, 2.4329100
7: 0.4067101, 2.8217998, 0.4052863, 2.8421242, -2.2276411, 2.0806131
8: -3.7013478, -0.9872293, -3.7202139, -0.9862318, -2.5074792, 2.6210442
9: 0.1630857, 2.2653706, 0.1555148, 2.2660573, -2.1029716, 2.1098557

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3147533
time: 3.53 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3479621
time: 4.24 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.36 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 7, lower bound: -1.2794832, upper bound: 1.2596606
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 7, lower bound: -1.2912191, upper bound: 1.2646407
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3051362
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3378985
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 7, lower bound: -1.2794832, upper bound: 1.2689856
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 7, lower bound: -1.2912191, upper bound: 1.2739639
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3147533
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3479621

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -8.1017628, -5.3419604, -8.1315422, -5.3772955, -2.2481556, 2.3135800
1: -9.1121407, -6.2175026, -9.2125854, -6.2145157, -2.8976250, 2.9901094
2: -9.8964005, -7.0154099, -9.9325294, -6.9942551, -2.5656300, 2.5697556
3: -10.8685255, -8.3544941, -10.8183479, -8.2861433, -2.1618814, 2.0568211
4: -5.5066533, -3.5205691, -5.5460248, -3.5406802, -1.8070145, 1.8745317
5: -8.8716488, -6.1878257, -8.8606596, -6.2009301, -1.8796711, 1.8758831
6: -12.9054070, -9.7149897, -12.9648142, -9.7574940, -2.3453069, 2.4390960
7: 0.3899102, 2.7289963, 0.4576831, 2.8328843, -2.1790247, 2.0239596
8: -3.6411233, -0.9481559, -3.7036271, -1.0205255, -2.4404593, 2.5648849
9: 0.2725549, 2.2609763, 0.1651098, 2.2581301, -1.9855752, 2.0958667

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2596612
time: 3.43 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2596612
time: 3.68 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -8.1075745, -5.3462200, -8.1318598, -5.3776622, -2.2702894, 2.3094251
1: -9.1141558, -6.2001801, -9.2125959, -6.2128034, -2.9013524, 3.0124159
2: -9.9195118, -7.0170517, -9.9337139, -6.9945531, -2.6143303, 2.5753312
3: -10.8691959, -8.3513546, -10.8185129, -8.2860947, -2.1624331, 2.0558493
4: -5.5066934, -3.5180712, -5.5457835, -3.5402966, -1.8107471, 1.8817782
5: -8.8755360, -6.1881118, -8.8609257, -6.2009373, -1.8890910, 1.8752873
6: -12.9122295, -9.7133923, -12.9649363, -9.7572584, -2.3556309, 2.4396589
7: 0.3887777, 2.7398057, 0.4559155, 2.8333387, -2.1752753, 2.0469437
8: -3.6378427, -0.9476552, -3.7046099, -1.0213299, -2.4726133, 2.5708871
9: 0.2710613, 2.2633841, 0.1651475, 2.2585316, -1.9874703, 2.0982366

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739632, upper bound: 1.2646400
time: 3.30 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739632, upper bound: 1.2646417
time: 3.24 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.1494045, -5.3684616, -8.0774212, -5.3410282, -2.3308911, 2.2389100
1: -9.2154179, -6.2061758, -9.1012020, -6.2079577, -3.0074601, 2.8950262
2: -9.9465389, -6.9635944, -9.8916731, -7.0308886, -2.5797219, 2.5856190
3: -10.8288841, -8.2815189, -10.8603325, -8.3695946, -2.0506196, 2.1640499
4: -5.5510807, -3.5192468, -5.5002260, -3.5368199, -1.8601022, 1.8280525
5: -8.8805809, -6.1929984, -8.8562021, -6.1959600, -1.8908019, 1.8722818
6: -12.9631948, -9.7513294, -12.9008265, -9.7182255, -2.4376993, 2.3474910
7: 0.4175200, 2.8213553, 0.3875179, 2.7263870, -2.0605555, 2.1901913
8: -3.6994205, -0.9950175, -3.6383677, -0.9676361, -2.5681753, 2.4558175
9: 0.1645412, 2.2641039, 0.2698810, 2.2579274, -2.0933862, 1.9942229

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2701551
time: 3.37 seconds

## Relational analysis of IS_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739634, upper bound: 1.2815942
time: 3.41 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1494045, -5.3684616, -8.1204453, -5.3759203, -2.2775755, 2.2626834
1: -9.2154179, -6.2061758, -9.2042723, -6.2131233, -2.9443517, 2.9415569
2: -9.9465389, -6.9635944, -9.9321079, -6.9953933, -2.5962682, 2.6109357
3: -10.8288841, -8.2815189, -10.8173695, -8.2978544, -2.0558519, 2.0619814
4: -5.5510807, -3.5192468, -5.5411634, -3.5408163, -1.8644876, 1.8753891
5: -8.8805809, -6.1929984, -8.8605442, -6.2012568, -1.8922725, 1.8813956
6: -12.9631948, -9.7513294, -12.9573345, -9.7571383, -2.4035301, 2.4043722
7: 0.4175200, 2.8213553, 0.4528751, 2.8134508, -2.0849762, 2.0584548
8: -3.6994205, -0.9950175, -3.6860209, -1.0203171, -2.4824963, 2.4887283
9: 0.1645412, 2.2641039, 0.1713096, 2.2579260, -2.0933847, 2.0927944

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3147528, upper bound: 1.2914279
time: 3.17 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3479616, upper bound: 1.3378970
time: 3.71 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -8.1105738, -5.3414984, -8.1694431, -5.3693914, -2.2699499, 2.3334537
1: -9.1143990, -6.2159834, -9.2259703, -6.2060976, -2.9083014, 3.0070767
2: -9.8983250, -7.0054874, -9.9487457, -6.9525347, -2.5835533, 2.5965047
3: -10.8716803, -8.3511286, -10.8330221, -8.2662373, -2.1927514, 2.0752146
4: -5.5085230, -3.5141048, -5.5578146, -3.5126381, -1.8199720, 1.8934898
5: -8.8778753, -6.1869965, -8.8869333, -6.1918411, -1.8957124, 1.8896990
6: -12.9068975, -9.7136812, -12.9721498, -9.7504826, -2.3589830, 2.4496317
7: 0.3788533, 2.7294817, 0.4115062, 2.8413720, -2.1993890, 2.0445189
8: -3.6429420, -0.9402838, -3.7188363, -0.9874349, -2.4645662, 2.5915270
9: 0.2711178, 2.2622745, 0.1569275, 2.2655573, -1.9944395, 2.1053472

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2689856
time: 3.44 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2689851
time: 3.68 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.1164227, -5.3457594, -8.1697617, -5.3697577, -2.2920618, 2.3293021
1: -9.1164370, -6.1986699, -9.2259817, -6.2043800, -2.9120569, 3.0273118
2: -9.9214334, -7.0071316, -9.9499416, -6.9528341, -2.6322618, 2.6020875
3: -10.8723545, -8.3479443, -10.8331852, -8.2661877, -2.1933026, 2.0742478
4: -5.5085449, -3.5116096, -5.5575738, -3.5122552, -1.8237052, 1.9007416
5: -8.8817644, -6.1872816, -8.8871975, -6.1918507, -1.9051175, 1.8891034
6: -12.9137135, -9.7120819, -12.9722786, -9.7502441, -2.3693256, 2.4502089
7: 0.3777070, 2.7402911, 0.4097381, 2.8418314, -2.1956592, 2.0675025
8: -3.6396461, -0.9397798, -3.7198272, -0.9882383, -2.4967718, 2.5975337
9: 0.2696328, 2.2646761, 0.1569660, 2.2659659, -1.9963331, 2.1077101

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739632, upper bound: 1.2739634
time: 3.31 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739632, upper bound: 1.2739633
time: 3.71 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.1583176, -5.3680158, -8.1148834, -5.3330421, -2.3527327, 2.2588320
1: -9.2176466, -6.2046900, -9.1146488, -6.1993079, -3.0183387, 2.9099588
2: -9.9483614, -6.9536695, -9.9084682, -6.9891806, -2.5973926, 2.6130505
3: -10.8320341, -8.2779436, -10.8748140, -8.3504868, -2.0803528, 2.1828258
4: -5.5529528, -3.5127759, -5.5119295, -3.5088513, -1.8734922, 1.8467174
5: -8.8868170, -6.1921725, -8.8824034, -6.1868315, -1.9068995, 1.8859797
6: -12.9646549, -9.7501202, -12.9083214, -9.7108307, -2.4529715, 2.3575392
7: 0.4067101, 2.8217998, 0.3405704, 2.7348576, -2.0805230, 2.2119436
8: -3.7013478, -0.9872293, -3.6534615, -0.9343452, -2.5927773, 2.4824226
9: 0.1630857, 2.2653706, 0.2617226, 2.2655325, -2.1024468, 2.0036480

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_B2_A2_B1_B1

### Relational analysis result of IS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2794834
time: 3.45 seconds

## Relational analysis of IS_B2_A2_B1_B2

### Relational analysis result of IS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739634, upper bound: 1.2912189
time: 3.36 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1583176, -5.3680158, -8.1583109, -5.3680158, -2.2993989, 2.2825665
1: -9.2176466, -6.2046900, -9.2176418, -6.2046900, -2.9831243, 2.9581518
2: -9.9483614, -6.9536695, -9.9483604, -6.9536762, -2.6139750, 2.6377890
3: -10.8320341, -8.2779436, -10.8320303, -8.2779474, -2.0857344, 2.0804417
4: -5.5529528, -3.5127759, -5.5529518, -3.5127769, -1.8778234, 1.8941002
5: -8.8868170, -6.1921725, -8.8868132, -6.1921716, -1.9083433, 1.8952355
6: -12.9646549, -9.7501202, -12.9646530, -9.7501202, -2.4190240, 2.4147105
7: 0.4067101, 2.8217998, 0.4067149, 2.8217978, -2.1050892, 2.0796089
8: -3.7013478, -0.9872293, -3.7013464, -0.9872322, -2.5066986, 2.5152221
9: 0.1630857, 2.2653706, 0.1630850, 2.2653689, -2.1022832, 2.1022856

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3147528, upper bound: 1.3007510
time: 3.24 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3479616, upper bound: 1.3479611
time: 3.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.46 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2596612
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2596612
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2739632, upper bound: 1.2646400
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2739632, upper bound: 1.2646417
IS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2701551
IS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2739634, upper bound: 1.2815942
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.3147528, upper bound: 1.2914279
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.3479616, upper bound: 1.3378970
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2689856
IS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2689851
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2739632, upper bound: 1.2739634
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2739632, upper bound: 1.2739633
IS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2794834
IS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2739634, upper bound: 1.2912189
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.3147528, upper bound: 1.3007510
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.3479616, upper bound: 1.3479611

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -8.1017628, -5.3419604, -8.0774212, -5.3410282, -2.2755914, 2.2506618
1: -9.1121407, -6.2175026, -9.1012020, -6.2079577, -2.8871441, 2.8714862
2: -9.8964005, -7.0154099, -9.8916731, -7.0308886, -2.5427275, 2.5390720
3: -10.8685255, -8.3544941, -10.8603325, -8.3695946, -2.0500865, 2.0580389
4: -5.5066533, -3.5205691, -5.5002260, -3.5368199, -1.8278785, 1.8439951
5: -8.8716488, -6.1878257, -8.8562021, -6.1959600, -1.8808942, 1.8726208
6: -12.9054070, -9.7149897, -12.9008265, -9.7182255, -2.3688369, 2.3644989
7: 0.3899102, 2.7289963, 0.3875179, 2.7263870, -2.0324388, 2.0465481
8: -3.6411233, -0.9481559, -3.6383677, -0.9676361, -2.4277763, 2.4273007
9: 0.2725549, 2.2609763, 0.2698810, 2.2579274, -1.9853725, 1.9910953

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2564420
time: 3.45 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2596612
time: 3.37 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -8.1017628, -5.3419604, -8.1204453, -5.3759203, -2.2504549, 2.3025684
1: -9.1121407, -6.2175026, -9.2042723, -6.2131233, -2.8990173, 2.9844241
2: -9.8964005, -7.0154099, -9.9321079, -6.9953933, -2.5638671, 2.5690627
3: -10.8685255, -8.3544941, -10.8173695, -8.2978544, -2.1546364, 2.0560484
4: -5.5066533, -3.5205691, -5.5411634, -3.5408163, -1.8066497, 1.8649211
5: -8.8716488, -6.1878257, -8.8605442, -6.2012568, -1.8758163, 1.8752286
6: -12.9054070, -9.7149897, -12.9573345, -9.7571383, -2.3433466, 2.4286799
7: 0.3899102, 2.7289963, 0.4528751, 2.8134508, -2.1716375, 2.0301261
8: -3.6411233, -0.9481559, -3.6860209, -1.0203171, -2.4419050, 2.5599523
9: 0.2725549, 2.2609763, 0.1713096, 2.2579260, -1.9853711, 2.0896668

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2794832, upper bound: 1.2596606
time: 3.51 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2794832, upper bound: 1.2596612
time: 3.52 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.1075745, -5.3462200, -8.0774212, -5.3410282, -2.2961512, 2.2463133
1: -9.1141558, -6.2001801, -9.1012020, -6.2079577, -2.8873415, 2.9010220
2: -9.9195118, -7.0170517, -9.8916731, -7.0308886, -2.5893426, 2.5436988
3: -10.8691959, -8.3513546, -10.8603325, -8.3695946, -2.0505896, 2.0568647
4: -5.5066934, -3.5180712, -5.5002260, -3.5368199, -1.8314538, 1.8508558
5: -8.8755360, -6.1881118, -8.8562021, -6.1959600, -1.8901238, 1.8720908
6: -12.9122295, -9.7133923, -12.9008265, -9.7182255, -2.3781877, 2.3652141
7: 0.3887777, 2.7398057, 0.3875179, 2.7263870, -2.0284014, 2.0659173
8: -3.6378427, -0.9476552, -3.6383677, -0.9676361, -2.4586744, 2.4325631
9: 0.2710613, 2.2633841, 0.2698810, 2.2579274, -1.9868661, 1.9935031

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_B1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2689850, upper bound: 1.2564420
time: 3.61 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2623211
time: 3.60 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1075745, -5.3462200, -8.1204453, -5.3759203, -2.2710147, 2.2982202
1: -9.1141558, -6.2001801, -9.2042723, -6.2131233, -2.9010324, 3.0040922
2: -9.9195118, -7.0170517, -9.9321079, -6.9953933, -2.6104827, 2.5736895
3: -10.8691959, -8.3513546, -10.8173695, -8.2978544, -2.1551399, 2.0548742
4: -5.5066934, -3.5180712, -5.5411634, -3.5408163, -1.8102245, 1.8717813
5: -8.8755360, -6.1881118, -8.8605442, -6.2012568, -1.8850465, 1.8746984
6: -12.9122295, -9.7133923, -12.9573345, -9.7571383, -2.3526974, 2.4293952
7: 0.3887777, 2.7398057, 0.4528751, 2.8134508, -2.1675997, 2.0494957
8: -3.6378427, -0.9476552, -3.6860209, -1.0203171, -2.4728031, 2.5652146
9: 0.2710613, 2.2633841, 0.1713096, 2.2579260, -1.9868647, 2.0920744

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2912191, upper bound: 1.2646407
time: 3.35 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2912191, upper bound: 1.2646407
time: 3.34 seconds

## BFS IS instance: IS_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.1487999, -5.3698459, -8.0731163, -5.3494797, -2.3170800, 2.2333755
1: -9.2153845, -6.2086620, -9.1009560, -6.2245474, -2.9848499, 2.8922939
2: -9.9449797, -6.9658456, -9.8816700, -7.0472002, -2.5531235, 2.5747888
3: -10.8284254, -8.2816057, -10.8571997, -8.3702316, -2.0495105, 2.1614931
4: -5.5506639, -3.5200090, -5.4967670, -3.5420833, -1.8534040, 1.8162966
5: -8.8799438, -6.1930189, -8.8516674, -6.1961284, -1.8849993, 1.8643489
6: -12.9630127, -9.7518167, -12.8994246, -9.7210789, -2.4277577, 2.3413382
7: 0.4237385, 2.8205805, 0.4257979, 2.7210050, -2.0490689, 2.1442404
8: -3.6980257, -0.9962201, -3.6279364, -0.9735832, -2.5527611, 2.4458697
9: 0.1659136, 2.2636085, 0.2792642, 2.2547159, -2.0888023, 1.9843442

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2564420
time: 3.59 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2

### Relational analysis result of IS_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2701554
time: 3.59 seconds

## BFS IS instance: IS_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.1491137, -5.3702102, -8.0788498, -5.3537364, -2.3129172, 2.2555079
1: -9.2153959, -6.2069569, -9.1029472, -6.2072001, -3.0081959, 2.8959904
2: -9.9461613, -6.9661450, -9.9048271, -7.0488420, -2.5586982, 2.6233273
3: -10.8285875, -8.2815552, -10.8578720, -8.3671989, -2.0484791, 2.1620417
4: -5.5504222, -3.5196273, -5.4968624, -3.5395865, -1.8605385, 1.8200383
5: -8.8802061, -6.1930270, -8.8555508, -6.1964049, -1.8844070, 1.8737981
6: -12.9631319, -9.7515802, -12.9062557, -9.7194767, -2.4281993, 2.3516688
7: 0.4219723, 2.8210542, 0.4247084, 2.7318225, -2.0720425, 2.1404567
8: -3.6990170, -0.9970255, -3.6246996, -0.9730840, -2.5587873, 2.4779730
9: 0.1659901, 2.2640107, 0.2777462, 2.2571371, -2.0911469, 1.9862645

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739634, upper bound: 1.2646398
time: 3.38 seconds

## Relational analysis of IS_B1_A2_B1_B2_A2

### Relational analysis result of IS_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739634, upper bound: 1.2815943
time: 3.38 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.1060734, -5.3335080, -8.1204453, -5.3759203, -2.2536597, 2.3159831
1: -9.1123877, -6.2008524, -9.2042723, -6.2131233, -2.8992643, 3.0034199
2: -9.9065113, -6.9990969, -9.9321079, -6.9953933, -2.5713444, 2.5944290
3: -10.8716583, -8.3538561, -10.8173695, -8.2978544, -2.1571140, 2.0566900
4: -5.5100656, -3.5153117, -5.5411634, -3.5408163, -1.8174996, 1.8703837
5: -8.8761806, -6.1876626, -8.8605442, -6.2012568, -1.8830442, 1.8799956
6: -12.9068270, -9.7121391, -12.9573345, -9.7571383, -2.3478770, 2.4381032
7: 0.3516245, 2.7343678, 0.4528751, 2.8134508, -2.2170706, 2.0345206
8: -3.6516080, -0.9422121, -3.6860209, -1.0203171, -2.4496746, 2.5741777
9: 0.2631596, 2.2642236, 0.1713096, 2.2579260, -1.9947664, 2.0929141

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B1_A2_B2_A1_A1

### Relational analysis result of IS_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2596611
time: 3.52 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2

### Relational analysis result of IS_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739632, upper bound: 1.2646400
time: 3.92 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.1494045, -5.3684616, -8.1204453, -5.3759203, -2.2775755, 2.2626834
1: -9.2154179, -6.2061758, -9.2042723, -6.2131233, -2.9443517, 2.9415569
2: -9.9465389, -6.9635944, -9.9321079, -6.9953933, -2.5962682, 2.6109357
3: -10.8288841, -8.2815189, -10.8173695, -8.2978544, -2.0558519, 2.0619814
4: -5.5510807, -3.5192468, -5.5411634, -3.5408163, -1.8644876, 1.8753891
5: -8.8805809, -6.1929984, -8.8605442, -6.2012568, -1.8922725, 1.8813956
6: -12.9631948, -9.7513294, -12.9573345, -9.7571383, -2.4035301, 2.4043722
7: 0.4175200, 2.8213553, 0.4528751, 2.8134508, -2.0849762, 2.0584548
8: -3.6994205, -0.9950175, -3.6860209, -1.0203171, -2.4824963, 2.4887283
9: 0.1645412, 2.2641039, 0.1713096, 2.2579260, -2.0933847, 2.0927944

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3051362
time: 3.70 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3479611, upper bound: 1.3378974
time: 3.76 seconds

## BFS IS instance: IS_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -8.1105738, -5.3414984, -8.1148834, -5.3330421, -2.2974300, 2.2705474
1: -9.1143990, -6.2159834, -9.1146488, -6.1993079, -2.9150910, 2.8882446
2: -9.8983250, -7.0054874, -9.9084682, -6.9891806, -2.5605845, 2.5664783
3: -10.8716803, -8.3511286, -10.8748140, -8.3504868, -2.0799775, 2.0765324
4: -5.5085230, -3.5141048, -5.5119295, -3.5088513, -1.8411150, 1.8627810
5: -8.8778753, -6.1869965, -8.8824034, -6.1868315, -1.8969717, 1.8863306
6: -12.9068975, -9.7136812, -12.9083214, -9.7108307, -2.3837843, 2.3748655
7: 0.3788533, 2.7294817, 0.3405704, 2.7348576, -2.0527487, 2.0682797
8: -3.6429420, -0.9402838, -3.6534615, -0.9343452, -2.4523077, 2.4540312
9: 0.2711178, 2.2622745, 0.2617226, 2.2655325, -1.9944147, 2.0005519

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_B2_A1_A1_B1_B1

### Relational analysis result of IS_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2657663
time: 3.49 seconds

## Relational analysis of IS_B2_A1_A1_B1_B2

### Relational analysis result of IS_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2689856
time: 3.46 seconds

## BFS IS instance: IS_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -8.1105738, -5.3414984, -8.1583109, -5.3680158, -2.2722459, 2.3224440
1: -9.1143990, -6.2159834, -9.2176418, -6.2046900, -2.9097090, 3.0014033
2: -9.8983250, -7.0054874, -9.9483604, -6.9536762, -2.5817895, 2.5958352
3: -10.8716803, -8.3511286, -10.8320303, -8.2779474, -2.1856403, 2.0744290
4: -5.5085230, -3.5141048, -5.5529518, -3.5127769, -1.8195839, 1.8843088
5: -8.8778753, -6.1869965, -8.8868132, -6.1921716, -1.8918571, 1.8890281
6: -12.9068975, -9.7136812, -12.9646530, -9.7501202, -2.3570256, 2.4392302
7: 0.3788533, 2.7294817, 0.4067149, 2.8217978, -2.1920147, 2.0506363
8: -3.6429420, -0.9402838, -3.7013464, -0.9872322, -2.4660187, 2.5866270
9: 0.2711178, 2.2622745, 0.1630850, 2.2653689, -1.9942511, 2.0991895

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2794832, upper bound: 1.2689856
time: 3.45 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2794832, upper bound: 1.2689856
time: 3.63 seconds

## BFS IS instance: IS_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.1164227, -5.3457594, -8.1148834, -5.3330421, -2.3179669, 2.2662001
1: -9.1164370, -6.1986699, -9.1146488, -6.1993079, -2.9171290, 2.9159789
2: -9.9214334, -7.0071316, -9.9084682, -6.9891806, -2.6072092, 2.5711076
3: -10.8723545, -8.3479443, -10.8748140, -8.3504868, -2.0804811, 2.0753641
4: -5.5085449, -3.5116096, -5.5119295, -3.5088513, -1.8446903, 1.8696461
5: -8.8817644, -6.1872816, -8.8824034, -6.1868315, -1.9061866, 1.8858004
6: -12.9137135, -9.7120819, -12.9083214, -9.7108307, -2.3931527, 2.3755922
7: 0.3777070, 2.7402911, 0.3405704, 2.7348576, -2.0487270, 2.0876515
8: -3.6396461, -0.9397798, -3.6534615, -0.9343452, -2.4832582, 2.4592869
9: 0.2696328, 2.2646761, 0.2617226, 2.2655325, -1.9958997, 2.0029535

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_B2_A1_A2_B1_B1

### Relational analysis result of IS_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2689850, upper bound: 1.2657663
time: 3.44 seconds

## Relational analysis of IS_B2_A1_A2_B1_B2

### Relational analysis result of IS_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2716443
time: 4.01 seconds

## BFS IS instance: IS_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1164227, -5.3457594, -8.1583109, -5.3680158, -2.2927828, 2.3180966
1: -9.1164370, -6.1986699, -9.2176418, -6.2046900, -2.9117470, 3.0189719
2: -9.9214334, -7.0071316, -9.9483604, -6.9536762, -2.6284142, 2.6004648
3: -10.8723545, -8.3479443, -10.8320303, -8.2779474, -2.1861439, 2.0732610
4: -5.5085449, -3.5116096, -5.5529518, -3.5127769, -1.8231597, 1.8911734
5: -8.8817644, -6.1872816, -8.8868132, -6.1921716, -1.9010725, 1.8884981
6: -12.9137135, -9.7120819, -12.9646530, -9.7501202, -2.3663940, 2.4399564
7: 0.3777070, 2.7402911, 0.4067149, 2.8217978, -2.1879930, 2.0700078
8: -3.6396461, -0.9397798, -3.7013464, -0.9872322, -2.4969692, 2.5918827
9: 0.2696328, 2.2646761, 0.1630850, 2.2653689, -1.9957361, 2.1015911

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A1_A2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2912191, upper bound: 1.2739639
time: 3.30 seconds

## Relational analysis of IS_B2_A1_A2_B2_A2

### Relational analysis result of IS_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2912191, upper bound: 1.2739639
time: 3.20 seconds

## BFS IS instance: IS_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.1577110, -5.3693995, -8.1105680, -5.3414979, -2.3389201, 2.2532773
1: -9.2176142, -6.2071805, -9.1143990, -6.2159834, -3.0016308, 2.9072185
2: -9.9468012, -6.9559202, -9.8983250, -7.0054951, -2.5707908, 2.6023231
3: -10.8315735, -8.2780304, -10.8716784, -8.3511286, -2.0792508, 2.1802680
4: -5.5525355, -3.5135365, -5.5085216, -3.5141072, -1.8667831, 1.8349614
5: -8.8861780, -6.1921945, -8.8778696, -6.1869974, -1.9011083, 1.8780422
6: -12.9644756, -9.7506037, -12.9068956, -9.7136793, -2.4430180, 2.3513892
7: 0.4129291, 2.8210230, 0.3788590, 2.7294817, -2.0689945, 2.1660411
8: -3.6999526, -0.9884338, -3.6429410, -0.9402876, -2.5773864, 2.4723845
9: 0.1644568, 2.2648716, 0.2711182, 2.2622728, -2.0978160, 1.9937534

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_B1_B1_A1

### Relational analysis result of IS_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2657662
time: 3.64 seconds

## Relational analysis of IS_B2_A2_B1_B1_A2

### Relational analysis result of IS_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2794836
time: 3.56 seconds

## BFS IS instance: IS_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.1580286, -5.3697662, -8.1164207, -5.3457589, -2.3347583, 2.2753146
1: -9.2176237, -6.2054729, -9.1164341, -6.1986704, -3.0189533, 2.9109612
2: -9.9479847, -6.9562201, -9.9214334, -7.0071354, -2.5763698, 2.6508851
3: -10.8317366, -8.2779818, -10.8723507, -8.3479462, -2.0782838, 2.1808181
4: -5.5522938, -3.5131552, -5.5085440, -3.5116093, -1.8739233, 1.8386989
5: -8.8864403, -6.1922026, -8.8817596, -6.1872821, -1.9005141, 1.8874445
6: -12.9645920, -9.7503700, -12.9137125, -9.7120810, -2.4435005, 2.3617129
7: 0.4111643, 2.8214965, 0.3777118, 2.7402916, -2.0919785, 2.1623187
8: -3.7009468, -0.9892368, -3.6396441, -0.9397831, -2.5833812, 2.5045590
9: 0.1645324, 2.2652781, 0.2696338, 2.2646768, -2.1001444, 1.9956443

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_B1_B2_A1

### Relational analysis result of IS_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739634, upper bound: 1.2739630
time: 3.43 seconds

## Relational analysis of IS_B2_A2_B1_B2_A2

### Relational analysis result of IS_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739634, upper bound: 1.2912191
time: 3.44 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.1148891, -5.3330436, -8.1583109, -5.3680158, -2.2754545, 2.3358598
1: -9.1146488, -6.1993046, -9.2176418, -6.2046900, -2.9099588, 3.0183372
2: -9.9084711, -6.9891744, -9.9483604, -6.9536762, -2.5892334, 2.6212044
3: -10.8748178, -8.3504839, -10.8320303, -8.2779474, -2.1881180, 2.0750599
4: -5.5119309, -3.5088503, -5.5529518, -3.5127769, -1.8304372, 1.8897734
5: -8.8824062, -6.1868315, -8.8868132, -6.1921716, -1.8990870, 1.8937922
6: -12.9083214, -9.7108288, -12.9646530, -9.7501202, -2.3615479, 2.4486601
7: 0.3405662, 2.7348566, 0.4067149, 2.8217978, -2.2374306, 2.0550413
8: -3.6534634, -0.9343410, -3.7013464, -0.9872322, -2.4737272, 2.6008496
9: 0.2617211, 2.2655323, 0.1630850, 2.2653689, -2.0036478, 2.1024473

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B2_A2_B2_A1_A1

### Relational analysis result of IS_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2689851
time: 3.93 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2

### Relational analysis result of IS_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739632, upper bound: 1.2739633
time: 3.93 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.1583176, -5.3680158, -8.1583109, -5.3680158, -2.2993989, 2.2825665
1: -9.2176466, -6.2046900, -9.2176418, -6.2046900, -2.9831243, 2.9581518
2: -9.9483614, -6.9536695, -9.9483604, -6.9536762, -2.6139750, 2.6377890
3: -10.8320341, -8.2779436, -10.8320303, -8.2779474, -2.0857344, 2.0804417
4: -5.5529528, -3.5127759, -5.5529518, -3.5127769, -1.8778234, 1.8941002
5: -8.8868170, -6.1921725, -8.8868132, -6.1921716, -1.9083433, 1.8952355
6: -12.9646549, -9.7501202, -12.9646530, -9.7501202, -2.4190240, 2.4147105
7: 0.4067101, 2.8217998, 0.4067149, 2.8217978, -2.1050892, 2.0796089
8: -3.7013478, -0.9872293, -3.7013464, -0.9872322, -2.5066986, 2.5152221
9: 0.1630857, 2.2653706, 0.1630850, 2.2653689, -2.1022832, 2.1022856

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3147533
time: 3.57 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3479611, upper bound: 1.3479612
time: 3.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 21.86 seconds
IS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2564420
IS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2596612
IS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2794832, upper bound: 1.2596606
IS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2794832, upper bound: 1.2596612
IS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2689850, upper bound: 1.2564420
IS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2623211
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2912191, upper bound: 1.2646407
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2912191, upper bound: 1.2646407
IS_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2564420
IS_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2701554
IS_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2739634, upper bound: 1.2646398
IS_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2739634, upper bound: 1.2815943
IS_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2596611
IS_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2739632, upper bound: 1.2646400
IS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3051362
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.3479611, upper bound: 1.3378974
IS_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2657663
IS_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2689856
IS_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2794832, upper bound: 1.2689856
IS_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2794832, upper bound: 1.2689856
IS_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2689850, upper bound: 1.2657663
IS_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2716443
IS_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2912191, upper bound: 1.2739639
IS_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2912191, upper bound: 1.2739639
IS_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2657662
IS_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2689851, upper bound: 1.2794836
IS_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2739634, upper bound: 1.2739630
IS_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2739634, upper bound: 1.2912191
IS_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2689851
IS_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.2739632, upper bound: 1.2739633
IS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3147533
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.86
Output dim: 7, lower bound: -1.3479611, upper bound: 1.3479612

## BFS IS instance: IS_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.1017628, -5.3419604, -8.0731163, -5.3494797, -2.2621765, 2.2474675
1: -9.1121407, -6.2175026, -9.1009560, -6.2245474, -2.8543110, 2.8516746
2: -9.8964005, -7.0154099, -9.8816700, -7.0472002, -2.5173578, 2.5315228
3: -10.8685255, -8.3544941, -10.8571997, -8.3702316, -2.0494499, 2.0555656
4: -5.5066533, -3.5205691, -5.4967670, -3.5420833, -1.8224249, 1.8331432
5: -8.8716488, -6.1878257, -8.8516674, -6.1961284, -1.8761196, 1.8653982
6: -12.9054070, -9.7149897, -12.8994246, -9.7210789, -2.3594203, 2.3599606
7: 0.3899102, 2.7289963, 0.4257979, 2.7210050, -2.0280724, 2.0010815
8: -3.6411233, -0.9481559, -3.6279364, -0.9735832, -2.4135399, 2.4195859
9: 0.2725549, 2.2609763, 0.2792642, 2.2547159, -1.9821610, 1.9817121

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B1_A1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2564414
time: 3.30 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2564420
time: 3.38 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.1017628, -5.3419604, -8.0788498, -5.3537364, -2.2677879, 2.2680254
1: -9.1121407, -6.2175026, -9.1029472, -6.2072001, -2.8723116, 2.8518686
2: -9.8964005, -7.0154099, -9.9048271, -7.0488420, -2.5345855, 2.5781298
3: -10.8685255, -8.3544941, -10.8578720, -8.3671989, -2.0482178, 2.0547009
4: -5.5066533, -3.5205691, -5.4968624, -3.5395865, -1.8252096, 1.8367219
5: -8.8716488, -6.1878257, -8.8555508, -6.1964049, -1.8755908, 1.8689909
6: -12.9054070, -9.7149897, -12.9062557, -9.7194767, -2.3617663, 2.3693142
7: 0.3899102, 2.7289963, 0.4247084, 2.7318225, -2.0474334, 2.0217032
8: -3.6411233, -0.9481559, -3.6246996, -0.9730840, -2.4292784, 2.4504342
9: 0.2725549, 2.2609763, 0.2777462, 2.2571371, -1.9845822, 1.9832301

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B1_A1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2596607
time: 3.41 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2596612
time: 3.37 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.1017628, -5.3419604, -8.1204453, -5.3759203, -2.2504549, 2.3025684
1: -9.1121407, -6.2175026, -9.2042723, -6.2131233, -2.8990173, 2.9844241
2: -9.8964005, -7.0154099, -9.9321079, -6.9953933, -2.5638671, 2.5690627
3: -10.8685255, -8.3544941, -10.8173695, -8.2978544, -2.1546364, 2.0560484
4: -5.5066533, -3.5205691, -5.5411634, -3.5408163, -1.8066497, 1.8649211
5: -8.8716488, -6.1878257, -8.8605442, -6.2012568, -1.8758163, 1.8752286
6: -12.9054070, -9.7149897, -12.9573345, -9.7571383, -2.3433466, 2.4286799
7: 0.3899102, 2.7289963, 0.4528751, 2.8134508, -2.1716375, 2.0301261
8: -3.6411233, -0.9481559, -3.6860209, -1.0203171, -2.4419050, 2.5599523
9: 0.2725549, 2.2609763, 0.1713096, 2.2579260, -1.9853711, 2.0896668

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2564428
time: 3.90 seconds

## Relational analysis of IS_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2596621
time: 3.56 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.1451473, -5.3784542, -8.1204453, -5.3759203, -2.2744794, 2.2492685
1: -9.2151814, -6.2233510, -9.2042723, -6.2131233, -2.9234920, 2.9120321
2: -9.9355059, -6.9798536, -9.9321079, -6.9953933, -2.5878258, 2.5858026
3: -10.8255901, -8.2821369, -10.8173695, -8.2978544, -2.0530872, 2.0613256
4: -5.5480771, -3.5245481, -5.5411634, -3.5408163, -1.8534727, 1.8700709
5: -8.8762665, -6.1931505, -8.8605442, -6.2012568, -1.8851581, 1.8756950
6: -12.9618950, -9.7545795, -12.9573345, -9.7571383, -2.3989673, 2.3949420
7: 0.4614162, 2.8159013, 0.4528751, 2.8134508, -2.0379148, 2.0542297
8: -3.6895561, -1.0033970, -3.6860209, -1.0203171, -2.4742479, 2.4719841
9: 0.1737354, 2.2605672, 0.1713096, 2.2579260, -2.0841906, 2.0892577

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B1_A1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2596612
time: 3.27 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657658, upper bound: 1.2596608
time: 3.72 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.1075745, -5.3462200, -8.0731163, -5.3494797, -2.2827363, 2.2530742
1: -9.1141558, -6.2001801, -9.1009560, -6.2245474, -2.8545084, 2.8696547
2: -9.9195118, -7.0170517, -9.8816700, -7.0472002, -2.5639729, 2.5487480
3: -10.8691959, -8.3513546, -10.8571997, -8.3702316, -2.0485916, 2.0543914
4: -5.5066934, -3.5180712, -5.4967670, -3.5420833, -1.8260002, 1.8359280
5: -8.8755360, -6.1881118, -8.8516674, -6.1961284, -1.8797197, 1.8648679
6: -12.9122295, -9.7133923, -12.8994246, -9.7210789, -2.3687711, 2.3623099
7: 0.3887777, 2.7398057, 0.4257979, 2.7210050, -2.0486631, 2.0204506
8: -3.6378427, -0.9476552, -3.6279364, -0.9735832, -2.4444380, 2.4353278
9: 0.2710613, 2.2633841, 0.2792642, 2.2547159, -1.9836546, 1.9841199

Time for backsubstitution: 14.40 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.2286510467529297
rel_dist={7: [-1.3704629636566878, 1.370462874585793]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0564516, upper bound: 1.0619857
time: 3.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0634191, upper bound: 1.0634194
time: 3.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.83
Output dim: 7, lower bound: -1.0564516, upper bound: 1.0619857
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.83
Output dim: 7, lower bound: -1.0634191, upper bound: 1.0634194

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.1321526, -5.3759112, -8.1541252, -5.3688135, -1.9647970, 1.9710855
1: -9.2126160, -6.2120185, -9.2219868, -6.2062807, -2.7747984, 2.7768798
2: -9.9340887, -6.9920025, -9.9470692, -6.9679933, -2.3545904, 2.3456397
3: -10.8188086, -8.2860575, -10.8278522, -8.2726040, -1.9381051, 1.9309196
4: -5.5464420, -3.5399156, -5.5548630, -3.5234246, -1.6267018, 1.6196413
5: -8.8613005, -6.2009053, -8.8764372, -6.1933260, -1.5921268, 1.5989120
6: -12.9650011, -9.7570105, -12.9697342, -9.7521791, -2.0641489, 2.0642102
7: 0.4514637, 2.8336329, 0.4246011, 2.8412473, -1.9719834, 1.9891829
8: -3.7049961, -1.0193210, -3.7168264, -1.0001507, -2.3096113, 2.3108845
9: 0.1637008, 2.2586241, 0.1581248, 2.2637877, -2.0663147, 2.0667229

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0401422, upper bound: 1.0429294
time: 3.41 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0433297, upper bound: 1.0488645
time: 3.73 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.1700544, -5.3680086, -8.1700602, -5.3680067, -1.9792852, 2.0021968
1: -9.2260036, -6.2035923, -9.2260036, -6.2035904, -2.7927999, 2.8134942
2: -9.9503212, -6.9502831, -9.9503231, -6.9502764, -2.3886638, 2.3608904
3: -10.8334780, -8.2661514, -10.8334799, -8.2661486, -1.9596353, 1.9643636
4: -5.5582314, -3.5118756, -5.5582323, -3.5118732, -1.6504760, 1.6314802
5: -8.8875713, -6.1918206, -8.8875771, -6.1918206, -1.6045208, 1.6198111
6: -12.9723415, -9.7499933, -12.9723463, -9.7499943, -2.0753284, 2.0790241
7: 0.4052863, 2.8421242, 0.4052820, 2.8421249, -1.9888692, 2.0185976
8: -3.7202139, -0.9862318, -3.7202168, -0.9862270, -2.3451529, 2.3306284
9: 0.1555148, 2.2660573, 0.1555145, 2.2660584, -2.0793943, 2.0765686

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0471929, upper bound: 1.0444868
time: 3.20 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0503690, upper bound: 1.0503708
time: 3.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.63 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.63
Output dim: 7, lower bound: -1.0401422, upper bound: 1.0429294
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.63
Output dim: 7, lower bound: -1.0433297, upper bound: 1.0488645
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.63
Output dim: 7, lower bound: -1.0471929, upper bound: 1.0444868
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.63
Output dim: 7, lower bound: -1.0503690, upper bound: 1.0503708

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.1301994, -5.3804221, -8.1497993, -5.3788061, -1.9465814, 1.9605336
1: -9.2125111, -6.2200603, -9.2217522, -6.2235341, -2.7310925, 2.7430658
2: -9.9290009, -6.9993453, -9.9359570, -6.9842548, -2.3267851, 2.3261123
3: -10.8173113, -8.2863407, -10.8245468, -8.2732182, -1.9359703, 1.9272401
4: -5.5450821, -3.5423148, -5.5518618, -3.5287511, -1.6161823, 1.6076016
5: -8.8592997, -6.2009764, -8.8721075, -6.1934776, -1.5839157, 1.5893018
6: -12.9643784, -9.7585125, -12.9683590, -9.7554379, -2.0507154, 2.0547357
7: 0.4717464, 2.8312354, 0.4684858, 2.8359716, -1.9454055, 1.9381471
8: -3.7005353, -1.0232463, -3.7070985, -1.0085478, -2.2899456, 2.2957726
9: 0.1681457, 2.2569871, 0.1676918, 2.2602358, -2.0278468, 2.0296206

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9824768, upper bound: 0.9929308
time: 3.42 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0240454, upper bound: 1.0260768
time: 3.24 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.1312551, -5.3815145, -8.1580372, -5.3811913, -1.9485745, 1.9899497
1: -9.2125502, -6.2144632, -9.2234488, -6.2070642, -2.7507534, 2.7524223
2: -9.9329100, -7.0001688, -9.9591093, -6.9861135, -2.3348827, 2.3746548
3: -10.8178673, -8.2861738, -10.8253679, -8.2714119, -1.9376245, 1.9270897
4: -5.5443354, -3.5410914, -5.5508871, -3.5263932, -1.6217909, 1.6101027
5: -8.8601456, -6.2010012, -8.8756838, -6.1936774, -1.5831871, 1.5962749
6: -12.9648066, -9.7577724, -12.9752827, -9.7534161, -2.0489721, 2.0678699
7: 0.4650822, 2.8327231, 0.4548550, 2.8502035, -1.9898047, 1.9397564
8: -3.7037830, -1.0257502, -3.7074795, -1.0034161, -2.3027377, 2.3337736
9: 0.1682518, 2.2583351, 0.1652803, 2.2648942, -2.0317273, 2.0518913

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9896258, upper bound: 1.0040019
time: 3.72 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0268035, upper bound: 1.0323422
time: 3.48 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.1680984, -5.3725224, -8.1657276, -5.3779998, -1.9610577, 1.9916387
1: -9.2258930, -6.2116680, -9.2257576, -6.2208872, -2.7489586, 2.7795811
2: -9.9451761, -6.9576263, -9.9391613, -6.9665380, -2.3608294, 2.3413310
3: -10.8319855, -8.2664356, -10.8301716, -8.2667685, -1.9575090, 1.9606869
4: -5.5568714, -3.5142715, -5.5552311, -3.5171962, -1.6399698, 1.6194439
5: -8.8855715, -6.1918893, -8.8832493, -6.1919737, -1.5963125, 1.6102035
6: -12.9717083, -9.7514992, -12.9709606, -9.7532501, -2.0618911, 2.0695560
7: 0.4255710, 2.8397164, 0.4491963, 2.8368356, -1.9622536, 1.9677095
8: -3.7157164, -0.9901538, -3.7104497, -0.9946213, -2.3254499, 2.3155611
9: 0.1599681, 2.2644022, 0.1650845, 2.2624884, -2.0409093, 2.0394115

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9897055, upper bound: 0.9946257
time: 3.35 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0311066, upper bound: 1.0275890
time: 3.32 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.1691523, -5.3736115, -8.1740465, -5.3803864, -1.9630585, 2.0210204
1: -9.2259350, -6.2060428, -9.2274475, -6.2044239, -2.7689810, 2.7889748
2: -9.9491301, -6.9584494, -9.9623280, -6.9683962, -2.3689518, 2.3899083
3: -10.8325357, -8.2662630, -10.8309975, -8.2649584, -1.9591651, 1.9605370
4: -5.5561247, -3.5130486, -5.5542531, -3.5148370, -1.6454153, 1.6219363
5: -8.8864174, -6.1919155, -8.8868275, -6.1921787, -1.5955849, 1.6171880
6: -12.9721470, -9.7507610, -12.9778938, -9.7512283, -2.0602841, 2.0827188
7: 0.4189682, 2.8412151, 0.4355679, 2.8510728, -2.0066562, 1.9691291
8: -3.7189960, -0.9926515, -3.7109232, -0.9894881, -2.3382673, 2.3536077
9: 0.1600727, 2.2657666, 0.1626732, 2.2671456, -2.0448117, 2.0616741

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9968545, upper bound: 1.0056968
time: 3.55 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0338649, upper bound: 1.0338671
time: 3.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.99 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 7, lower bound: -0.9824768, upper bound: 0.9929308
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 7, lower bound: -1.0240454, upper bound: 1.0260768
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 7, lower bound: -0.9896258, upper bound: 1.0040019
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 7, lower bound: -1.0268035, upper bound: 1.0323422
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 7, lower bound: -0.9897055, upper bound: 0.9946257
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 7, lower bound: -1.0311066, upper bound: 1.0275890
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 7, lower bound: -0.9968545, upper bound: 1.0056968
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 7, lower bound: -1.0338649, upper bound: 1.0338671

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.1039419, -5.3764763, -8.0948343, -5.3423328, -1.9447470, 1.9035549
1: -9.1621637, -6.2211642, -9.1103449, -6.2187376, -2.6273365, 2.6227255
2: -9.9149570, -7.0092044, -9.8948793, -7.0232015, -2.2823658, 2.2927263
3: -10.8095570, -8.3332291, -10.8660431, -8.3572073, -1.8179464, 1.8747621
4: -5.5260410, -3.5444942, -5.5051756, -3.5256412, -1.6246142, 1.5713553
5: -8.8548040, -6.2014389, -8.8667545, -6.1885099, -1.5806599, 1.5810244
6: -12.9361696, -9.7575121, -12.9042301, -9.7160435, -2.0349393, 1.9822319
7: 0.4582853, 2.7854729, 0.3986130, 2.7286053, -1.8152862, 1.8887169
8: -3.6739621, -1.0293846, -3.6396751, -0.9543552, -2.2150135, 2.1577983
9: 0.2115762, 2.2535343, 0.2737110, 2.2599435, -1.9922900, 1.9312854

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929309
time: 3.42 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929309
time: 3.75 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.1307764, -5.3759127, -8.1381378, -5.3788147, -1.9464760, 1.9279625
1: -9.2116528, -6.2121477, -9.2134085, -6.2245412, -2.7246370, 2.6255465
2: -9.9338589, -6.9923944, -9.9340878, -6.9876442, -2.3168693, 2.3344731
3: -10.8186407, -8.2874222, -10.8231144, -8.2850170, -1.8290911, 1.9253983
4: -5.5458302, -3.5400190, -5.5465798, -3.5296299, -1.6177883, 1.6141920
5: -8.8612127, -6.2009463, -8.8713617, -6.1938276, -1.5869708, 1.5893786
6: -12.9636583, -9.7570229, -12.9607468, -9.7555523, -2.0477214, 2.0314791
7: 0.4516292, 2.8310819, 0.4699116, 2.8155558, -1.8495941, 1.9377873
8: -3.7024031, -1.0194368, -3.6880379, -1.0095320, -2.2918715, 2.2092466
9: 0.1646426, 2.2585430, 0.1749170, 2.2595663, -2.0378695, 1.9838920

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0204923, upper bound: 1.0260768
time: 3.30 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0204923, upper bound: 1.0260773
time: 3.34 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -8.1039419, -5.3764763, -8.1006136, -5.3465900, -1.9434261, 1.9241319
1: -9.1621637, -6.2211642, -9.1123466, -6.2014055, -2.6572399, 2.6229210
2: -9.9149570, -7.0092044, -9.9179945, -7.0248408, -2.2891335, 2.3393342
3: -10.8095570, -8.3332291, -10.8667145, -8.3540974, -1.8167577, 1.8747833
4: -5.5260410, -3.5444942, -5.5052347, -3.5231447, -1.6308489, 1.5749326
5: -8.8548040, -6.2014389, -8.8706398, -6.1887960, -1.5801311, 1.5896001
6: -12.9361696, -9.7575121, -12.9110546, -9.7144442, -2.0363007, 1.9915845
7: 0.4582853, 2.7854729, 0.3974924, 2.7394192, -1.8346529, 1.8873205
8: -3.6739621, -1.0293846, -3.6364050, -0.9538584, -2.2238665, 2.1886744
9: 0.2115762, 2.2535343, 0.2722095, 2.2623568, -1.9968872, 1.9519496

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9983790
time: 3.46 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0003907
time: 3.54 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.1307764, -5.3759127, -8.1466608, -5.3812013, -1.9478178, 1.9482696
1: -9.2116528, -6.2121477, -9.2151070, -6.2080841, -2.7444153, 2.6247935
2: -9.9338589, -6.9923944, -9.9571867, -6.9895267, -2.3236237, 2.3762264
3: -10.8186407, -8.2874222, -10.8239021, -8.2825403, -1.8278708, 1.9250562
4: -5.5458302, -3.5400190, -5.5456052, -3.5272965, -1.6217337, 1.6178818
5: -8.8612127, -6.2009463, -8.8748837, -6.1940312, -1.5862336, 1.5956745
6: -12.9636583, -9.7570229, -12.9682026, -9.7535324, -2.0467434, 2.0404346
7: 0.4516292, 2.8310819, 0.4562778, 2.8287704, -1.8785238, 1.9384196
8: -3.7024031, -1.0194368, -3.6890674, -1.0044146, -2.3022299, 2.2431703
9: 0.1646426, 2.2585430, 0.1728168, 2.2642145, -2.0425467, 2.0014124

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0204923, upper bound: 1.0295853
time: 3.38 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0204923, upper bound: 1.0294866
time: 3.61 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.1415272, -5.3685589, -8.1105709, -5.3414984, -1.9591789, 1.9346700
1: -9.1757336, -6.2126327, -9.1144009, -6.2159834, -2.6452255, 2.6590605
2: -9.9315119, -6.9674878, -9.8983259, -7.0054879, -2.3166924, 2.3083196
3: -10.8241072, -8.3133202, -10.8716803, -8.3511276, -1.8388410, 1.9082730
4: -5.5378137, -3.5164883, -5.5085230, -3.5141060, -1.6484141, 1.5830665
5: -8.8810616, -6.1923513, -8.8778744, -6.1869965, -1.5930109, 1.6018736
6: -12.9436150, -9.7504921, -12.9068975, -9.7136822, -2.0466704, 1.9966714
7: 0.4121637, 2.7939692, 0.3788552, 2.7294827, -1.8320169, 1.9184585
8: -3.6890268, -0.9963384, -3.6429400, -0.9402828, -2.2507887, 2.1775894
9: 0.2033907, 2.2610557, 0.2711182, 2.2622745, -2.0050745, 1.9410272

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9946257
time: 3.35 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9946257
time: 3.88 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.1686764, -5.3680077, -8.1540546, -5.3780065, -1.9609575, 1.9590905
1: -9.2250195, -6.2037201, -9.2174091, -6.2218895, -2.7425632, 2.6620069
2: -9.9500942, -6.9506745, -9.9373016, -6.9699302, -2.3509712, 2.3496990
3: -10.8333101, -8.2675152, -10.8287373, -8.2785664, -1.8502755, 1.9588394
4: -5.5576210, -3.5119791, -5.5499496, -3.5180748, -1.6415606, 1.6263337
5: -8.8874855, -6.1918612, -8.8825035, -6.1923218, -1.5994115, 1.6102617
6: -12.9709873, -9.7500114, -12.9633560, -9.7533646, -2.0589008, 2.0477219
7: 0.4054527, 2.8395746, 0.4506230, 2.8163366, -1.8670654, 1.9673004
8: -3.7176204, -0.9863477, -3.6914630, -0.9956064, -2.3274221, 2.2290323
9: 0.1564586, 2.2659776, 0.1722679, 2.2618239, -2.0509782, 1.9937696

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0275419, upper bound: 1.0275889
time: 3.35 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0275419, upper bound: 1.0275889
time: 3.86 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -8.1415272, -5.3685589, -8.1164236, -5.3457589, -1.9578590, 1.9552083
1: -9.1757336, -6.2126327, -9.1164341, -6.1986713, -2.6751490, 2.6592588
2: -9.9315119, -6.9674878, -9.9214325, -7.0071297, -2.3234673, 2.3549435
3: -10.8241072, -8.3133202, -10.8723536, -8.3479452, -1.8376741, 1.9082956
4: -5.5378137, -3.5164883, -5.5085444, -3.5116091, -1.6546578, 1.5866418
5: -8.8810616, -6.1923513, -8.8817644, -6.1872816, -1.5924807, 1.6104238
6: -12.9436150, -9.7504921, -12.9137135, -9.7120819, -2.0480509, 2.0060406
7: 0.4121637, 2.7939692, 0.3777075, 2.7402921, -1.8513880, 1.9169369
8: -3.6890268, -0.9963384, -3.6396461, -0.9397793, -2.2596288, 2.2085400
9: 0.2033907, 2.2610557, 0.2696337, 2.2646773, -2.0097013, 1.9616790

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0000742
time: 3.69 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0020837
time: 3.61 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.1686764, -5.3680077, -8.1626568, -5.3803930, -1.9623003, 1.9793606
1: -9.2250195, -6.2037201, -9.2191048, -6.2054434, -2.7623730, 2.6612558
2: -9.9500942, -6.9506745, -9.9604149, -6.9718108, -2.3577309, 2.3914909
3: -10.8333101, -8.2675152, -10.8295307, -8.2760296, -1.8490591, 1.9585004
4: -5.5576210, -3.5119791, -5.5489697, -3.5157418, -1.6455040, 1.6300020
5: -8.8874855, -6.1918612, -8.8860245, -6.1925268, -1.5986729, 1.6165748
6: -12.9709873, -9.7500114, -12.9707804, -9.7513475, -2.0580459, 2.0566940
7: 0.4054527, 2.8395746, 0.4369941, 2.8296504, -1.8960280, 1.9677119
8: -3.7176204, -0.9863477, -3.6925058, -0.9904881, -2.3377686, 2.2630329
9: 0.1564586, 2.2659776, 0.1701901, 2.2664709, -2.0556216, 2.0112777

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0275419, upper bound: 1.0311092
time: 3.51 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0275419, upper bound: 1.0310023
time: 3.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.65 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929309
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929309
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -1.0204923, upper bound: 1.0260768
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -1.0204923, upper bound: 1.0260773
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9983790
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0003907
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -1.0204923, upper bound: 1.0295853
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -1.0204923, upper bound: 1.0294866
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9946257
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9946257
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -1.0275419, upper bound: 1.0275889
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -1.0275419, upper bound: 1.0275889
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0000742
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0020837
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -1.0275419, upper bound: 1.0311092
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.65
Output dim: 7, lower bound: -1.0275419, upper bound: 1.0310023

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -8.1278324, -5.3859034, -8.0948343, -5.3423328, -1.9735885, 1.8874426
1: -9.2123814, -6.2292414, -9.1103449, -6.2187376, -2.6842098, 2.6002231
2: -9.9230490, -7.0082626, -9.8948793, -7.0232015, -2.2942371, 2.2827926
3: -10.8154984, -8.2866716, -10.8660431, -8.3572073, -1.8216667, 1.9284139
4: -5.5434432, -3.5452466, -5.5051756, -3.5256412, -1.6321130, 1.5705757
5: -8.8569717, -6.2010608, -8.8667545, -6.1885099, -1.5806017, 1.5824158
6: -12.9636412, -9.7602673, -12.9042301, -9.7160435, -2.0730343, 1.9729178
7: 0.4953136, 2.8283672, 0.3986130, 2.7286053, -1.7708559, 1.9567327
8: -3.6953039, -1.0277319, -3.6396751, -0.9543552, -2.2723989, 2.1483865
9: 0.1732543, 2.2550933, 0.2737110, 2.2599435, -2.0169497, 1.9091530

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863948
time: 3.56 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929309
time: 3.58 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -8.1359806, -5.3882895, -8.0948343, -5.3423328, -1.9977875, 1.8989229
1: -9.2140770, -6.2127647, -9.1103449, -6.2187376, -2.6853151, 2.6162381
2: -9.9462013, -7.0101213, -9.8948793, -7.0232015, -2.3360753, 2.2974291
3: -10.8163214, -8.2848644, -10.8660431, -8.3572073, -1.8225942, 1.9293933
4: -5.5424709, -3.5428865, -5.5051756, -3.5256412, -1.6340814, 1.5727806
5: -8.8605442, -6.2012596, -8.8667545, -6.1885099, -1.5838780, 1.5818541
6: -12.9705677, -9.7582445, -12.9042301, -9.7160435, -2.0829988, 1.9755123
7: 0.4816909, 2.8425953, 0.3986130, 2.7286053, -1.7986917, 1.9899416
8: -3.6956553, -1.0225992, -3.6396751, -0.9543552, -2.3062305, 2.1668339
9: 0.1708477, 2.2597508, 0.2737110, 2.2599435, -2.0191550, 1.9138436

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863936
time: 3.88 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929297
time: 3.71 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -8.1278324, -5.3859034, -8.1381378, -5.3788147, -1.9449463, 1.9145525
1: -9.2123814, -6.2292414, -9.2134085, -6.2245412, -2.7079210, 2.5967917
2: -9.9230490, -7.0082626, -9.9340878, -6.9876442, -2.3086095, 2.3112907
3: -10.8154984, -8.2866716, -10.8231144, -8.2850170, -1.8264751, 1.9256680
4: -5.5434432, -3.5452466, -5.5465798, -3.5296299, -1.6099415, 1.6090288
5: -8.8569717, -6.2010608, -8.8713617, -6.1938276, -1.5800467, 1.5849831
6: -12.9636412, -9.7602673, -12.9607468, -9.7555523, -2.0447626, 2.0222366
7: 0.4953136, 2.8283672, 0.4699116, 2.8155558, -1.8026514, 1.9353914
8: -3.6953039, -1.0277319, -3.6880379, -1.0095320, -2.2849498, 2.1925650
9: 0.1732543, 2.2550933, 0.1749170, 2.2595663, -2.0139527, 1.9584656

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863954
time: 3.42 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0260323
time: 3.35 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -8.1359806, -5.3882895, -8.1381378, -5.3788147, -1.9691453, 1.9201598
1: -9.2140770, -6.2127647, -9.2134085, -6.2245412, -2.7090263, 2.6140876
2: -9.9462013, -7.0101213, -9.9340878, -6.9876442, -2.3519402, 2.3259273
3: -10.8163214, -8.2848644, -10.8231144, -8.2850170, -1.8262801, 1.9266474
4: -5.5424709, -3.5428865, -5.5465798, -3.5296299, -1.6119108, 1.6115603
5: -8.8605442, -6.2012596, -8.8713617, -6.1938276, -1.5833254, 1.5844214
6: -12.9705677, -9.7582445, -12.9607468, -9.7555523, -2.0547266, 2.0245743
7: 0.4816909, 2.8425953, 0.4699116, 2.8155558, -1.8245325, 1.9688749
8: -3.6956553, -1.0225992, -3.6880379, -1.0095320, -2.3187823, 2.2126007
9: 0.1708477, 2.2597508, 0.1749170, 2.2595663, -2.0161586, 1.9635477

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863934
time: 4.01 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0260323
time: 3.43 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -8.1278324, -5.3859034, -8.1006136, -5.3465900, -1.9791980, 1.9080191
1: -9.2123814, -6.2292414, -9.1123466, -6.2014055, -2.7021961, 2.6004186
2: -9.9230490, -7.0082626, -9.9179945, -7.0248408, -2.3114591, 2.3294005
3: -10.8154984, -8.2866716, -10.8667145, -8.3540974, -1.8204780, 1.9275632
4: -5.5434432, -3.5452466, -5.5052347, -3.5231447, -1.6348944, 1.5741534
5: -8.8569717, -6.2010608, -8.8706398, -6.1887960, -1.5800719, 1.5860140
6: -12.9636412, -9.7602673, -12.9110546, -9.7144442, -2.0753841, 1.9822705
7: 0.4953136, 2.8283672, 0.3974924, 2.7394192, -1.7902231, 1.9773200
8: -3.6953039, -1.0277319, -3.6364050, -0.9538584, -2.2881479, 2.1792622
9: 0.1732543, 2.2550933, 0.2722095, 2.2623568, -2.0215468, 1.9116359

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9880128
time: 3.47 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9983793
time: 3.51 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -8.1359806, -5.3882895, -8.1006136, -5.3465900, -1.9884853, 1.9025383
1: -9.2140770, -6.2127647, -9.1123466, -6.2014055, -2.7174826, 2.6251540
2: -9.9462013, -7.0101213, -9.9179945, -7.0248408, -2.3341522, 2.3243859
3: -10.8163214, -8.2848644, -10.8667145, -8.3540974, -1.8215618, 1.9295745
4: -5.5424709, -3.5428865, -5.5052347, -3.5231447, -1.6428771, 1.5816450
5: -8.8605442, -6.2012596, -8.8706398, -6.1887960, -1.5876904, 1.5921659
6: -12.9705677, -9.7582445, -12.9110546, -9.7144442, -2.0818629, 1.9789019
7: 0.4816909, 2.8425953, 0.3974924, 2.7394192, -1.7786427, 1.9648085
8: -3.6956553, -1.0225992, -3.6364050, -0.9538584, -2.3087988, 2.1840045
9: 0.1708477, 2.2597508, 0.2722095, 2.2623568, -2.0508204, 1.9453163

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9951624
time: 3.62 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0040025
time: 3.62 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.1278324, -5.3859034, -8.1466608, -5.3812013, -1.9564261, 1.9348593
1: -9.2123814, -6.2292414, -9.2151070, -6.2080841, -2.7239304, 2.5960388
2: -9.9230490, -7.0082626, -9.9571867, -6.9895267, -2.3254256, 2.3530440
3: -10.8154984, -8.2866716, -10.8239021, -8.2825403, -1.8252554, 1.9265635
4: -5.5434432, -3.5452466, -5.5456052, -3.5272965, -1.6121545, 1.6127186
5: -8.8569717, -6.2010608, -8.8748837, -6.1940312, -1.5793095, 1.5882246
6: -12.9636412, -9.7602673, -12.9682026, -9.7535324, -2.0473895, 2.0311918
7: 0.4953136, 2.8283672, 0.4562778, 2.8287704, -1.8315811, 1.9629557
8: -3.6953039, -1.0277319, -3.6890674, -1.0044146, -2.3034201, 2.2264881
9: 0.1732543, 2.2550933, 0.1728168, 2.2642145, -2.0186300, 1.9594188

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9880123
time: 3.75 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0295847
time: 3.67 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -8.1359806, -5.3882895, -8.1466608, -5.3812013, -1.9625058, 1.9277949
1: -9.2140770, -6.2127647, -9.2151070, -6.2080841, -2.7310677, 2.6383052
2: -9.9462013, -7.0101213, -9.9571867, -6.9895267, -2.3499293, 2.3493195
3: -10.8163214, -8.2848644, -10.8239021, -8.2825403, -1.8267145, 1.9264653
4: -5.5424709, -3.5428865, -5.5456052, -3.5272965, -1.6184149, 1.6223822
5: -8.8605442, -6.2012596, -8.8748837, -6.1940312, -1.5909615, 1.5924542
6: -12.9705677, -9.7582445, -12.9682026, -9.7535324, -2.0512514, 2.0320139
7: 0.4816909, 2.8425953, 0.4562778, 2.8287704, -1.8137293, 1.9454961
8: -3.6956553, -1.0225992, -3.6890674, -1.0044146, -2.3228569, 2.2317801
9: 0.1708477, 2.2597508, 0.1728168, 2.2642145, -2.0461884, 1.9923372

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9951611
time: 3.77 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0323422
time: 3.98 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -8.1657238, -5.3779998, -8.1105709, -5.3414984, -1.9880199, 1.9185467
1: -9.2257586, -6.2208891, -9.1144009, -6.2159834, -2.7021704, 2.6363707
2: -9.9391594, -6.9665422, -9.8983259, -7.0054879, -2.3281870, 2.2983890
3: -10.8301706, -8.2667694, -10.8716803, -8.3511276, -1.8427410, 1.9621418
4: -5.5552292, -3.5171974, -5.5085230, -3.5141060, -1.6561074, 1.5823555
5: -8.8832455, -6.1919756, -8.8778744, -6.1869965, -1.5930080, 1.6032724
6: -12.9709625, -9.7532539, -12.9068975, -9.7136822, -2.0847831, 1.9873502
7: 0.4491997, 2.8368356, 0.3788552, 2.7294827, -1.7880058, 1.9864397
8: -3.7104492, -0.9946251, -3.6429400, -0.9402828, -2.3080740, 2.1680613
9: 0.1650854, 2.2624879, 0.2711182, 2.2622745, -2.0297227, 1.9188051

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9880898
time: 3.42 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9946258
time: 3.46 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -8.1740389, -5.3803868, -8.1105709, -5.3414984, -2.0121498, 1.9300275
1: -9.2274456, -6.2044249, -9.1144009, -6.2159834, -2.7032785, 2.6523561
2: -9.9623280, -6.9684019, -9.8983259, -7.0054879, -2.3700857, 2.3130255
3: -10.8309956, -8.2649622, -10.8716803, -8.3511276, -1.8436708, 1.9631228
4: -5.5542521, -3.5148377, -5.5085230, -3.5141060, -1.6580544, 1.5845613
5: -8.8868237, -6.1921759, -8.8778744, -6.1869965, -1.5962963, 1.6027091
6: -12.9778900, -9.7512283, -12.9068975, -9.7136822, -2.0947447, 1.9899490
7: 0.4355721, 2.8510728, 0.3788552, 2.7294827, -1.8157668, 2.0188584
8: -3.7109237, -0.9894915, -3.6429400, -0.9402828, -2.3419466, 2.1865110
9: 0.1626736, 2.2671447, 0.2711182, 2.2622745, -2.0320106, 1.9234133

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9880886
time: 3.97 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9946246
time: 3.82 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -8.1657238, -5.3779998, -8.1540546, -5.3780065, -1.9594049, 1.9456806
1: -9.2257586, -6.2208891, -9.2174091, -6.2218895, -2.7257366, 2.6331730
2: -9.9391594, -6.9665422, -9.9373016, -6.9699302, -2.3426380, 2.3265166
3: -10.8301706, -8.2667694, -10.8287373, -8.2785664, -1.8476577, 1.9591160
4: -5.5552292, -3.5171974, -5.5499496, -3.5180748, -1.6337333, 1.6211610
5: -8.8832455, -6.1919756, -8.8825035, -6.1923218, -1.5924764, 1.6058760
6: -12.9709625, -9.7532539, -12.9633560, -9.7533646, -2.0559392, 2.0384414
7: 0.4491997, 2.8368356, 0.4506230, 2.8163366, -1.8200974, 1.9649024
8: -3.7104492, -0.9946251, -3.6914630, -0.9956064, -2.3204098, 2.2123802
9: 0.1650854, 2.2624879, 0.1722679, 2.2618239, -2.0269251, 1.9681478

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9880894
time: 3.82 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0275432
time: 4.20 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -8.1740389, -5.3803868, -8.1540546, -5.3780065, -1.9835343, 1.9512811
1: -9.2274456, -6.2044249, -9.2174091, -6.2218895, -2.7268438, 2.6504788
2: -9.9623280, -6.9684019, -9.9373016, -6.9699302, -2.3860569, 2.3411527
3: -10.8309956, -8.2649622, -10.8287373, -8.2785664, -1.8474727, 1.9600971
4: -5.5542521, -3.5148377, -5.5499496, -3.5180748, -1.6356802, 1.6236959
5: -8.8868237, -6.1921759, -8.8825035, -6.1923218, -1.5957394, 1.6053128
6: -12.9778900, -9.7512283, -12.9633560, -9.7533646, -2.0659008, 2.0408139
7: 0.4355721, 2.8510728, 0.4506230, 2.8163366, -1.8423986, 1.9973207
8: -3.7109237, -0.9894915, -3.6914630, -0.9956064, -2.3542814, 2.2323682
9: 0.1626736, 2.2671447, 0.1722679, 2.2618239, -2.0292130, 1.9732332

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9880883
time: 4.09 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0275432
time: 4.01 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -8.1657238, -5.3779998, -8.1164236, -5.3457589, -1.9936261, 1.9390850
1: -9.2257586, -6.2208891, -9.1164341, -6.1986713, -2.7201433, 2.6365690
2: -9.9391594, -6.9665422, -9.9214325, -7.0071297, -2.3454151, 2.3450129
3: -10.8301706, -8.2667694, -10.8723536, -8.3479452, -1.8415737, 1.9612701
4: -5.5552292, -3.5171974, -5.5085444, -3.5116091, -1.6588979, 1.5859308
5: -8.8832455, -6.1919756, -8.8817644, -6.1872816, -1.5924778, 1.6068752
6: -12.9709625, -9.7532539, -12.9137135, -9.7120819, -2.0871310, 1.9967194
7: 0.4491997, 2.8368356, 0.3777075, 2.7402921, -1.8073769, 2.0070148
8: -3.7104492, -0.9946251, -3.6396461, -0.9397793, -2.3238077, 2.1990123
9: 0.1650854, 2.2624879, 0.2696337, 2.2646773, -2.0343494, 1.9213123

Time for backsubstitution: 16.91 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9897077
time: 4.14 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0000742
time: 3.30 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -8.1740389, -5.3803868, -8.1164236, -5.3457589, -2.0028472, 1.9336119
1: -9.2274456, -6.2044249, -9.1164341, -6.1986713, -2.7358122, 2.6614847
2: -9.9623280, -6.9684019, -9.9214325, -7.0071297, -2.3681374, 2.3399997
3: -10.8309956, -8.2649622, -10.8723536, -8.3479452, -1.8426604, 1.9633052
4: -5.5542521, -3.5148377, -5.5085444, -3.5116091, -1.6666808, 1.5933771
5: -8.8868237, -6.1921759, -8.8817644, -6.1872816, -1.6001225, 1.6129887
6: -12.9778900, -9.7512283, -12.9137135, -9.7120819, -2.0936203, 1.9936223
7: 0.4355721, 2.8510728, 0.3777075, 2.7402921, -1.7957292, 1.9944024
8: -3.7109237, -0.9894915, -3.6396461, -0.9397793, -2.3445020, 2.2037263
9: 0.1626736, 2.2671447, 0.2696337, 2.2646773, -2.0636458, 1.9549532

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9968574
time: 3.45 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0056989
time: 3.70 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.1657238, -5.3779998, -8.1626568, -5.3803930, -1.9708862, 1.9659507
1: -9.2257586, -6.2208891, -9.2191048, -6.2054434, -2.7417326, 2.6324220
2: -9.9391594, -6.9665422, -9.9604149, -6.9718108, -2.3594594, 2.3683081
3: -10.8301706, -8.2667694, -10.8295307, -8.2760296, -1.8464403, 1.9600146
4: -5.5552292, -3.5171974, -5.5489697, -3.5157418, -1.6359453, 1.6248293
5: -8.8832455, -6.1919756, -8.8860245, -6.1925268, -1.5917382, 1.6091206
6: -12.9709625, -9.7532539, -12.9707804, -9.7513475, -2.0585666, 2.0474136
7: 0.4491997, 2.8368356, 0.4369941, 2.8296504, -1.8490601, 1.9925518
8: -3.7104492, -0.9946251, -3.6925058, -0.9904881, -2.3388858, 2.2463808
9: 0.1650854, 2.2624879, 0.1701901, 2.2664709, -2.0315681, 1.9691281

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9897067
time: 3.43 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0311085
time: 3.51 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -8.1740389, -5.3803868, -8.1626568, -5.3803930, -1.9768944, 1.9588671
1: -9.2274456, -6.2044249, -9.2191048, -6.2054434, -2.7492638, 2.6747146
2: -9.9623280, -6.9684019, -9.9604149, -6.9718108, -2.3839893, 2.3645692
3: -10.8309956, -8.2649622, -10.8295307, -8.2760296, -1.8479033, 1.9599178
4: -5.5542521, -3.5148377, -5.5489697, -3.5157418, -1.6420059, 1.6344962
5: -8.8868237, -6.1921759, -8.8860245, -6.1925268, -1.6033263, 1.6133549
6: -12.9778900, -9.7512283, -12.9707804, -9.7513475, -2.0625410, 2.0482743
7: 0.4355721, 2.8510728, 0.4369941, 2.8296504, -1.8313346, 1.9747982
8: -3.7109237, -0.9894915, -3.6925058, -0.9904881, -2.3583441, 2.2516155
9: 0.1626736, 2.2671447, 0.1701901, 2.2664709, -2.0591598, 2.0021825

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9968565
time: 3.63 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9968567
time: 3.90 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.11 seconds
IS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863948
IS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929309
IS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863936
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929297
IS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863954
IS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0260323
IS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863934
IS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0260323
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9880128
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9983793
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9951624
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0040025
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9880123
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0295847
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9951611
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0323422
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9880898
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9946258
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9880886
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9946246
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9880894
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0275432
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9880883
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0275432
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9897077
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0000742
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9968574
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0056989
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9897067
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0311085
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9968565
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9968567

## BFS IS instance: IS_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -8.0731163, -5.3494797, -8.0948343, -5.3423328, -1.9099784, 1.9160233
1: -9.1009560, -6.2245474, -9.1103449, -6.2187376, -2.5608959, 2.5626888
2: -9.8816700, -7.0472002, -9.8948793, -7.0232015, -2.2634721, 2.2549357
3: -10.8571997, -8.3702316, -10.8660431, -8.3572073, -1.8233418, 1.8164954
4: -5.4967670, -3.5420833, -5.5051756, -3.5256412, -1.5985904, 1.5913754
5: -8.8516674, -6.1961284, -8.8667545, -6.1885099, -1.5763845, 1.5830777
6: -12.8994246, -9.7210789, -12.9042301, -9.7160435, -1.9976006, 1.9968879
7: 0.4257979, 2.7210050, 0.3986130, 2.7286053, -1.7905068, 1.8093674
8: -3.6279364, -0.9735832, -3.6396751, -0.9543552, -2.1337833, 2.1353078
9: 0.2792642, 2.2547159, 0.2737110, 2.2599435, -1.9113331, 1.9114275

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863948
time: 3.58 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863949
time: 3.60 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -8.1161938, -5.3859129, -8.0948343, -5.3423328, -1.9622698, 1.8874030
1: -9.2040396, -6.2302504, -9.1103449, -6.2187376, -2.6767836, 2.5865254
2: -9.9211702, -7.0116529, -9.8948793, -7.0232015, -2.2924032, 2.2777495
3: -10.8140755, -8.2984705, -10.8660431, -8.3572073, -1.8204279, 1.9210980
4: -5.5381622, -3.5461254, -5.5051756, -3.5256412, -1.6219277, 1.5692682
5: -8.8562260, -6.2014089, -8.8667545, -6.1885099, -1.5788965, 1.5779698
6: -12.9560547, -9.7603846, -12.9042301, -9.7160435, -2.0625811, 1.9693034
7: 0.4967237, 2.8080168, 0.3986130, 2.7286053, -1.7699032, 1.9488652
8: -3.6762104, -1.0287127, -3.6396751, -0.9543552, -2.2659397, 2.1476007
9: 0.1805308, 2.2544208, 0.2737110, 2.2599435, -2.0099707, 1.9084911

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929309
time: 3.56 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929309
time: 3.66 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.0788498, -5.3537364, -8.0948343, -5.3423328, -1.9305363, 1.9216347
1: -9.1029472, -6.2072001, -9.1103449, -6.2187376, -2.5610905, 2.5806880
2: -9.9048271, -7.0488420, -9.8948793, -7.0232015, -2.3100791, 2.2721634
3: -10.8578720, -8.3671989, -10.8660431, -8.3572073, -1.8224773, 1.8152628
4: -5.4968624, -3.5395865, -5.5051756, -3.5256412, -1.6021695, 1.5941601
5: -8.8555508, -6.1964049, -8.8667545, -6.1885099, -1.5799770, 1.5825489
6: -12.9062557, -9.7194767, -12.9042301, -9.7160435, -2.0069542, 1.9992340
7: 0.4247084, 2.7318225, 0.3986130, 2.7286053, -1.8111286, 1.8287280
8: -3.6246996, -0.9730840, -3.6396751, -0.9543552, -2.1646318, 2.1510463
9: 0.2777462, 2.2571371, 0.2737110, 2.2599435, -1.9137836, 1.9160180

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9824767, upper bound: 0.9863948
time: 3.56 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863953
time: 3.54 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -8.1246204, -5.3883009, -8.0948343, -5.3423328, -1.9864569, 1.8988829
1: -9.2057343, -6.2137890, -9.1103449, -6.2187376, -2.6781435, 2.6025486
2: -9.9442692, -7.0135350, -9.8948793, -7.0232015, -2.3341351, 2.2923946
3: -10.8148670, -8.2960634, -10.8660431, -8.3572073, -1.8213253, 1.9221470
4: -5.5371871, -3.5437903, -5.5051756, -3.5256412, -1.6232276, 1.5714798
5: -8.8597441, -6.2016115, -8.8667545, -6.1885099, -1.5821309, 1.5774055
6: -12.9635191, -9.7583609, -12.9042301, -9.7160435, -2.0728722, 1.9719269
7: 0.4830995, 2.8211555, 0.3986130, 2.7286053, -1.7976580, 1.9825048
8: -3.6772470, -1.0235934, -3.6396751, -0.9543552, -2.2980785, 2.1660452
9: 0.1784050, 2.2590675, 0.2737110, 2.2599435, -2.0126910, 1.9131966

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9824767, upper bound: 0.9929309
time: 3.51 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929295
time: 4.05 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 22.19 seconds
IS_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863948
IS_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863949
IS_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929309
IS_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929309
IS_A1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.19
Output dim: 7, lower bound: -0.9824767, upper bound: 0.9863948
IS_A1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863953
IS_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.19
Output dim: 7, lower bound: -0.9824767, upper bound: 0.9929309
IS_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9929295
IS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863954
IS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0260323
IS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9863934
IS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0260323
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9880128
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9983793
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9951624
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0040025
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9880123
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0295847
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 0.9951611
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9808591, upper bound: 1.0323422
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9880898
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9946258
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9880886
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9946246
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9880894
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0275432
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9880883
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0275432
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9897077
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0000742
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9968574
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0056989
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9897067
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 1.0311085
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9968565
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.19
Output dim: 7, lower bound: -0.9880877, upper bound: 0.9968567
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.0185980796813965
rel_dist={7: [-1.0634246550232, 1.0634268200891506]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 6181

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7927058, upper bound: 0.7908401
time: 5.86 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7927058, upper bound: 0.7927019
time: 4.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.17 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 10.17
Output dim: 7, lower bound: -0.7927058, upper bound: 0.7908401
IS_B2, status: Status.UNKNOWN, split count: 1, time: 10.17
Output dim: 7, lower bound: -0.7927058, upper bound: 0.7927019

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -8.1438475, -5.3693533, -8.1321526, -5.3759112, -1.7367940, 1.7392712
1: -9.2193661, -6.2081175, -9.2126160, -6.2120185, -2.5805340, 2.5798807
2: -9.9449568, -6.9793901, -9.9340887, -6.9920025, -2.1695905, 2.1701241
3: -10.8242207, -8.2769356, -10.8188086, -8.2860575, -1.7751293, 1.7807391
4: -5.5527210, -3.5308518, -5.5464420, -3.5399156, -1.4643035, 1.4664822
5: -8.8692360, -6.1943703, -8.8613005, -6.2009053, -1.3996754, 1.3987799
6: -12.9680328, -9.7536488, -12.9650011, -9.7570105, -1.8212919, 1.8219295
7: 0.4370689, 2.8406568, 0.4514637, 2.8336329, -1.8380723, 1.8311238
8: -3.7145891, -1.0091510, -3.7049961, -1.0193210, -2.1241279, 2.1191883
9: 0.1598777, 2.2622950, 0.1637008, 2.2586241, -1.9715652, 1.9713230

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7877629, upper bound: 0.7844324
time: 4.50 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7888895, upper bound: 0.7870232
time: 4.70 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -8.1700554, -5.3680096, -8.1700544, -5.3680086, -1.7776303, 1.7506766
1: -9.2260036, -6.2035933, -9.2260036, -6.2035923, -2.6175137, 2.5969296
2: -9.9503222, -6.9502802, -9.9503212, -6.9502831, -2.1843905, 2.2148056
3: -10.8334799, -8.2661505, -10.8334780, -8.2661514, -1.8109612, 1.8062468
4: -5.5582309, -3.5118744, -5.5582314, -3.5118756, -1.4766459, 1.4974513
5: -8.8875732, -6.1918216, -8.8875713, -6.1918206, -1.4276528, 1.4109085
6: -12.9723434, -9.7499933, -12.9723415, -9.7499933, -1.8378420, 1.8341618
7: 0.4052849, 2.8421261, 0.4052863, 2.8421242, -1.8785605, 1.8460064
8: -3.7202153, -0.9862299, -3.7202139, -0.9862318, -2.1421757, 2.1606879
9: 0.1555160, 2.2660570, 0.1555148, 2.2660573, -1.9827895, 1.9856052

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7863214, upper bound: 0.7877626
time: 4.42 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7888893, upper bound: 0.7888881
time: 5.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.76 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 24.76
Output dim: 7, lower bound: -0.7877629, upper bound: 0.7844324
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 24.76
Output dim: 7, lower bound: -0.7888895, upper bound: 0.7870232
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 24.76
Output dim: 7, lower bound: -0.7863214, upper bound: 0.7877626
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 24.76
Output dim: 7, lower bound: -0.7888893, upper bound: 0.7888881

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -8.1404819, -5.3771114, -8.1278324, -5.3859034, -1.7176170, 1.7232375
1: -9.2191820, -6.2216749, -9.2123814, -6.2292414, -2.5311403, 2.5345769
2: -9.9362364, -6.9920187, -9.9230490, -7.0082626, -2.1389351, 2.1429429
3: -10.8216524, -8.2774181, -10.8154984, -8.2866716, -1.7718902, 1.7768674
4: -5.5503883, -3.5349774, -5.5434432, -3.5452466, -1.4507937, 1.4523678
5: -8.8658495, -6.1944904, -8.8569717, -6.2010608, -1.3890924, 1.3876257
6: -12.9669552, -9.7561951, -12.9636412, -9.7602673, -1.8064642, 1.8087385
7: 0.4717989, 2.8365469, 0.4953136, 2.8283672, -1.7952518, 1.7788568
8: -3.7069578, -1.0158901, -3.6953039, -1.0277319, -2.1019258, 2.0988936
9: 0.1673906, 2.2594929, 0.1732543, 2.2550933, -1.9249372, 1.9255567

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B1_B1_B1

### Relational analysis result of IS_B1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7684810, upper bound: 0.7675624
time: 4.84 seconds

## Relational analysis of IS_B1_B1_B2

### Relational analysis result of IS_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7823014, upper bound: 0.7782524
time: 5.76 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -8.1423512, -5.3789668, -8.1359806, -5.3882895, -1.7221513, 1.7563510
1: -9.2192526, -6.2122517, -9.2140770, -6.2127647, -2.5493555, 2.5505490
2: -9.9429579, -6.9933891, -9.9462013, -7.0101213, -2.1506948, 2.1962266
3: -10.8226137, -8.2771320, -10.8163214, -8.2848644, -1.7740426, 1.7765162
4: -5.5491085, -3.5328336, -5.5424709, -3.5428865, -1.4573097, 1.4552965
5: -8.8673220, -6.1945353, -8.8605442, -6.2012596, -1.3883047, 1.3946657
6: -12.9677143, -9.7549267, -12.9705677, -9.7582445, -1.8047123, 1.8241866
7: 0.4602118, 2.8391309, 0.4816909, 2.8425953, -1.8496642, 1.7826896
8: -3.7125435, -1.0201621, -3.6956553, -1.0225992, -2.1188269, 2.1399155
9: 0.1675760, 2.2618110, 0.1708477, 2.2597508, -1.9282413, 1.9499869

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7721265, upper bound: 0.7729493
time: 4.66 seconds

## Relational analysis of IS_B1_B2_B2

### Relational analysis result of IS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7833717, upper bound: 0.7815073
time: 5.32 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -8.1657238, -5.3780003, -8.1666880, -5.3757653, -1.7615809, 1.7314897
1: -9.2257586, -6.2208891, -9.2258101, -6.2172070, -2.5720482, 2.5473680
2: -9.9391623, -6.9665413, -9.9415369, -6.9629092, -2.1571312, 2.1841111
3: -10.8301697, -8.2667675, -10.8309088, -8.2666368, -1.8070917, 1.8030124
4: -5.5552301, -3.5171971, -5.5558963, -3.5159910, -1.4625740, 1.4839764
5: -8.8832474, -6.1919746, -8.8841858, -6.1919422, -1.4165030, 1.4003315
6: -12.9709616, -9.7532520, -12.9712524, -9.7525415, -1.8246546, 1.8193271
7: 0.4491982, 2.8368359, 0.4400659, 2.8379993, -1.8265762, 1.8033457
8: -3.7104492, -0.9946232, -3.7125354, -0.9929681, -2.1218920, 2.1384330
9: 0.1650847, 2.2624869, 0.1630304, 2.2632337, -1.9368782, 1.9389753

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7688954
time: 4.47 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823007
time: 4.59 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -8.1740417, -5.3803864, -8.1685543, -5.3776188, -1.7946992, 1.7360344
1: -9.2274466, -6.2044239, -9.2258844, -6.2077346, -2.5880785, 2.5659404
2: -9.9623289, -6.9684000, -9.9483147, -6.9642820, -2.2104821, 2.1959047
3: -10.8309956, -8.2649593, -10.8318720, -8.2663422, -1.8067365, 1.8051658
4: -5.5542531, -3.5148380, -5.5546207, -3.5138516, -1.4654546, 1.4903440
5: -8.8868256, -6.1921778, -8.8856649, -6.1919842, -1.4235644, 1.3995435
6: -12.9778910, -9.7512283, -12.9720230, -9.7512722, -1.8401289, 1.8178539
7: 0.4355698, 2.8510737, 0.4284830, 2.8405981, -1.8303685, 1.8575420
8: -3.7109232, -0.9894896, -3.7181625, -0.9972324, -2.1630011, 2.1553926
9: 0.1626729, 2.2671459, 0.1632159, 2.2655725, -1.9613080, 1.9422607

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7751179, upper bound: 0.7725652
time: 4.41 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7833717, upper bound: 0.7833741
time: 4.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.58 seconds
IS_B1_B1_B1, status: Status.VERIFIED, split count: 3, time: 23.58
Output dim: 7, lower bound: -0.7684810, upper bound: 0.7675624
IS_B1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 23.58
Output dim: 7, lower bound: -0.7823014, upper bound: 0.7782524
IS_B1_B2_B1, status: Status.VERIFIED, split count: 3, time: 23.58
Output dim: 7, lower bound: -0.7721265, upper bound: 0.7729493
IS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 23.58
Output dim: 7, lower bound: -0.7833717, upper bound: 0.7815073
IS_B2_A1_A1, status: Status.VERIFIED, split count: 3, time: 23.58
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7688954
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 23.58
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823007
IS_B2_A2_A1, status: Status.VERIFIED, split count: 3, time: 23.58
Output dim: 7, lower bound: -0.7751179, upper bound: 0.7725652
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 23.58
Output dim: 7, lower bound: -0.7833717, upper bound: 0.7833741

## BFS IS instance: IS_B1_B1_B2

### Backsubstitution after applying IS history:
0: -8.1358967, -5.3693581, -8.1161938, -5.3859129, -1.7121830, 1.6959538
1: -9.2137871, -6.2088575, -9.2040396, -6.2302504, -2.5258694, 2.4155965
2: -9.9436369, -6.9816589, -9.9211702, -7.0116529, -2.1290641, 2.1561351
3: -10.8232517, -8.2848320, -10.8140755, -8.2984705, -1.6649780, 1.7710907
4: -5.5491905, -3.5314548, -5.5381622, -3.5461254, -1.4490738, 1.4542561
5: -8.8687296, -6.1946063, -8.8562260, -6.2014089, -1.3928547, 1.3867283
6: -12.9627028, -9.7537355, -12.9560547, -9.7603846, -1.7985144, 1.7849884
7: 0.4380250, 2.8263783, 0.4967237, 2.8080168, -1.7187567, 1.7752118
8: -3.7012429, -1.0098205, -3.6762104, -1.0287127, -2.1029763, 2.0255148
9: 0.1652347, 2.2618301, 0.1805308, 2.2544208, -1.9394002, 1.8849497

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B1_B1_B2_A1

### Relational analysis result of IS_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7801781, upper bound: 0.7782533
time: 7.61 seconds

## Relational analysis of IS_B1_B1_B2_A2

### Relational analysis result of IS_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7801781, upper bound: 0.7782554
time: 4.64 seconds

## BFS IS instance: IS_B1_B2_B2

### Backsubstitution after applying IS history:
0: -8.1358967, -5.3693581, -8.1246204, -5.3883009, -1.7155433, 1.7162075
1: -9.2137871, -6.2088575, -9.2057343, -6.2137890, -2.5429134, 2.4148431
2: -9.9436369, -6.9816589, -9.9442692, -7.0135350, -2.1369939, 2.1978669
3: -10.8232517, -8.2848320, -10.8148670, -8.2960634, -1.6637406, 1.7704294
4: -5.5491905, -3.5314548, -5.5371871, -3.5437903, -1.4526029, 1.4579687
5: -8.8687296, -6.1946063, -8.8597441, -6.2016115, -1.3921213, 1.3925667
6: -12.9627028, -9.7537355, -12.9635191, -9.7583609, -1.7978311, 1.7939415
7: 0.4380250, 2.8263783, 0.4830995, 2.8211555, -1.7476645, 1.7773895
8: -3.7012429, -1.0098205, -3.6772470, -1.0235934, -2.1157131, 2.0594063
9: 0.1652347, 2.2618301, 0.1784050, 2.2590675, -1.9441051, 1.9011912

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2153
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B1_B2_B2_A1

### Relational analysis result of IS_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7801410, upper bound: 0.7804133
time: 5.12 seconds

## Relational analysis of IS_B1_B2_B2_A2

### Relational analysis result of IS_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7801410, upper bound: 0.7793814
time: 4.91 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.1540508, -5.3780093, -8.1620913, -5.3680143, -1.7342343, 1.7260566
1: -9.2174072, -6.2218895, -9.2204046, -6.2043295, -2.4529424, 2.5421457
2: -9.9372997, -6.9699330, -9.9490099, -6.9525518, -2.1703401, 2.1743135
3: -10.8287363, -8.2785645, -10.8325090, -8.2740440, -1.8012958, 1.6954229
4: -5.5499477, -3.5180764, -5.5546989, -3.5124776, -1.4646273, 1.4822183
5: -8.8825016, -6.1923246, -8.8870659, -6.1920552, -1.4155784, 1.4041214
6: -12.9633512, -9.7533693, -12.9670238, -9.7500782, -1.8018260, 1.8113976
7: 0.4506269, 2.8163381, 0.4062428, 2.8277011, -1.8229361, 1.7270432
8: -3.6914635, -0.9956088, -3.7070031, -0.9869022, -2.0484886, 2.1395431
9: 0.1722676, 2.2618244, 0.1608398, 2.2655969, -1.8963470, 1.9534354

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7801815
time: 5.06 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823006
time: 4.92 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.1626558, -5.3803940, -8.1620913, -5.3680143, -1.7545042, 1.7294164
1: -9.2191038, -6.2054434, -9.2204046, -6.2043295, -2.4521909, 2.5592375
2: -9.9604149, -6.9718146, -9.9490099, -6.9525518, -2.2121315, 2.1822433
3: -10.8295288, -8.2760334, -10.8325090, -8.2740440, -1.8006353, 1.6942067
4: -5.5489678, -3.5157428, -5.5546989, -3.5124776, -1.4682951, 1.4857483
5: -8.8860235, -6.1925268, -8.8870659, -6.1920552, -1.4214487, 1.4033835
6: -12.9707804, -9.7513466, -12.9670238, -9.7500782, -1.8107986, 1.8109779
7: 0.4369965, 2.8296504, 0.4062428, 2.8277011, -1.8250136, 1.7560048
8: -3.6925054, -0.9904900, -3.7070031, -0.9869022, -2.0824890, 2.1522779
9: 0.1701906, 2.2664695, 0.1608398, 2.2655969, -1.9125628, 1.9580793

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2809

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7823013, upper bound: 0.7801444
time: 4.53 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7823015, upper bound: 0.7801398
time: 5.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.65 seconds
IS_B1_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 24.65
Output dim: 7, lower bound: -0.7801781, upper bound: 0.7782533
IS_B1_B1_B2_A2, status: Status.VERIFIED, split count: 4, time: 24.65
Output dim: 7, lower bound: -0.7801781, upper bound: 0.7782554
IS_B1_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 24.65
Output dim: 7, lower bound: -0.7801410, upper bound: 0.7804133
IS_B1_B2_B2_A2, status: Status.VERIFIED, split count: 4, time: 24.65
Output dim: 7, lower bound: -0.7801410, upper bound: 0.7793814
IS_B2_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 24.65
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7801815
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.65
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823006
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.65
Output dim: 7, lower bound: -0.7823013, upper bound: 0.7801444
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.65
Output dim: 7, lower bound: -0.7823015, upper bound: 0.7801398

## BFS IS instance: IS_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1540508, -5.3780093, -8.1740389, -5.3803868, -1.7264471, 1.7549267
1: -9.2174072, -6.2218895, -9.2274456, -6.2044249, -2.4457226, 2.5309739
2: -9.9372997, -6.9699330, -9.9623280, -6.9684019, -2.1646538, 2.2104654
3: -10.8287363, -8.2785645, -10.8309956, -8.2649622, -1.8066950, 1.6933336
4: -5.5499477, -3.5180764, -5.5542521, -3.5148377, -1.4626598, 1.4826550
5: -8.8825016, -6.1923246, -8.8868237, -6.1921759, -1.4131546, 1.4013569
6: -12.9633512, -9.7533693, -12.9778900, -9.7512283, -1.7955933, 1.8247354
7: 0.4506269, 2.8163381, 0.4355721, 2.8510728, -1.8572841, 1.7029333
8: -3.6914635, -0.9956088, -3.7109237, -0.9894915, -2.0522561, 2.1698179
9: 0.1722676, 2.2618244, 0.1626736, 2.2671447, -1.8761835, 1.9354239

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B2_A1_A2_B2_B1

### Relational analysis result of IS_B2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7719111
time: 4.50 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2

### Relational analysis result of IS_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7823047
time: 5.16 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.1626558, -5.3803940, -8.1657238, -5.3779998, -1.7411165, 1.7422781
1: -9.2191038, -6.2054434, -9.2257586, -6.2208891, -2.4276648, 2.5458646
2: -9.9604149, -6.9718146, -9.9391594, -6.9665422, -2.1918087, 2.1838684
3: -10.8295288, -8.2760334, -10.8301706, -8.2667694, -1.8066125, 1.6923025
4: -5.5489678, -3.5157428, -5.5552292, -3.5171974, -1.4637918, 1.4829202
5: -8.8860235, -6.1925268, -8.8832455, -6.1919756, -1.4169631, 1.3973560
6: -12.9707804, -9.7513466, -12.9709625, -9.7532539, -1.8021932, 1.8174009
7: 0.4369965, 2.8296504, 0.4491997, 2.8368356, -1.8525138, 1.7095938
8: -3.6925054, -0.9904900, -3.7104492, -0.9946251, -2.0662689, 2.1544204
9: 0.1701906, 2.2664695, 0.1650854, 2.2624879, -1.8720775, 1.9377799

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1760
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7688990, upper bound: 0.7697106
time: 4.69 seconds

## Relational analysis of IS_B2_A2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7688966, upper bound: 0.7697123
time: 4.65 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1626558, -5.3803940, -8.1740389, -5.3803868, -1.7353034, 1.7503042
1: -9.2191038, -6.2054434, -9.2274456, -6.2044249, -2.4657960, 2.5506759
2: -9.9604149, -6.9718146, -9.9623280, -6.9684019, -2.1894994, 2.2095675
3: -10.8295288, -8.2760334, -10.8309956, -8.2649622, -1.8061948, 1.6932678
4: -5.5489678, -3.5157428, -5.5542521, -3.5148377, -1.4730444, 1.4885678
5: -8.8860235, -6.1925268, -8.8868237, -6.1921759, -1.4207544, 1.4085314
6: -12.9707804, -9.7513466, -12.9778900, -9.7512283, -1.8032198, 1.8218107
7: 0.4369965, 2.8296504, 0.4355721, 2.8510728, -1.8364267, 1.6908941
8: -3.6925054, -0.9904900, -3.7109237, -0.9894915, -2.0736532, 2.1762681
9: 0.1701906, 2.2664695, 0.1626736, 2.2671447, -1.9038405, 1.9635863

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 18

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7688966, upper bound: 0.7751195
time: 4.68 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7688966, upper bound: 0.7833753
time: 4.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.14 seconds
IS_B2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 24.14
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7719111
IS_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7823047
IS_B2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 24.14
Output dim: 7, lower bound: -0.7688990, upper bound: 0.7697106
IS_B2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 24.14
Output dim: 7, lower bound: -0.7688966, upper bound: 0.7697123
IS_B2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 24.14
Output dim: 7, lower bound: -0.7688966, upper bound: 0.7751195
IS_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 7, lower bound: -0.7688966, upper bound: 0.7833753

## BFS IS instance: IS_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.1540508, -5.3780093, -8.1626539, -5.3803945, -1.7264047, 1.7140536
1: -9.2174072, -6.2218895, -9.2191048, -6.2054424, -2.4382992, 2.3995104
2: -9.9372997, -6.9699330, -9.9604130, -6.9718180, -2.1516171, 2.2085204
3: -10.8287363, -8.2785645, -10.8295269, -8.2760334, -1.6957293, 1.6920168
4: -5.5499477, -3.5180764, -5.5489688, -3.5157428, -1.4614229, 1.4833732
5: -8.8825016, -6.1923246, -8.8860226, -6.1925278, -1.4124770, 1.3996775
6: -12.9633512, -9.7533693, -12.9707794, -9.7513447, -1.7943745, 1.7972679
7: 0.4506269, 2.8163381, 0.4369979, 2.8296485, -1.7411709, 1.7019053
8: -3.6914635, -0.9956088, -3.6925073, -0.9904914, -2.0514765, 2.0838776
9: 0.1722676, 2.2618244, 0.1701918, 2.2664702, -1.8755445, 1.8742976

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7823047
time: 5.11 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2_A2

### Relational analysis result of IS_B2_A1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7801445
time: 5.11 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.1626558, -5.3803940, -8.1626539, -5.3803945, -1.7352624, 1.7082393
1: -9.2191038, -6.2054434, -9.2191048, -6.2054424, -2.4576826, 2.4371147
2: -9.9604149, -6.9718146, -9.9604130, -6.9718180, -2.1772208, 2.2076402
3: -10.8295288, -8.2760334, -10.8295269, -8.2760334, -1.6966681, 1.6919568
4: -5.5489678, -3.5157428, -5.5489688, -3.5157428, -1.4718084, 1.4926062
5: -8.8860235, -6.1925268, -8.8860226, -6.1925278, -1.4237027, 1.4069600
6: -12.9707804, -9.7513466, -12.9707794, -9.7513447, -1.8019395, 1.7982321
7: 0.4369965, 2.8296504, 0.4369979, 2.8296485, -1.7224321, 1.6898842
8: -3.6925054, -0.9904900, -3.6925073, -0.9904914, -2.0728722, 2.0912843
9: 0.1701906, 2.2664695, 0.1701918, 2.2664702, -1.9031677, 1.9060359

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7822997
time: 4.61 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7812441
time: 4.84 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 24.25 seconds
IS_B2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 24.25
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7823047
IS_B2_A1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 24.25
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7801445
IS_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 24.25
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7822997
IS_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 24.25
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7812441

## BFS IS instance: IS_B2_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.1540508, -5.3780093, -8.1626539, -5.3803945, -1.7264047, 1.7140536
1: -9.2174072, -6.2218895, -9.2191048, -6.2054424, -2.4382992, 2.3995104
2: -9.9372997, -6.9699330, -9.9604130, -6.9718180, -2.1516171, 2.2085204
3: -10.8287363, -8.2785645, -10.8295269, -8.2760334, -1.6957293, 1.6920168
4: -5.5499477, -3.5180764, -5.5489688, -3.5157428, -1.4614229, 1.4833732
5: -8.8825016, -6.1923246, -8.8860226, -6.1925278, -1.4124770, 1.3996775
6: -12.9633512, -9.7533693, -12.9707794, -9.7513447, -1.7943745, 1.7972679
7: 0.4506269, 2.8163381, 0.4369979, 2.8296485, -1.7411709, 1.7019053
8: -3.6914635, -0.9956088, -3.6925073, -0.9904914, -2.0514765, 2.0838776
9: 0.1722676, 2.2618244, 0.1701918, 2.2664702, -1.8755445, 1.8742976

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A1_A2_B2_B2_A1_A1

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7681467, upper bound: 0.7688957
time: 4.93 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7681467, upper bound: 0.7823012
time: 6.59 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.1540508, -5.3780093, -8.1626539, -5.3803945, -1.7264047, 1.7140536
1: -9.2174072, -6.2218895, -9.2191048, -6.2054424, -2.4382992, 2.3995104
2: -9.9372997, -6.9699330, -9.9604130, -6.9718180, -2.1516171, 2.2085204
3: -10.8287363, -8.2785645, -10.8295269, -8.2760334, -1.6957293, 1.6920168
4: -5.5499477, -3.5180764, -5.5489688, -3.5157428, -1.4614229, 1.4833732
5: -8.8825016, -6.1923246, -8.8860226, -6.1925278, -1.4124770, 1.3996775
6: -12.9633512, -9.7533693, -12.9707794, -9.7513447, -1.7943745, 1.7972679
7: 0.4506269, 2.8163381, 0.4369979, 2.8296485, -1.7411709, 1.7019053
8: -3.6914635, -0.9956088, -3.6925073, -0.9904914, -2.0514765, 2.0838776
9: 0.1722676, 2.2618244, 0.1701918, 2.2664702, -1.8755445, 1.8742976

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7681467, upper bound: 0.7688956
time: 6.62 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7681467, upper bound: 0.7823012
time: 6.92 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -8.1626558, -5.3803940, -8.1626539, -5.3803945, -1.7352624, 1.7082393
1: -9.2191038, -6.2054434, -9.2191048, -6.2054424, -2.4576826, 2.4371147
2: -9.9604149, -6.9718146, -9.9604130, -6.9718180, -2.1772208, 2.2076402
3: -10.8295288, -8.2760334, -10.8295269, -8.2760334, -1.6966681, 1.6919568
4: -5.5489678, -3.5157428, -5.5489688, -3.5157428, -1.4718084, 1.4926062
5: -8.8860235, -6.1925268, -8.8860226, -6.1925278, -1.4237027, 1.4069600
6: -12.9707804, -9.7513466, -12.9707794, -9.7513447, -1.8019395, 1.7982321
7: 0.4369965, 2.8296504, 0.4369979, 2.8296485, -1.7224321, 1.6898842
8: -3.6925054, -0.9904900, -3.6925073, -0.9904914, -2.0728722, 2.0912843
9: 0.1701906, 2.2664695, 0.1701918, 2.2664702, -1.9031677, 1.9060359

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7725680
time: 4.94 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7833754
time: 4.64 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 24.25 seconds
IS_B2_A1_A2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 24.25
Output dim: 7, lower bound: -0.7681467, upper bound: 0.7688957
IS_B2_A1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 24.25
Output dim: 7, lower bound: -0.7681467, upper bound: 0.7823012
IS_B2_A2_A2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 24.25
Output dim: 7, lower bound: -0.7681467, upper bound: 0.7688956
IS_B2_A2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 24.25
Output dim: 7, lower bound: -0.7681467, upper bound: 0.7823012
IS_B2_A2_A2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 24.25
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7725680
IS_B2_A2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 24.25
Output dim: 7, lower bound: -0.7681490, upper bound: 0.7833754

## BFS IS instance: IS_B2_A1_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.1540508, -5.3780093, -8.1626539, -5.3803945, -1.7264047, 1.7140536
1: -9.2174072, -6.2218895, -9.2191048, -6.2054424, -2.4382992, 2.3995104
2: -9.9372997, -6.9699330, -9.9604130, -6.9718180, -2.1516171, 2.2085204
3: -10.8287363, -8.2785645, -10.8295269, -8.2760334, -1.6957293, 1.6920168
4: -5.5499477, -3.5180764, -5.5489688, -3.5157428, -1.4614229, 1.4833732
5: -8.8825016, -6.1923246, -8.8860226, -6.1925278, -1.4124770, 1.3996775
6: -12.9633512, -9.7533693, -12.9707794, -9.7513447, -1.7943745, 1.7972679
7: 0.4506269, 2.8163381, 0.4369979, 2.8296485, -1.7411709, 1.7019053
8: -3.6914635, -0.9956088, -3.6925073, -0.9904914, -2.0514765, 2.0838776
9: 0.1722676, 2.2618244, 0.1701918, 2.2664702, -1.8755445, 1.8742976

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B2_A1_A2_B2_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7697137, upper bound: 0.7823004
time: 5.03 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7823022
time: 7.44 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.1540508, -5.3780093, -8.1626539, -5.3803945, -1.7264047, 1.7140536
1: -9.2174072, -6.2218895, -9.2191048, -6.2054424, -2.4382992, 2.3995104
2: -9.9372997, -6.9699330, -9.9604130, -6.9718180, -2.1516171, 2.2085204
3: -10.8287363, -8.2785645, -10.8295269, -8.2760334, -1.6957293, 1.6920168
4: -5.5499477, -3.5180764, -5.5489688, -3.5157428, -1.4614229, 1.4833732
5: -8.8825016, -6.1923246, -8.8860226, -6.1925278, -1.4124770, 1.3996775
6: -12.9633512, -9.7533693, -12.9707794, -9.7513447, -1.7943745, 1.7972679
7: 0.4506269, 2.8163381, 0.4369979, 2.8296485, -1.7411709, 1.7019053
8: -3.6914635, -0.9956088, -3.6925073, -0.9904914, -2.0514765, 2.0838776
9: 0.1722676, 2.2618244, 0.1701918, 2.2664702, -1.8755445, 1.8742976

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7697137, upper bound: 0.7823004
time: 5.15 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7823022
time: 7.38 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.1626558, -5.3803940, -8.1626539, -5.3803945, -1.7352624, 1.7082393
1: -9.2191038, -6.2054434, -9.2191048, -6.2054424, -2.4576826, 2.4371147
2: -9.9604149, -6.9718146, -9.9604130, -6.9718180, -2.1772208, 2.2076402
3: -10.8295288, -8.2760334, -10.8295269, -8.2760334, -1.6966681, 1.6919568
4: -5.5489678, -3.5157428, -5.5489688, -3.5157428, -1.4718084, 1.4926062
5: -8.8860235, -6.1925268, -8.8860226, -6.1925278, -1.4237027, 1.4069600
6: -12.9707804, -9.7513466, -12.9707794, -9.7513447, -1.8019395, 1.7982321
7: 0.4369965, 2.8296504, 0.4369979, 2.8296485, -1.7224321, 1.6898842
8: -3.6925054, -0.9904900, -3.6925073, -0.9904914, -2.0728722, 2.0912843
9: 0.1701906, 2.2664695, 0.1701918, 2.2664702, -1.9031677, 1.9060359

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7823022
time: 7.55 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7751179, upper bound: 0.7833712
time: 5.01 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 27.27 seconds
IS_B2_A1_A2_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 27.27
Output dim: 7, lower bound: -0.7697137, upper bound: 0.7823004
IS_B2_A1_A2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 27.27
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7823022
IS_B2_A2_A2_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 27.27
Output dim: 7, lower bound: -0.7697137, upper bound: 0.7823004
IS_B2_A2_A2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 27.27
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7823022
IS_B2_A2_A2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 27.27
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7823022
IS_B2_A2_A2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 27.27
Output dim: 7, lower bound: -0.7751179, upper bound: 0.7833712

## BFS IS instance: IS_B2_A1_A2_B2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -8.1540508, -5.3780093, -8.1626539, -5.3803945, -1.7264047, 1.7140536
1: -9.2174072, -6.2218895, -9.2191048, -6.2054424, -2.4382992, 2.3995104
2: -9.9372997, -6.9699330, -9.9604130, -6.9718180, -2.1516171, 2.2085204
3: -10.8287363, -8.2785645, -10.8295269, -8.2760334, -1.6957293, 1.6920168
4: -5.5499477, -3.5180764, -5.5489688, -3.5157428, -1.4614229, 1.4833732
5: -8.8825016, -6.1923246, -8.8860226, -6.1925278, -1.4124770, 1.3996775
6: -12.9633512, -9.7533693, -12.9707794, -9.7513447, -1.7943745, 1.7972679
7: 0.4506269, 2.8163381, 0.4369979, 2.8296485, -1.7411709, 1.7019053
8: -3.6914635, -0.9956088, -3.6925073, -0.9904914, -2.0514765, 2.0838776
9: 0.1722676, 2.2618244, 0.1701918, 2.2664702, -1.8755445, 1.8742976

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A1_A2_B2_B2_A1_A2_A1_A1

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7688954
time: 4.63 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2_A1_A2_A1_A2

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823006
time: 4.80 seconds

## BFS IS instance: IS_B2_A1_A2_B2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -8.1626558, -5.3803940, -8.1626539, -5.3803945, -1.7352624, 1.7082393
1: -9.2191038, -6.2054434, -9.2191048, -6.2054424, -2.4576826, 2.4371147
2: -9.9604149, -6.9718146, -9.9604130, -6.9718180, -2.1772208, 2.2076402
3: -10.8295288, -8.2760334, -10.8295269, -8.2760334, -1.6966681, 1.6919568
4: -5.5489678, -3.5157428, -5.5489688, -3.5157428, -1.4718084, 1.4926062
5: -8.8860235, -6.1925268, -8.8860226, -6.1925278, -1.4237027, 1.4069600
6: -12.9707804, -9.7513466, -12.9707794, -9.7513447, -1.8019395, 1.7982321
7: 0.4369965, 2.8296504, 0.4369979, 2.8296485, -1.7224321, 1.6898842
8: -3.6925054, -0.9904900, -3.6925073, -0.9904914, -2.0728722, 2.0912843
9: 0.1701906, 2.2664695, 0.1701918, 2.2664702, -1.9031677, 1.9060359

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A1_A2_B2_B2_A1_A2_A2_A1

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7688953
time: 4.60 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2_A1_A2_A2_A2

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823003
time: 4.96 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -8.1540508, -5.3780093, -8.1626539, -5.3803945, -1.7264047, 1.7140536
1: -9.2174072, -6.2218895, -9.2191048, -6.2054424, -2.4382992, 2.3995104
2: -9.9372997, -6.9699330, -9.9604130, -6.9718180, -2.1516171, 2.2085204
3: -10.8287363, -8.2785645, -10.8295269, -8.2760334, -1.6957293, 1.6920168
4: -5.5499477, -3.5180764, -5.5489688, -3.5157428, -1.4614229, 1.4833732
5: -8.8825016, -6.1923246, -8.8860226, -6.1925278, -1.4124770, 1.3996775
6: -12.9633512, -9.7533693, -12.9707794, -9.7513447, -1.7943745, 1.7972679
7: 0.4506269, 2.8163381, 0.4369979, 2.8296485, -1.7411709, 1.7019053
8: -3.6914635, -0.9956088, -3.6925073, -0.9904914, -2.0514765, 2.0838776
9: 0.1722676, 2.2618244, 0.1701918, 2.2664702, -1.8755445, 1.8742976

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2_A1_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7688954
time: 4.69 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2_A1_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823005
time: 4.86 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -8.1626558, -5.3803940, -8.1626539, -5.3803945, -1.7352624, 1.7082393
1: -9.2191038, -6.2054434, -9.2191048, -6.2054424, -2.4576826, 2.4371147
2: -9.9604149, -6.9718146, -9.9604130, -6.9718180, -2.1772208, 2.2076402
3: -10.8295288, -8.2760334, -10.8295269, -8.2760334, -1.6966681, 1.6919568
4: -5.5489678, -3.5157428, -5.5489688, -3.5157428, -1.4718084, 1.4926062
5: -8.8860235, -6.1925268, -8.8860226, -6.1925278, -1.4237027, 1.4069600
6: -12.9707804, -9.7513466, -12.9707794, -9.7513447, -1.8019395, 1.7982321
7: 0.4369965, 2.8296504, 0.4369979, 2.8296485, -1.7224321, 1.6898842
8: -3.6925054, -0.9904900, -3.6925073, -0.9904914, -2.0728722, 2.0912843
9: 0.1701906, 2.2664695, 0.1701918, 2.2664702, -1.9031677, 1.9060359

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7688953
time: 4.47 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823003
time: 4.81 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -8.1540508, -5.3780093, -8.1626539, -5.3803945, -1.7264047, 1.7140536
1: -9.2174072, -6.2218895, -9.2191048, -6.2054424, -2.4382992, 2.3995104
2: -9.9372997, -6.9699330, -9.9604130, -6.9718180, -2.1516171, 2.2085204
3: -10.8287363, -8.2785645, -10.8295269, -8.2760334, -1.6957293, 1.6920168
4: -5.5499477, -3.5180764, -5.5489688, -3.5157428, -1.4614229, 1.4833732
5: -8.8825016, -6.1923246, -8.8860226, -6.1925278, -1.4124770, 1.3996775
6: -12.9633512, -9.7533693, -12.9707794, -9.7513447, -1.7943745, 1.7972679
7: 0.4506269, 2.8163381, 0.4369979, 2.8296485, -1.7411709, 1.7019053
8: -3.6914635, -0.9956088, -3.6925073, -0.9904914, -2.0514765, 2.0838776
9: 0.1722676, 2.2618244, 0.1701918, 2.2664702, -1.8755445, 1.8742976

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2_A1_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7688954
time: 4.66 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2_A1_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823006
time: 4.76 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -8.1626558, -5.3803940, -8.1626539, -5.3803945, -1.7352624, 1.7082393
1: -9.2191038, -6.2054434, -9.2191048, -6.2054424, -2.4576826, 2.4371147
2: -9.9604149, -6.9718146, -9.9604130, -6.9718180, -2.1772208, 2.2076402
3: -10.8295288, -8.2760334, -10.8295269, -8.2760334, -1.6966681, 1.6919568
4: -5.5489678, -3.5157428, -5.5489688, -3.5157428, -1.4718084, 1.4926062
5: -8.8860235, -6.1925268, -8.8860226, -6.1925278, -1.4237027, 1.4069600
6: -12.9707804, -9.7513466, -12.9707794, -9.7513447, -1.8019395, 1.7982321
7: 0.4369965, 2.8296504, 0.4369979, 2.8296485, -1.7224321, 1.6898842
8: -3.6925054, -0.9904900, -3.6925073, -0.9904914, -2.0728722, 2.0912843
9: 0.1701906, 2.2664695, 0.1701918, 2.2664702, -1.9031677, 1.9060359

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7725648
time: 4.91 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823047
time: 5.04 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 24.59 seconds
IS_B2_A1_A2_B2_B2_A1_A2_A1_A1, status: Status.VERIFIED, split count: 9, time: 24.59
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7688954
IS_B2_A1_A2_B2_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 24.59
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823006
IS_B2_A1_A2_B2_B2_A1_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 24.59
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7688953
IS_B2_A1_A2_B2_B2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 24.59
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823003
IS_B2_A2_A2_B2_B2_A1_A2_A1_A1, status: Status.VERIFIED, split count: 9, time: 24.59
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7688954
IS_B2_A2_A2_B2_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 24.59
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823005
IS_B2_A2_A2_B2_B2_A1_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 24.59
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7688953
IS_B2_A2_A2_B2_B2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 24.59
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823003
IS_B2_A2_A2_B2_B2_A2_A2_A1_A1, status: Status.VERIFIED, split count: 9, time: 24.59
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7688954
IS_B2_A2_A2_B2_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 24.59
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823006
IS_B2_A2_A2_B2_B2_A2_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 24.59
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7725648
IS_B2_A2_A2_B2_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 24.59
Output dim: 7, lower bound: -0.7801411, upper bound: 0.7823047

## BFS IS instance: IS_B2_A1_A2_B2_B2_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -8.1540508, -5.3780093, -8.1626539, -5.3803945, -1.7264047, 1.7140536
1: -9.2174072, -6.2218895, -9.2191048, -6.2054424, -2.4382992, 2.3995104
2: -9.9372997, -6.9699330, -9.9604130, -6.9718180, -2.1516171, 2.2085204
3: -10.8287363, -8.2785645, -10.8295269, -8.2760334, -1.6957293, 1.6920168
4: -5.5499477, -3.5180764, -5.5489688, -3.5157428, -1.4614229, 1.4833732
5: -8.8825016, -6.1923246, -8.8860226, -6.1925278, -1.4124770, 1.3996775
6: -12.9633512, -9.7533693, -12.9707794, -9.7513447, -1.7943745, 1.7972679
7: 0.4506269, 2.8163381, 0.4369979, 2.8296485, -1.7411709, 1.7019053
8: -3.6914635, -0.9956088, -3.6925073, -0.9904914, -2.0514765, 2.0838776
9: 0.1722676, 2.2618244, 0.1701918, 2.2664702, -1.8755445, 1.8742976

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2482
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B2_A1_A2_B2_B2_A1_A2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7697137, upper bound: 0.7823000
time: 5.11 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2_A1_A2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7823020
time: 7.24 seconds

## BFS IS instance: IS_B2_A1_A2_B2_B2_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -8.1626558, -5.3803940, -8.1626539, -5.3803945, -1.7352624, 1.7082393
1: -9.2191038, -6.2054434, -9.2191048, -6.2054424, -2.4576826, 2.4371147
2: -9.9604149, -6.9718146, -9.9604130, -6.9718180, -2.1772208, 2.2076402
3: -10.8295288, -8.2760334, -10.8295269, -8.2760334, -1.6966681, 1.6919568
4: -5.5489678, -3.5157428, -5.5489688, -3.5157428, -1.4718084, 1.4926062
5: -8.8860235, -6.1925268, -8.8860226, -6.1925278, -1.4237027, 1.4069600
6: -12.9707804, -9.7513466, -12.9707794, -9.7513447, -1.8019395, 1.7982321
7: 0.4369965, 2.8296504, 0.4369979, 2.8296485, -1.7224321, 1.6898842
8: -3.6925054, -0.9904900, -3.6925073, -0.9904914, -2.0728722, 2.0912843
9: 0.1701906, 2.2664695, 0.1701918, 2.2664702, -1.9031677, 1.9060359

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2153
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1949
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1949
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1760
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2482
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 2809
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 2809
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1760

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_B2_A1_A2_B2_B2_A1_A2_A2_A2_A1

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7823020
time: 7.26 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2_A1_A2_A2_A2_A2

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7697114, upper bound: 0.7823022
time: 6.41 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -8.1540508, -5.3780093, -8.1626539, -5.3803945, -1.7264047, 1.7140536
1: -9.2174072, -6.2218895, -9.2191048, -6.2054424, -2.4382992, 2.3995104
2: -9.9372997, -6.9699330, -9.9604130, -6.9718180, -2.1516171, 2.2085204
3: -10.8287363, -8.2785645, -10.8295269, -8.2760334, -1.6957293, 1.6920168
4: -5.5499477, -3.5180764, -5.5489688, -3.5157428, -1.4614229, 1.4833732
5: -8.8825016, -6.1923246, -8.8860226, -6.1925278, -1.4124770, 1.3996775
6: -12.9633512, -9.7533693, -12.9707794, -9.7513447, -1.7943745, 1.7972679
7: 0.4506269, 2.8163381, 0.4369979, 2.8296485, -1.7411709, 1.7019053
8: -3.6914635, -0.9956088, -3.6925073, -0.9904914, -2.0514765, 2.0838776
9: 0.1722676, 2.2618244, 0.1701918, 2.2664702, -1.8755445, 1.8742976

Time for backsubstitution: 14.44 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.8785624504089355
rel_dist={7: [-0.7927072900922123, 0.7927031607168487]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 2410.71 seconds
