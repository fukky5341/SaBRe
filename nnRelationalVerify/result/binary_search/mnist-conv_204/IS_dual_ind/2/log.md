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
execution time: IAR + LP analysis = 15.08 + 32.26 = 47.34 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.66 seconds, max iter: 100)

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
Binary search time: 147.13 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 3405.53 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3603419, upper bound: 1.3704547
time: 3.94 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3704540, upper bound: 1.3704541
time: 3.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.85 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.85
Output dim: 7, lower bound: -1.3603419, upper bound: 1.3704547
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.85
Output dim: 7, lower bound: -1.3704540, upper bound: 1.3704541

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.1321526, -5.3759112, -8.1611385, -5.3684554, -2.3022795, 2.3172426
1: -9.2126160, -6.2120185, -9.2237587, -6.2050738, -3.0075421, 3.0117402
2: -9.9340887, -6.9920025, -9.9485035, -6.9602008, -2.6226640, 2.6079078
3: -10.8188086, -8.2860575, -10.8303308, -8.2697258, -2.1705294, 2.1637528
4: -5.5464420, -3.5399156, -5.5563612, -3.5183458, -1.8611650, 1.8507237
5: -8.8613005, -6.2009053, -8.8813400, -6.1926470, -1.8811326, 1.8919771
6: -12.9650011, -9.7570105, -12.9708843, -9.7512064, -2.4262557, 2.4267941
7: 0.4514637, 2.8336329, 0.4160919, 2.8416376, -2.1825786, 2.2085896
8: -3.7049961, -1.0193210, -3.7183290, -0.9940157, -2.5953140, 2.5891223
9: 0.1637008, 2.2586241, 0.1569617, 2.2647943, -2.1010935, 2.1016624

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3603403, upper bound: 1.3603409
time: 3.88 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3603403, upper bound: 1.3704547
time: 3.86 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.1700544, -5.3680086, -8.1700583, -5.3680067, -2.3221960, 2.3390398
1: -9.2260036, -6.2035923, -9.2260056, -6.2035913, -3.0224123, 3.0224133
2: -9.9503212, -6.9502831, -9.9503212, -6.9502754, -2.6494493, 2.6256390
3: -10.8334780, -8.2661514, -10.8334827, -8.2661476, -2.1891727, 2.1944654
4: -5.5582314, -3.5118756, -5.5582318, -3.5118723, -1.8800120, 1.8637280
5: -8.8875713, -6.1918206, -8.8875761, -6.1918221, -1.8949375, 1.9080455
6: -12.9723415, -9.7499933, -12.9723425, -9.7499943, -2.4364748, 2.4407964
7: 0.4052863, 2.8421242, 0.4052801, 2.8421252, -2.2031622, 2.2286510
8: -3.7202139, -0.9862318, -3.7202172, -0.9862285, -2.6218419, 2.6133051
9: 0.1555148, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3704541, upper bound: 1.3603403
time: 3.95 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3704541, upper bound: 1.3704541
time: 3.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.59 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.59
Output dim: 7, lower bound: -1.3603403, upper bound: 1.3603409
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.59
Output dim: 7, lower bound: -1.3603403, upper bound: 1.3704547
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.59
Output dim: 7, lower bound: -1.3704541, upper bound: 1.3603403
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.59
Output dim: 7, lower bound: -1.3704541, upper bound: 1.3704541

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.1321526, -5.3759112, -8.1321526, -5.3759112, -2.2924886, 2.2924886
1: -9.2126160, -6.2120185, -9.2126160, -6.2120185, -3.0005975, 3.0005975
2: -9.9340887, -6.9920025, -9.9340887, -6.9920025, -2.5923147, 2.5923147
3: -10.8188086, -8.2860575, -10.8188086, -8.2860575, -2.1530113, 2.1530111
4: -5.5464420, -3.5399156, -5.5464420, -3.5399156, -1.8401213, 1.8401213
5: -8.8613005, -6.2009053, -8.8613005, -6.2009053, -1.8721457, 1.8721457
6: -12.9650011, -9.7570105, -12.9650011, -9.7570105, -2.4191689, 2.4191685
7: 0.4514637, 2.8336329, 0.4514637, 2.8336329, -2.1737628, 2.1737633
8: -3.7049961, -1.0193210, -3.7049961, -1.0193210, -2.5748315, 2.5748315
9: 0.1637008, 2.2586241, 0.1637008, 2.2586241, -2.0949233, 2.0949233

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3051355, upper bound: 1.2914261
time: 3.37 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3378969, upper bound: 1.3378948
time: 3.84 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.1321526, -5.3759112, -8.1700544, -5.3680086, -2.3024945, 2.3290277
1: -9.2126160, -6.2120185, -9.2260036, -6.2035923, -3.0090237, 3.0139852
2: -9.9340887, -6.9920025, -9.9503212, -6.9502831, -2.6319842, 2.6097777
3: -10.8188086, -8.2860575, -10.8334780, -8.2661514, -2.1735907, 2.1679692
4: -5.5464420, -3.5399156, -5.5582314, -3.5118756, -1.8674531, 1.8526793
5: -8.8613005, -6.2009053, -8.8875713, -6.1918206, -1.8820643, 1.8981249
6: -12.9650011, -9.7570105, -12.9723415, -9.7499933, -2.4267941, 2.4281564
7: 0.4514637, 2.8336329, 0.4052863, 2.8421242, -2.1832519, 2.2153645
8: -3.7049961, -1.0193210, -3.7202139, -0.9862318, -2.6067276, 2.5899377
9: 0.1637008, 2.2586241, 0.1555148, 2.2660573, -2.1023564, 2.1031094

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3051355, upper bound: 1.3007508
time: 3.54 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3378969, upper bound: 1.3479612
time: 3.97 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.1700544, -5.3680086, -8.1321526, -5.3759112, -2.3290277, 2.3024945
1: -9.2260036, -6.2035923, -9.2126160, -6.2120185, -3.0139852, 3.0090237
2: -9.9503212, -6.9502831, -9.9340887, -6.9920025, -2.6097775, 2.6319847
3: -10.8334780, -8.2661514, -10.8188086, -8.2860575, -2.1679688, 2.1735899
4: -5.5582314, -3.5118756, -5.5464420, -3.5399156, -1.8526793, 1.8674526
5: -8.8875713, -6.1918206, -8.8613005, -6.2009053, -1.8981252, 1.8820646
6: -12.9723415, -9.7499933, -12.9650011, -9.7570105, -2.4281564, 2.4267941
7: 0.4052863, 2.8421242, 0.4514637, 2.8336329, -2.2153645, 2.1832523
8: -3.7202139, -0.9862318, -3.7049961, -1.0193210, -2.5899377, 2.6067276
9: 0.1555148, 2.2660573, 0.1637008, 2.2586241, -2.1031094, 2.1023564

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3147522, upper bound: 1.2914263
time: 3.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3479609, upper bound: 1.3378946
time: 3.81 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.1700544, -5.3680086, -8.1700544, -5.3680086, -2.3221946, 2.3221951
1: -9.2260036, -6.2035923, -9.2260036, -6.2035923, -3.0224113, 3.0224113
2: -9.9503212, -6.9502831, -9.9503212, -6.9502831, -2.6256371, 2.6256371
3: -10.8334780, -8.2661514, -10.8334780, -8.2661514, -2.1944637, 2.1944637
4: -5.5582314, -3.5118756, -5.5582314, -3.5118756, -1.8637266, 1.8637266
5: -8.8875713, -6.1918206, -8.8875713, -6.1918206, -1.8949370, 1.8949373
6: -12.9723415, -9.7499933, -12.9723415, -9.7499933, -2.4407949, 2.4407954
7: 0.4052863, 2.8421242, 0.4052863, 2.8421242, -2.2031622, 2.2031622
8: -3.7202139, -0.9862318, -3.7202139, -0.9862318, -2.6133022, 2.6133022
9: 0.1555148, 2.2660573, 0.1555148, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3147528, upper bound: 1.2914263
time: 3.26 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3479616, upper bound: 1.3382181
time: 3.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.73 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.73
Output dim: 7, lower bound: -1.3051355, upper bound: 1.2914261
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.73
Output dim: 7, lower bound: -1.3378969, upper bound: 1.3378948
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.73
Output dim: 7, lower bound: -1.3051355, upper bound: 1.3007508
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.73
Output dim: 7, lower bound: -1.3378969, upper bound: 1.3479612
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.73
Output dim: 7, lower bound: -1.3147522, upper bound: 1.2914263
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.73
Output dim: 7, lower bound: -1.3479609, upper bound: 1.3378946
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.73
Output dim: 7, lower bound: -1.3147528, upper bound: 1.2914263
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.73
Output dim: 7, lower bound: -1.3479616, upper bound: 1.3382181

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.0774212, -5.3410282, -8.1169834, -5.3761945, -2.2287521, 2.3011217
1: -9.1012020, -6.2079577, -9.1863155, -6.2166510, -2.8845510, 2.9734106
2: -9.8916731, -7.0308886, -9.9244270, -7.0008945, -2.5532355, 2.5555501
3: -10.8603325, -8.3695946, -10.8141279, -8.3104239, -2.1252689, 2.0374672
4: -5.5002260, -3.5368199, -5.5358391, -3.5422475, -1.8054180, 1.8513360
5: -8.8562021, -6.1959600, -8.8579979, -6.2011814, -1.8645263, 1.8688498
6: -12.9008265, -9.7182255, -12.9497137, -9.7572651, -2.3428593, 2.4184172
7: 0.3875179, 2.7263870, 0.4549294, 2.8082418, -2.1514058, 2.0240641
8: -3.6383677, -0.9676361, -3.6880112, -1.0243864, -2.4327779, 2.5259671
9: 0.2698810, 2.2579274, 0.1884406, 2.2560494, -1.9861684, 2.0694869

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2564418
time: 3.42 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2815945, upper bound: 1.2646406
time: 3.54 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1321526, -5.3759112, -2.2529054, 2.2924476
1: -9.2042723, -6.2131233, -9.2126160, -6.2120185, -2.9371018, 2.9994926
2: -9.9321079, -6.9953933, -9.9340887, -6.9920025, -2.5903878, 2.5825586
3: -10.8173695, -8.2978544, -10.8188086, -8.2860575, -2.1517677, 2.0464244
4: -5.5411634, -3.5408163, -5.5464420, -3.5399156, -1.8552942, 1.8388624
5: -8.8605442, -6.2012568, -8.8613005, -6.2009053, -1.8704686, 1.8740385
6: -12.9573345, -9.7571383, -12.9650011, -9.7570105, -2.3973188, 2.4156036
7: 0.4528751, 2.8134508, 0.4514637, 2.8336329, -2.1728077, 2.0506411
8: -3.6860209, -1.0203171, -3.7049961, -1.0193210, -2.4690247, 2.5740428
9: 0.1713096, 2.2579260, 0.1637008, 2.2586241, -2.0873146, 2.0942252

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2914278, upper bound: 1.3051362
time: 4.00 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2914278, upper bound: 1.3378968
time: 3.90 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.0774212, -5.3410282, -8.1548538, -5.3682842, -2.2387619, 2.3376036
1: -9.1012020, -6.2079577, -9.1998711, -6.2081742, -2.8930278, 2.9889274
2: -9.8916731, -7.0308886, -9.9408169, -6.9591765, -2.5929041, 2.5731409
3: -10.8603325, -8.3695946, -10.8287354, -8.2905121, -2.1458063, 2.0523422
4: -5.5002260, -3.5368199, -5.5476270, -3.5142245, -1.8327250, 1.8637180
5: -8.8562021, -6.1959600, -8.8842621, -6.1920948, -1.8744454, 1.8947973
6: -12.9008265, -9.7182255, -12.9571075, -9.7502499, -2.3504848, 2.4273160
7: 0.3875179, 2.7263870, 0.4087830, 2.8167381, -2.1608891, 2.0664885
8: -3.6383677, -0.9676361, -3.7030644, -0.9913211, -2.4646158, 2.5410850
9: 0.2698810, 2.2579274, 0.1802540, 2.2635264, -1.9936454, 2.0776734

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2657664
time: 3.49 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2815945, upper bound: 1.2739639
time: 3.35 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1700544, -5.3680086, -2.2629471, 2.3289866
1: -9.2042723, -6.2131233, -9.2260036, -6.2035923, -2.9461946, 3.0128803
2: -9.9321079, -6.9953933, -9.9503212, -6.9502831, -2.6300573, 2.6000962
3: -10.8173695, -8.2978544, -10.8334780, -8.2661514, -2.1723461, 2.0614457
4: -5.5411634, -3.5408163, -5.5582314, -3.5118756, -1.8830323, 1.8514204
5: -8.8605442, -6.2012568, -8.8875713, -6.1918206, -1.8803873, 1.9000700
6: -12.9573345, -9.7571383, -12.9723415, -9.7499933, -2.4065366, 2.4245911
7: 0.4528751, 2.8134508, 0.4052863, 2.8421242, -2.1822972, 2.0941272
8: -3.6860209, -1.0203171, -3.7202139, -0.9862318, -2.5009522, 2.5891490
9: 0.1713096, 2.2579260, 0.1555148, 2.2660573, -2.0947475, 2.1024113

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2914279, upper bound: 1.3147533
time: 3.42 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2914279, upper bound: 1.3479634
time: 3.53 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.1148834, -5.3330421, -8.1169834, -5.3761945, -2.2650752, 2.3111718
1: -9.1146488, -6.1993079, -9.1863155, -6.2166510, -2.8979979, 2.9826498
2: -9.9084682, -6.9891806, -9.9244270, -7.0008945, -2.5713491, 2.5951548
3: -10.8748140, -8.3504868, -10.8141279, -8.3104239, -2.1403275, 2.0570674
4: -5.5119295, -3.5088513, -5.5358391, -3.5422475, -1.8178024, 1.8789415
5: -8.8824034, -6.1868315, -8.8579979, -6.2011814, -1.8903971, 1.8788049
6: -12.9083214, -9.7108307, -12.9497137, -9.7572651, -2.3516774, 2.4273098
7: 0.3405704, 2.7348576, 0.4549294, 2.8082418, -2.1945252, 2.0334921
8: -3.6534615, -0.9343452, -3.6880112, -1.0243864, -2.4479594, 2.5578132
9: 0.2617226, 2.2655325, 0.1884406, 2.2560494, -1.9943268, 2.0770919

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2888456, upper bound: 1.2564413
time: 3.65 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2912188, upper bound: 1.2646405
time: 3.46 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.1583109, -5.3680158, -8.1321526, -5.3759112, -2.2893891, 2.3024507
1: -9.2176418, -6.2046900, -9.2126160, -6.2120185, -2.9521003, 3.0079260
2: -9.9483604, -6.9536762, -9.9340887, -6.9920025, -2.6078672, 2.6221752
3: -10.8320303, -8.2779474, -10.8188086, -8.2860575, -2.1667128, 2.0660853
4: -5.5529518, -3.5127769, -5.5464420, -3.5399156, -1.8675709, 1.8661695
5: -8.8868132, -6.1921716, -8.8613005, -6.2009053, -1.8964305, 1.8839493
6: -12.9646530, -9.7501202, -12.9650011, -9.7570105, -2.4060197, 2.4232292
7: 0.4067149, 2.8217978, 0.4514637, 2.8336329, -2.2143998, 2.0600564
8: -3.7013464, -0.9872322, -3.7049961, -1.0193210, -2.4840832, 2.6059294
9: 0.1630850, 2.2653689, 0.1637008, 2.2586241, -2.0955391, 2.1016681

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3007504, upper bound: 1.3051362
time: 3.52 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3007504, upper bound: 1.3378965
time: 4.33 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.1148834, -5.3330421, -8.1548538, -5.3682842, -2.2584548, 2.3307881
1: -9.1146488, -6.1993079, -9.1998711, -6.2081742, -2.9064746, 3.0005631
2: -9.9084682, -6.9891806, -9.9408169, -6.9591765, -2.5872011, 2.5889354
3: -10.8748140, -8.3504868, -10.8287354, -8.2905121, -2.1667771, 2.0778592
4: -5.5119295, -3.5088513, -5.5476270, -3.5142245, -1.8288283, 1.8750439
5: -8.8824034, -6.1868315, -8.8842621, -6.1920948, -1.8872104, 1.8916457
6: -12.9083214, -9.7108307, -12.9571075, -9.7502499, -2.3640027, 2.4411345
7: 0.3405704, 2.7348576, 0.4087830, 2.8167381, -2.1819658, 2.0533323
8: -3.6534615, -0.9343452, -3.7030644, -0.9913211, -2.4712133, 2.5648386
9: 0.2617226, 2.2655325, 0.1802540, 2.2635264, -2.0018039, 2.0852785

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2888463, upper bound: 1.2564398
time: 3.36 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2912194, upper bound: 1.2646389
time: 3.20 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.1583109, -5.3680158, -8.1700544, -5.3680086, -2.2826066, 2.3221502
1: -9.2176418, -6.2046900, -9.2260036, -6.2035923, -2.9908991, 3.0213137
2: -9.9483604, -6.9536762, -9.9503212, -6.9502831, -2.6237268, 2.6159015
3: -10.8320303, -8.2779474, -10.8334780, -8.2661514, -2.1932073, 2.0870221
4: -5.5529518, -3.5127769, -5.5582314, -3.5118756, -1.8790317, 1.8624430
5: -8.8868132, -6.1921716, -8.8875713, -6.1918206, -1.8932424, 1.8968732
6: -12.9646530, -9.7501202, -12.9723415, -9.7499933, -2.4202423, 2.4372306
7: 0.4067149, 2.8217978, 0.4052863, 2.8421242, -2.2021518, 2.0806124
8: -3.7013464, -0.9872322, -3.7202139, -0.9862318, -2.5074773, 2.6125226
9: 0.1630850, 2.2653689, 0.1555148, 2.2660573, -2.1029723, 2.1098542

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3052355
time: 3.61 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3382195
time: 3.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.04 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2564418
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.2815945, upper bound: 1.2646406
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.2914278, upper bound: 1.3051362
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.2914278, upper bound: 1.3378968
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2657664
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.2815945, upper bound: 1.2739639
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.2914279, upper bound: 1.3147533
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.2914279, upper bound: 1.3479634
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.2888456, upper bound: 1.2564413
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.2912188, upper bound: 1.2646405
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.3007504, upper bound: 1.3051362
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.3007504, upper bound: 1.3378965
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.2888463, upper bound: 1.2564398
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.2912194, upper bound: 1.2646389
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3052355
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 7, lower bound: -1.3007510, upper bound: 1.3382195

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.0768328, -5.3422565, -8.1278324, -5.3859034, -2.2118258, 2.3128154
1: -9.1011686, -6.2102585, -9.2123814, -6.2292414, -2.8719273, 2.9904084
2: -9.8902674, -7.0331473, -9.9230490, -7.0082626, -2.5355372, 2.5537300
3: -10.8598995, -8.3696833, -10.8154984, -8.2866716, -2.1528201, 2.0376415
4: -5.4997187, -3.5375350, -5.5434432, -3.5452466, -1.8004947, 1.8501730
5: -8.8555584, -6.1959839, -8.8569717, -6.2010608, -1.8619442, 1.8646648
6: -12.9006338, -9.7186251, -12.9636412, -9.7602673, -2.3318481, 2.4354947
7: 0.3928185, 2.7256117, 0.4953136, 2.8283672, -2.1787205, 1.9764500
8: -3.6368237, -0.9684634, -3.6953039, -1.0277319, -2.4189701, 2.5501380
9: 0.2712268, 2.2574723, 0.1732543, 2.2550933, -1.9838666, 2.0842180

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2564418
time: 3.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2564418
time: 3.54 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.0771332, -5.3429375, -8.1359806, -5.3882895, -2.2103510, 2.3376100
1: -9.1011820, -6.2086754, -9.2140770, -6.2127647, -2.8884172, 2.9945021
2: -9.8913822, -7.0334311, -9.9462013, -7.0101213, -2.5392337, 2.5980258
3: -10.8600245, -8.3696289, -10.8163214, -8.2848644, -2.1536994, 2.0378673
4: -5.4996586, -3.5372024, -5.5424709, -3.5428865, -1.8059235, 1.8521013
5: -8.8558197, -6.1959939, -8.8605442, -6.2012596, -1.8612642, 1.8717623
6: -12.9007721, -9.7184896, -12.9705677, -9.7582445, -2.3296928, 2.4458234
7: 0.3929071, 2.7260880, 0.4816909, 2.8425953, -2.2148590, 1.9748592
8: -3.6379929, -0.9700279, -3.6956553, -1.0225992, -2.4265079, 2.5847178
9: 0.2715170, 2.2578671, 0.1708477, 2.2597508, -1.9882338, 2.0870194

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2815945, upper bound: 1.2646406
time: 3.45 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2815945, upper bound: 1.2646406
time: 3.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.0774212, -5.3410282, -2.3061643, 2.2291207
1: -9.2042723, -6.2131233, -9.1012020, -6.2079577, -2.9963145, 2.8880787
2: -9.9321079, -6.9953933, -9.8916731, -7.0308886, -2.5641170, 2.5552659
3: -10.8173695, -8.2978544, -10.8603325, -8.3695946, -2.0398889, 2.1464295
4: -5.5411634, -3.5408163, -5.5002260, -3.5368199, -1.8491797, 1.8070250
5: -8.8605442, -6.2012568, -8.8562021, -6.1959600, -1.8709807, 1.8632956
6: -12.9573345, -9.7571383, -12.9008265, -9.7182255, -2.4300733, 2.3404021
7: 0.4528751, 2.8134508, 0.3875179, 2.7263870, -2.0257502, 2.1813703
8: -3.6860209, -1.0203171, -3.6383677, -0.9676361, -2.5538683, 2.4353459
9: 0.1713096, 2.2579260, 0.2698810, 2.2579274, -2.0866179, 1.9880450

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2564412, upper bound: 1.2792189
time: 3.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2646397, upper bound: 1.2815943
time: 3.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1204453, -5.3759203, -2.2528687, 2.2528687
1: -9.2042723, -6.2131233, -9.2042723, -6.2131233, -2.9293418, 2.9293418
2: -9.9321079, -6.9953933, -9.9321079, -6.9953933, -2.5806146, 2.5806146
3: -10.8173695, -8.2978544, -10.8173695, -8.2978544, -2.0451469, 2.0451472
4: -5.5411634, -3.5408163, -5.5411634, -3.5408163, -1.8540955, 1.8540950
5: -8.8605442, -6.2012568, -8.8605442, -6.2012568, -1.8724160, 1.8724160
6: -12.9573345, -9.7571383, -12.9573345, -9.7571383, -2.3960986, 2.3960989
7: 0.4528751, 2.8134508, 0.4528751, 2.8134508, -2.0496912, 2.0496910
8: -3.6860209, -1.0203171, -3.6860209, -1.0203171, -2.4682360, 2.4682362
9: 0.1713096, 2.2579260, 0.1713096, 2.2579260, -2.0866165, 2.0866165

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3051355, upper bound: 1.2914279
time: 3.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3378969, upper bound: 1.3378964
time: 3.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.0768328, -5.3422565, -8.1657238, -5.3779998, -2.2218328, 2.3493378
1: -9.1011686, -6.2102585, -9.2257586, -6.2208891, -2.8802795, 3.0056591
2: -9.8902674, -7.0331473, -9.9391594, -6.9665422, -2.5752058, 2.5711222
3: -10.8598995, -8.3696833, -10.8301706, -8.2667694, -2.1734047, 2.0525944
4: -5.4997187, -3.5375350, -5.5552292, -3.5171974, -1.8278451, 1.8627501
5: -8.8555584, -6.1959839, -8.8832455, -6.1919756, -1.8718729, 1.8906367
6: -12.9006338, -9.7186251, -12.9709625, -9.7532539, -2.3394685, 2.4444809
7: 0.3928185, 2.7256117, 0.4491997, 2.8368356, -2.1881723, 2.0191159
8: -3.6368237, -0.9684634, -3.7104492, -0.9946251, -2.4508839, 2.5651526
9: 0.2712268, 2.2574723, 0.1650854, 2.2624879, -1.9912611, 2.0923867

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2657664
time: 3.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2657664
time: 3.47 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.0771332, -5.3429375, -8.1740389, -5.3803868, -2.2203584, 2.3741374
1: -9.1011820, -6.2086754, -9.2274456, -6.2044249, -2.8967571, 3.0097547
2: -9.8913822, -7.0334311, -9.9623280, -6.9684019, -2.5789032, 2.6154785
3: -10.8600245, -8.3696289, -10.8309956, -8.2649622, -2.1742873, 2.0528228
4: -5.4996586, -3.5372024, -5.5542521, -3.5148377, -1.8332739, 1.8646569
5: -8.8558197, -6.1959939, -8.8868237, -6.1921759, -1.8711905, 1.8977659
6: -12.9007721, -9.7184896, -12.9778900, -9.7512283, -2.3375778, 2.4548070
7: 0.3929071, 2.7260880, 0.4355721, 2.8510728, -2.2243347, 2.0178957
8: -3.6379929, -0.9700279, -3.7109237, -0.9894915, -2.4584188, 2.5997715
9: 0.2715170, 2.2578671, 0.1626736, 2.2671447, -1.9956276, 2.0951934

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2815945, upper bound: 1.2739639
time: 3.20 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2815945, upper bound: 1.2739639
time: 3.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1148834, -5.3330421, -2.3162146, 2.2654438
1: -9.2042723, -6.2131233, -9.1146488, -6.1993079, -3.0049644, 2.9015255
2: -9.9321079, -6.9953933, -9.9084682, -6.9891806, -2.6037216, 2.5733790
3: -10.8173695, -8.2978544, -10.8748140, -8.3504868, -2.0594893, 2.1614873
4: -5.5411634, -3.5408163, -5.5119295, -3.5088513, -1.8767853, 1.8194098
5: -8.8605442, -6.2012568, -8.8824034, -6.1868315, -1.8809361, 1.8891664
6: -12.9573345, -9.7571383, -12.9083214, -9.7108307, -2.4389658, 2.3492203
7: 0.4528751, 2.8134508, 0.3405704, 2.7348576, -2.0351787, 2.2245669
8: -3.6860209, -1.0203171, -3.6534615, -0.9343452, -2.5857143, 2.4505274
9: 0.1713096, 2.2579260, 0.2617226, 2.2655325, -2.0942230, 1.9962034

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2564413, upper bound: 1.2888462
time: 3.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2646398, upper bound: 1.2912192
time: 3.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1583109, -5.3680158, -2.2629075, 2.2893524
1: -9.2042723, -6.2131233, -9.2176418, -6.2046900, -2.9384146, 2.9443398
2: -9.9321079, -6.9953933, -9.9483604, -6.9536762, -2.6202312, 2.5981691
3: -10.8173695, -8.2978544, -10.8320303, -8.2779474, -2.0648079, 2.0601556
4: -5.5411634, -3.5408163, -5.5529518, -3.5127769, -1.8818226, 1.8663721
5: -8.8605442, -6.2012568, -8.8868132, -6.1921716, -1.8823271, 1.8984308
6: -12.9573345, -9.7571383, -12.9646530, -9.7501202, -2.4053164, 2.4047995
7: 0.4528751, 2.8134508, 0.4067149, 2.8217978, -2.0591068, 2.0931454
8: -3.6860209, -1.0203171, -3.7013464, -0.9872322, -2.5001545, 2.4832954
9: 0.1713096, 2.2579260, 0.1630850, 2.2653689, -2.0940595, 2.0948410

Time for backsubstitution: 15.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.43 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3051355, upper bound: 1.3007509
time: 4.13 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3378969, upper bound: 1.3479612
time: 4.46 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.1142960, -5.3342705, -8.1278324, -5.3859034, -2.2481470, 2.3228650
1: -9.1146135, -6.2016191, -9.2123814, -6.2292414, -2.8853722, 2.9996352
2: -9.9070463, -6.9914408, -9.9230490, -7.0082626, -2.5536342, 2.5933356
3: -10.8743801, -8.3505754, -10.8154984, -8.2866716, -2.1678772, 2.0572433
4: -5.5114365, -3.5095634, -5.5434432, -3.5452466, -1.8128800, 1.8777771
5: -8.8817596, -6.1868563, -8.8569717, -6.2010608, -1.8878145, 1.8746207
6: -12.9081221, -9.7112293, -12.9636412, -9.7602673, -2.3406653, 2.4443791
7: 0.3458734, 2.7341137, 0.4953136, 2.8283672, -2.2218468, 1.9858732
8: -3.6519032, -0.9351697, -3.6953039, -1.0277319, -2.4341393, 2.5819869
9: 0.2630699, 2.2650704, 0.1732543, 2.2550933, -1.9920235, 2.0918162

Time for backsubstitution: 14.83 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2888456, upper bound: 1.2564413
time: 3.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2888456, upper bound: 1.2564415
time: 3.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.1145945, -5.3349557, -8.1359806, -5.3882895, -2.2466745, 2.3476596
1: -9.1146240, -6.2000246, -9.2140770, -6.2127647, -2.9018593, 3.0037408
2: -9.9081755, -6.9917231, -9.9462013, -7.0101213, -2.5573454, 2.6376314
3: -10.8745022, -8.3505211, -10.8163214, -8.2848644, -2.1687570, 2.0574677
4: -5.5113554, -3.5092330, -5.5424709, -3.5428865, -1.8182387, 1.8797064
5: -8.8820190, -6.1868658, -8.8605442, -6.2012596, -1.8871350, 1.8817177
6: -12.9082642, -9.7110939, -12.9705677, -9.7582445, -2.3385139, 2.4547460
7: 0.3459630, 2.7345691, 0.4816909, 2.8425953, -2.2598746, 1.9842868
8: -3.6530876, -0.9367323, -3.6956553, -1.0225992, -2.4416876, 2.6165657
9: 0.2633590, 2.2654710, 0.1708477, 2.2597508, -1.9963919, 2.0946233

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2912188, upper bound: 1.2646405
time: 3.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2912188, upper bound: 1.2646405
time: 3.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.1583109, -5.3680158, -8.0774212, -5.3410282, -2.3426752, 2.2391238
1: -9.2176418, -6.2046900, -9.1012020, -6.2079577, -3.0096841, 2.8965120
2: -9.9483604, -6.9536762, -9.8916731, -7.0308886, -2.5815964, 2.5949354
3: -10.8320303, -8.2779474, -10.8603325, -8.3695946, -2.0548339, 2.1671429
4: -5.5529518, -3.5127769, -5.5002260, -3.5368199, -1.8621659, 1.8343320
5: -8.8868132, -6.1921716, -8.8562021, -6.1959600, -1.8969426, 1.8732142
6: -12.9646530, -9.7501202, -12.9008265, -9.7182255, -2.4390750, 2.3480279
7: 0.4067149, 2.8217978, 0.3875179, 2.7263870, -2.0681329, 2.1908665
8: -3.7013464, -0.9872322, -3.6383677, -0.9676361, -2.5689945, 2.4672325
9: 0.1630850, 2.2653689, 0.2698810, 2.2579274, -2.0948424, 1.9954879

Time for backsubstitution: 15.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657652, upper bound: 1.2792189
time: 4.29 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739626, upper bound: 1.2815945
time: 4.22 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1583109, -5.3680158, -8.1204453, -5.3759203, -2.2893524, 2.2629073
1: -9.2176418, -6.2046900, -9.2042723, -6.2131233, -2.9443393, 2.9384146
2: -9.9483604, -6.9536762, -9.9321079, -6.9953933, -2.5981693, 2.6202312
3: -10.8320303, -8.2779474, -10.8173695, -8.2978544, -2.0601559, 2.0648081
4: -5.5529518, -3.5127769, -5.5411634, -3.5408163, -1.8663721, 1.8818221
5: -8.8868132, -6.1921716, -8.8605442, -6.2012568, -1.8984308, 1.8823268
6: -12.9646530, -9.7501202, -12.9573345, -9.7571383, -2.4047995, 2.4053164
7: 0.4067149, 2.8217978, 0.4528751, 2.8134508, -2.0931456, 2.0591068
8: -3.7013464, -0.9872322, -3.6860209, -1.0203171, -2.4832954, 2.5001543
9: 0.1630850, 2.2653689, 0.1713096, 2.2579260, -2.0948410, 2.0940595

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3147522, upper bound: 1.2914280
time: 3.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3479609, upper bound: 1.3378962
time: 4.13 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.1142960, -5.3342705, -8.1657238, -5.3779998, -2.2415371, 2.3424890
1: -9.1146135, -6.2016191, -9.2257586, -6.2208891, -2.8937244, 3.0241394
2: -9.9070463, -6.9914408, -9.9391594, -6.9665422, -2.5694885, 2.5869188
3: -10.8743801, -8.3505754, -10.8301706, -8.2667694, -2.1943765, 2.0781121
4: -5.5114365, -3.5095634, -5.5552292, -3.5171974, -1.8239498, 1.8740730
5: -8.8817596, -6.1868563, -8.8832455, -6.1919756, -1.8846378, 1.8874850
6: -12.9081221, -9.7112293, -12.9709625, -9.7532539, -2.3529873, 2.4583941
7: 0.3458734, 2.7341137, 0.4491997, 2.8368356, -2.2092533, 2.0060935
8: -3.6519032, -0.9351697, -3.7104492, -0.9946251, -2.4573584, 2.5890412
9: 0.2630699, 2.2650704, 0.1650854, 2.2624879, -1.9994180, 2.0999851

Time for backsubstitution: 15.15 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2888463, upper bound: 1.2564398
time: 3.91 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2888463, upper bound: 1.2564398
time: 3.88 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.1145945, -5.3349557, -8.1740389, -5.3803868, -2.2400656, 2.3672128
1: -9.1146240, -6.2000246, -9.2274456, -6.2044249, -2.9101992, 3.0274210
2: -9.9081755, -6.9917231, -9.9623280, -6.9684019, -2.5731997, 2.6312740
3: -10.8745022, -8.3505211, -10.8309956, -8.2649622, -2.1952581, 2.0783401
4: -5.5113554, -3.5092330, -5.5542521, -3.5148377, -1.8293085, 1.8759809
5: -8.8820190, -6.1868658, -8.8868237, -6.1921759, -1.8839550, 1.8946149
6: -12.9082642, -9.7110939, -12.9778900, -9.7512283, -2.3510990, 2.4687836
7: 0.3459630, 2.7345691, 0.4355721, 2.8510728, -2.2454100, 2.0044532
8: -3.6530876, -0.9367323, -3.7109237, -0.9894915, -2.4648643, 2.6236820
9: 0.2633590, 2.2654710, 0.1626736, 2.2671447, -2.0037856, 2.1027975

Time for backsubstitution: 14.83 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2912194, upper bound: 1.2646389
time: 3.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2912194, upper bound: 1.2646389
time: 3.27 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.1583109, -5.3680158, -8.1148834, -5.3330421, -2.3358583, 2.2588310
1: -9.2176418, -6.2046900, -9.1146488, -6.1993079, -3.0183339, 2.9099588
2: -9.9483604, -6.9536762, -9.9084682, -6.9891806, -2.5973907, 2.5892327
3: -10.8320303, -8.2779474, -10.8748140, -8.3504868, -2.0803514, 2.1881151
4: -5.5529518, -3.5127769, -5.5119295, -3.5088513, -1.8734913, 1.8304358
5: -8.8868132, -6.1921716, -8.8824034, -6.1868315, -1.8937907, 1.8859792
6: -12.9646530, -9.7501202, -12.9083214, -9.7108307, -2.4529700, 2.3615463
7: 0.4067149, 2.8217978, 0.3405704, 2.7348576, -2.0550404, 2.2119431
8: -3.7013464, -0.9872322, -3.6534615, -0.9343452, -2.5927753, 2.4737267
9: 0.1630850, 2.2653689, 0.2617226, 2.2655325, -2.1024475, 2.0036464

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657656, upper bound: 1.2793248
time: 3.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2739630, upper bound: 1.2816928
time: 3.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1583109, -5.3680158, -8.1583109, -5.3680158, -2.2825656, 2.2825656
1: -9.2176418, -6.2046900, -9.2176418, -6.2046900, -2.9831185, 2.9831181
2: -9.9483604, -6.9536762, -9.9483604, -6.9536762, -2.6139746, 2.6139746
3: -10.8320303, -8.2779474, -10.8320303, -8.2779474, -2.0857315, 2.0857315
4: -5.5529518, -3.5127769, -5.5529518, -3.5127769, -1.8778224, 1.8778224
5: -8.8868132, -6.1921716, -8.8868132, -6.1921716, -1.8952351, 1.8952348
6: -12.9646530, -9.7501202, -12.9646530, -9.7501202, -2.4190226, 2.4190221
7: 0.4067149, 2.8217978, 0.4067149, 2.8217978, -2.0796080, 2.0796080
8: -3.7013464, -0.9872322, -3.7013464, -0.9872322, -2.5066967, 2.5066967
9: 0.1630850, 2.2653689, 0.1630850, 2.2653689, -2.1022840, 2.1022840

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3147528, upper bound: 1.2914263
time: 3.28 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3479614, upper bound: 1.3382181
time: 3.87 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.04 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2564418
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2564418
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2815945, upper bound: 1.2646406
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2815945, upper bound: 1.2646406
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2564412, upper bound: 1.2792189
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2646397, upper bound: 1.2815943
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.3051355, upper bound: 1.2914279
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.3378969, upper bound: 1.3378964
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2657664
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2657664
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2815945, upper bound: 1.2739639
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2815945, upper bound: 1.2739639
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2564413, upper bound: 1.2888462
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2646398, upper bound: 1.2912192
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.3051355, upper bound: 1.3007509
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.3378969, upper bound: 1.3479612
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2888456, upper bound: 1.2564413
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2888456, upper bound: 1.2564415
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2912188, upper bound: 1.2646405
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2912188, upper bound: 1.2646405
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2657652, upper bound: 1.2792189
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2739626, upper bound: 1.2815945
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.3147522, upper bound: 1.2914280
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.3479609, upper bound: 1.3378962
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2888463, upper bound: 1.2564398
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2888463, upper bound: 1.2564398
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2912194, upper bound: 1.2646389
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2912194, upper bound: 1.2646389
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2657656, upper bound: 1.2793248
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.2739630, upper bound: 1.2816928
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.3147528, upper bound: 1.2914263
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.04
Output dim: 7, lower bound: -1.3479614, upper bound: 1.3382181

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.0774212, -5.3410282, -8.1126461, -5.3861866, -2.2118526, 2.2980409
1: -9.1012020, -6.2079577, -9.1860714, -6.2335415, -2.8676605, 2.9557815
2: -9.8916731, -7.0308886, -9.9137249, -7.0171623, -2.5292916, 2.5471144
3: -10.8603325, -8.3695946, -10.8108149, -8.3110332, -2.1246672, 2.0340514
4: -5.5002260, -3.5368199, -5.5328379, -3.5475385, -1.7991281, 1.8415346
5: -8.8562021, -6.1959600, -8.8536797, -6.2013354, -1.8594909, 1.8615375
6: -12.9008265, -9.7182255, -12.9483557, -9.7605028, -2.3314085, 2.4139285
7: 0.3875179, 2.7263870, 0.4987369, 2.8029957, -2.1478968, 1.9744177
8: -3.6383677, -0.9676361, -3.6775084, -1.0324354, -2.4167042, 2.5180650
9: 0.2698810, 2.2579274, 0.1980879, 2.2526095, -1.9827285, 2.0598395

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2564418
time: 3.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2564418
time: 3.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1278324, -5.3859034, -2.2394900, 2.2895365
1: -9.2042723, -6.2131233, -9.2123814, -6.2292414, -2.9074478, 2.9992580
2: -9.9321079, -6.9953933, -9.9230490, -7.0082626, -2.5666037, 2.5740752
3: -10.8173695, -8.2978544, -10.8154984, -8.2866716, -2.1511712, 2.0436599
4: -5.5411634, -3.5408163, -5.5434432, -3.5452466, -1.8499928, 1.8296947
5: -8.8605442, -6.2012568, -8.8569717, -6.2010608, -1.8655443, 1.8669267
6: -12.9573345, -9.7571383, -12.9636412, -9.7602673, -2.3879347, 2.4112971
7: 0.4528751, 2.8134508, 0.4953136, 2.8283672, -2.1694722, 2.0035872
8: -3.6860209, -1.0203171, -3.6953039, -1.0277319, -2.4522510, 2.5662460
9: 0.1713096, 2.2579260, 0.1732543, 2.2550933, -2.0837836, 2.0846717

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2596604, upper bound: 1.2564419
time: 3.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2596604, upper bound: 1.2564414
time: 3.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.0774212, -5.3410282, -8.1202164, -5.3885727, -2.2101665, 2.3215432
1: -9.1012020, -6.2079577, -9.1874771, -6.2170000, -2.8842020, 2.9561882
2: -9.8916731, -7.0308886, -9.9367294, -7.0189891, -2.5323043, 2.5887425
3: -10.8603325, -8.3695946, -10.8116627, -8.3092213, -2.1255398, 2.0342443
4: -5.5002260, -3.5368199, -5.5320873, -3.5451603, -1.8036952, 1.8440504
5: -8.8562021, -6.1959600, -8.8573103, -6.2015314, -1.8589306, 1.8683903
6: -12.9008265, -9.7182255, -12.9553099, -9.7584934, -2.3290348, 2.4232543
7: 0.3875179, 2.7263870, 0.4850602, 2.8172898, -2.1796536, 1.9724317
8: -3.6383677, -0.9676361, -3.6763434, -1.0276613, -2.4234705, 2.5524302
9: 0.2698810, 2.2579274, 0.1956404, 2.2572865, -1.9874055, 2.0622869

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2792183, upper bound: 1.2564418
time: 3.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2815945, upper bound: 1.2646406
time: 3.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1359806, -5.3882895, -2.2373886, 2.3137357
1: -9.2042723, -6.2131233, -9.2140770, -6.2127647, -2.9473500, 3.0009537
2: -9.9321079, -6.9953933, -9.9462013, -7.0101213, -2.5694308, 2.6174057
3: -10.8173695, -8.2978544, -10.8163214, -8.2848644, -2.1521506, 2.0444722
4: -5.5411634, -3.5408163, -5.5424709, -3.5428865, -1.8565998, 1.8316641
5: -8.8605442, -6.2012568, -8.8605442, -6.2012596, -1.8649831, 1.8762712
6: -12.9573345, -9.7571383, -12.9705677, -9.7582445, -2.3902264, 2.4212615
7: 0.4528751, 2.8134508, 0.4816909, 2.8425953, -2.2018666, 2.0079327
8: -3.6860209, -1.0203171, -3.6956553, -1.0225992, -2.4603467, 2.6000781
9: 0.1713096, 2.2579260, 0.1708477, 2.2597508, -2.0884414, 2.0870783

Time for backsubstitution: 14.50 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.2286510467529297
rel_dist={7: [-1.3704629636566878, 1.370462874585793]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0564516, upper bound: 1.0619857
time: 3.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0634191, upper bound: 1.0634194
time: 3.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.88 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.88
Output dim: 7, lower bound: -1.0564516, upper bound: 1.0619857
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.88
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
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0564501, upper bound: 1.0564497
time: 3.82 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0564501, upper bound: 1.0619854
time: 3.89 seconds

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

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0619858, upper bound: 1.0564498
time: 3.72 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0619858, upper bound: 1.0564522
time: 3.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.12 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.12
Output dim: 7, lower bound: -1.0564501, upper bound: 1.0564497
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.12
Output dim: 7, lower bound: -1.0564501, upper bound: 1.0619854
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.12
Output dim: 7, lower bound: -1.0619858, upper bound: 1.0564498
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.12
Output dim: 7, lower bound: -1.0619858, upper bound: 1.0564522

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.1321526, -5.3759112, -8.1321526, -5.3759112, -1.9556451, 1.9556451
1: -9.2126160, -6.2120185, -9.2126160, -6.2120185, -2.7638922, 2.7638917
2: -9.9340887, -6.9920025, -9.9340887, -6.9920025, -2.3315291, 2.3315291
3: -10.8188086, -8.2860575, -10.8188086, -8.2860575, -1.9235001, 1.9235001
4: -5.5464420, -3.5399156, -5.5464420, -3.5399156, -1.6105857, 1.6105857
5: -8.8613005, -6.2009053, -8.8613005, -6.2009053, -1.5839109, 1.5839109
6: -12.9650011, -9.7570105, -12.9650011, -9.7570105, -2.0580521, 2.0580521
7: 0.4514637, 2.8336329, 0.4514637, 2.8336329, -1.9637098, 1.9637098
8: -3.7049961, -1.0193210, -3.7049961, -1.0193210, -2.2981420, 2.2981420
9: 0.1637008, 2.2586241, 0.1637008, 2.2586241, -2.0607829, 2.0607829

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0099303, upper bound: 1.0031626
time: 3.64 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0389483, upper bound: 1.0389480
time: 3.71 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.1321526, -5.3759112, -8.1700544, -5.3680086, -1.9656510, 1.9911079
1: -9.2126160, -6.2120185, -9.2260036, -6.2035923, -2.7729006, 2.7792535
2: -9.9340887, -6.9920025, -9.9503212, -6.9502831, -2.3711991, 2.3489923
3: -10.8188086, -8.2860575, -10.8334780, -8.2661514, -1.9440794, 1.9384580
4: -5.5464420, -3.5399156, -5.5582314, -3.5118756, -1.6379175, 1.6231437
5: -8.8613005, -6.2009053, -8.8875713, -6.1918206, -1.5938296, 1.6098902
6: -12.9650011, -9.7570105, -12.9723415, -9.7499933, -2.0656776, 2.0670395
7: 0.4514637, 2.8336329, 0.4052863, 2.8421242, -1.9731989, 1.9942327
8: -3.7049961, -1.0193210, -3.7202139, -0.9862318, -2.3257294, 2.3132482
9: 0.1637008, 2.2586241, 0.1555148, 2.2660573, -2.0681944, 2.0687380

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0099303, upper bound: 1.0086984
time: 3.77 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0389483, upper bound: 1.0444863
time: 3.75 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.1700544, -5.3680086, -8.1321526, -5.3759112, -1.9911079, 1.9656510
1: -9.2260036, -6.2035923, -9.2126160, -6.2120185, -2.7792530, 2.7729001
2: -9.9503212, -6.9502831, -9.9340887, -6.9920025, -2.3489919, 2.3711991
3: -10.8334780, -8.2661514, -10.8188086, -8.2860575, -1.9384575, 1.9440789
4: -5.5582314, -3.5118756, -5.5464420, -3.5399156, -1.6231437, 1.6379175
5: -8.8875713, -6.1918206, -8.8613005, -6.2009053, -1.6098900, 1.5938296
6: -12.9723415, -9.7499933, -12.9650011, -9.7570105, -2.0670400, 2.0656772
7: 0.4052863, 2.8421242, 0.4514637, 2.8336329, -1.9942327, 1.9731989
8: -3.7202139, -0.9862318, -3.7049961, -1.0193210, -2.3132482, 2.3257296
9: 0.1555148, 2.2660573, 0.1637008, 2.2586241, -2.0687380, 2.0681944

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0154672, upper bound: 1.0031625
time: 4.00 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0444843, upper bound: 1.0389480
time: 3.62 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.1700544, -5.3680086, -8.1700544, -5.3680086, -1.9792833, 1.9792838
1: -9.2260036, -6.2035923, -9.2260036, -6.2035923, -2.8134899, 2.8134899
2: -9.9503212, -6.9502831, -9.9503212, -6.9502831, -2.3608890, 2.3608890
3: -10.8334780, -8.2661514, -10.8334780, -8.2661514, -1.9643617, 1.9643614
4: -5.5582314, -3.5118756, -5.5582314, -3.5118756, -1.6314783, 1.6314783
5: -8.8875713, -6.1918206, -8.8875713, -6.1918206, -1.6045203, 1.6045203
6: -12.9723415, -9.7499933, -12.9723415, -9.7499933, -2.0790229, 2.0790229
7: 0.4052863, 2.8421242, 0.4052863, 2.8421242, -1.9888687, 1.9888687
8: -3.7202139, -0.9862318, -3.7202139, -0.9862318, -2.3306255, 2.3306255
9: 0.1555148, 2.2660573, 0.1555148, 2.2660573, -2.0793929, 2.0793929

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0154679, upper bound: 1.0051448
time: 3.75 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0444851, upper bound: 1.0408127
time: 3.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.20 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 7, lower bound: -1.0099303, upper bound: 1.0031626
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 7, lower bound: -1.0389483, upper bound: 1.0389480
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 7, lower bound: -1.0099303, upper bound: 1.0086984
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 7, lower bound: -1.0389483, upper bound: 1.0444863
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 7, lower bound: -1.0154672, upper bound: 1.0031625
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 7, lower bound: -1.0444843, upper bound: 1.0389480
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 7, lower bound: -1.0154679, upper bound: 1.0051448
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 7, lower bound: -1.0444851, upper bound: 1.0408127

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.0774212, -5.3410282, -8.1039419, -5.3764763, -1.8915310, 1.9489882
1: -9.1012020, -6.2079577, -9.1621637, -6.2211642, -2.6297617, 2.6491861
2: -9.8916731, -7.0308886, -9.9149570, -7.0092044, -2.2857614, 2.2846856
3: -10.8603325, -8.3695946, -10.8095570, -8.3332291, -1.8699989, 1.8045003
4: -5.5002260, -3.5368199, -5.5260410, -3.5444942, -1.5732446, 1.6138887
5: -8.8562021, -6.1959600, -8.8548040, -6.2014389, -1.5733109, 1.5771911
6: -12.9008265, -9.7182255, -12.9361696, -9.7575121, -1.9807029, 2.0375757
7: 0.3875179, 2.7263870, 0.4582853, 2.7854729, -1.9078417, 1.8114314
8: -3.6383677, -0.9676361, -3.6739621, -1.0293846, -2.1527824, 2.2180436
9: 0.2698810, 2.2579274, 0.2115762, 2.2535343, -1.9501662, 2.0138888

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9928398, upper bound: 0.9808605
time: 3.37 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9896252
time: 3.68 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1307764, -5.3759127, -1.9156575, 1.9542227
1: -9.2042723, -6.2131233, -9.2116528, -6.2121477, -2.6335649, 2.7496724
2: -9.9321079, -6.9953933, -9.9338589, -6.9923944, -2.3290009, 2.3189497
3: -10.8173695, -8.2978544, -10.8186407, -8.2874222, -1.9213905, 1.8156395
4: -5.5411634, -3.5408163, -5.5458302, -3.5400190, -1.6163144, 1.6080065
5: -8.8605442, -6.2012568, -8.8612127, -6.2009463, -1.5817051, 1.5844598
6: -12.9573345, -9.7571383, -12.9636583, -9.7570229, -2.0300045, 2.0531399
7: 0.4528751, 2.8134508, 0.4516292, 2.8310819, -1.9617810, 1.8455753
8: -3.6860209, -1.0203171, -3.7024031, -1.0194368, -2.2047539, 2.2964787
9: 0.1713096, 2.2579260, 0.1646426, 2.2585430, -2.0023375, 2.0592875

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0240459, upper bound: 1.0205391
time: 3.39 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0268038, upper bound: 1.0268046
time: 3.43 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.0774212, -5.3410282, -8.1415272, -5.3685589, -1.9015484, 1.9812961
1: -9.1012020, -6.2079577, -9.1757336, -6.2126327, -2.6388779, 2.6643653
2: -9.8916731, -7.0308886, -9.9315119, -6.9674878, -2.3254290, 2.3024545
3: -10.8603325, -8.3695946, -10.8241072, -8.3133202, -1.8903675, 1.8192730
4: -5.5002260, -3.5368199, -5.5378137, -3.5164883, -1.6005273, 1.6262722
5: -8.8562021, -6.1959600, -8.8810616, -6.1923513, -1.5832310, 1.6031075
6: -12.9008265, -9.7182255, -12.9436150, -9.7504921, -1.9883299, 2.0465446
7: 0.3875179, 2.7263870, 0.4121637, 2.7939692, -1.9173112, 1.8428569
8: -3.6383677, -0.9676361, -3.6890268, -0.9963384, -2.1845665, 2.2331562
9: 0.2698810, 2.2579274, 0.2033907, 2.2610557, -1.9576602, 2.0217190

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9928397, upper bound: 0.9863948
time: 3.43 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9951599
time: 3.69 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1686764, -5.3680077, -1.9256997, 1.9897170
1: -9.2042723, -6.2131233, -9.2250195, -6.2037201, -2.6426525, 2.7650342
2: -9.9321079, -6.9953933, -9.9500942, -6.9506745, -2.3686709, 2.3364892
3: -10.8173695, -8.2978544, -10.8333101, -8.2675152, -1.9419699, 1.8306594
4: -5.5411634, -3.5408163, -5.5576210, -3.5119791, -1.6440506, 1.6205640
5: -8.8605442, -6.2012568, -8.8874855, -6.1918612, -1.5916233, 1.6104887
6: -12.9573345, -9.7571383, -12.9709873, -9.7500114, -2.0392218, 2.0621281
7: 0.4528751, 2.8134508, 0.4054527, 2.8395746, -1.9712691, 1.8780124
8: -3.6860209, -1.0203171, -3.7176204, -0.9863477, -2.2311697, 2.3115835
9: 0.1713096, 2.2579260, 0.1564586, 2.2659776, -2.0098515, 2.0672421

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0240459, upper bound: 1.0260763
time: 3.52 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0268039, upper bound: 1.0323418
time: 3.60 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.1148834, -5.3330421, -8.1039419, -5.3764763, -1.9278541, 1.9590387
1: -9.1146488, -6.1993079, -9.1621637, -6.2211642, -2.6449137, 2.6584253
2: -9.9084682, -6.9891806, -9.9149570, -7.0092044, -2.3038750, 2.3242908
3: -10.8748140, -8.3504868, -10.8095570, -8.3332291, -1.8850565, 1.8241003
4: -5.5119295, -3.5088513, -5.5260410, -3.5444942, -1.5856299, 1.6414948
5: -8.8824034, -6.1868315, -8.8548040, -6.2014389, -1.5991812, 1.5871465
6: -12.9083214, -9.7108307, -12.9361696, -9.7575121, -1.9895210, 2.0464687
7: 0.3405704, 2.7348576, 0.4582853, 2.7854729, -1.9399354, 1.8208599
8: -3.6534615, -0.9343452, -3.6739621, -1.0293846, -2.1679640, 2.2448990
9: 0.2617226, 2.2655325, 0.2115762, 2.2535343, -1.9577093, 2.0214715

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9983771, upper bound: 0.9808604
time: 3.32 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0040017, upper bound: 0.9896251
time: 3.83 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.1583109, -5.3680158, -8.1307764, -5.3759127, -1.9521413, 1.9642258
1: -9.2176418, -6.2046900, -9.2116528, -6.2121477, -2.6485624, 2.7586446
2: -9.9483604, -6.9536762, -9.9338589, -6.9923944, -2.3464804, 2.3585663
3: -10.8320303, -8.2779474, -10.8186407, -8.2874222, -1.9363356, 1.8353004
4: -5.5529518, -3.5127769, -5.5458302, -3.5400190, -1.6285915, 1.6353130
5: -8.8868132, -6.1921716, -8.8612127, -6.2009463, -1.6076670, 1.5943706
6: -12.9646530, -9.7501202, -12.9636583, -9.7570229, -2.0387049, 2.0607655
7: 0.4067149, 2.8217978, 0.4516292, 2.8310819, -1.9923043, 1.8549912
8: -3.7013464, -0.9872322, -3.7024031, -1.0194368, -2.2198143, 2.3240535
9: 0.1630850, 2.2653689, 0.1646426, 2.2585430, -2.0101652, 2.0667067

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0295831, upper bound: 1.0205390
time: 3.34 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0323403, upper bound: 1.0268047
time: 3.80 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.1148834, -5.3330421, -8.1415272, -5.3685589, -1.9151587, 1.9725940
1: -9.1146488, -6.1993079, -9.1757336, -6.2126327, -2.6791153, 2.6987886
2: -9.9084682, -6.9891806, -9.9315119, -6.9674878, -2.3157625, 2.3142862
3: -10.8748140, -8.3504868, -10.8241072, -8.3133202, -1.9107480, 1.8441987
4: -5.5119295, -3.5088513, -5.5378137, -3.5164883, -1.5939188, 1.6348858
5: -8.8824034, -6.1868315, -8.8810616, -6.1923513, -1.5938148, 1.5977731
6: -12.9083214, -9.7108307, -12.9436150, -9.7504921, -2.0011926, 2.0596197
7: 0.3405704, 2.7348576, 0.4121637, 2.7939692, -1.9341488, 1.8364210
8: -3.6534615, -0.9343452, -3.6890268, -0.9963384, -2.1852970, 2.2508869
9: 0.2617226, 2.2655325, 0.2033907, 2.2610557, -1.9682961, 2.0322266

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0000726, upper bound: 0.9828385
time: 3.58 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0056974, upper bound: 0.9916033
time: 3.82 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.1583109, -5.3680158, -8.1686764, -5.3680077, -1.9392920, 1.9778571
1: -9.2176418, -6.2046900, -9.2250195, -6.2037201, -2.6828656, 2.7992349
2: -9.9483604, -6.9536762, -9.9500942, -6.9506745, -2.3583770, 2.3483310
3: -10.8320303, -8.2779474, -10.8333101, -8.2675152, -1.9622393, 1.8556437
4: -5.5529518, -3.5127769, -5.5576210, -3.5119791, -1.6373386, 1.6288748
5: -8.8868132, -6.1921716, -8.8874855, -6.1918612, -1.6022968, 1.6051106
6: -12.9646530, -9.7501202, -12.9709873, -9.7500114, -2.0522709, 2.0741110
7: 0.4067149, 2.8217978, 0.4054527, 2.8395746, -1.9868841, 1.8712976
8: -3.7013464, -0.9872322, -3.7176204, -0.9863477, -2.2372203, 2.3289690
9: 0.1630850, 2.2653689, 0.1564586, 2.2659776, -2.0209174, 2.0778885

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0311077, upper bound: 1.0224095
time: 3.37 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0338659, upper bound: 1.0286871
time: 3.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.92 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -0.9928398, upper bound: 0.9808605
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9896252
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -1.0240459, upper bound: 1.0205391
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -1.0268038, upper bound: 1.0268046
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -0.9928397, upper bound: 0.9863948
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9951599
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -1.0240459, upper bound: 1.0260763
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -1.0268039, upper bound: 1.0323418
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -0.9983771, upper bound: 0.9808604
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -1.0040017, upper bound: 0.9896251
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -1.0295831, upper bound: 1.0205390
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -1.0323403, upper bound: 1.0268047
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -1.0000726, upper bound: 0.9828385
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -1.0056974, upper bound: 0.9916033
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -1.0311077, upper bound: 1.0224095
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.92
Output dim: 7, lower bound: -1.0338659, upper bound: 1.0286871

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.0754862, -5.3450303, -8.1278324, -5.3859034, -1.8739791, 1.9717712
1: -9.1010914, -6.2154675, -9.2123814, -6.2292414, -2.5981102, 2.6901464
2: -9.8871136, -7.0382557, -9.9230490, -7.0082626, -2.2723346, 2.2847791
3: -10.8589172, -8.3698845, -10.8154984, -8.2866716, -1.9225345, 1.8079305
4: -5.4986420, -3.5391738, -5.5434432, -3.5452466, -1.5675578, 1.6189327
5: -8.8541164, -6.1960359, -8.8569717, -6.2010608, -1.5714507, 1.5749383
6: -12.9001904, -9.7195282, -12.9636412, -9.7602673, -1.9692664, 2.0714369
7: 0.4048052, 2.7239385, 0.4953136, 2.8283672, -1.9539204, 1.7650261
8: -3.6335325, -0.9703298, -3.6953039, -1.0277319, -2.1398554, 2.2689805
9: 0.2742224, 2.2564554, 0.1732543, 2.2550933, -1.9166512, 2.0261898

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9928398, upper bound: 0.9808605
time: 3.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9928398, upper bound: 0.9808605
time: 3.46 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.0765009, -5.3470984, -8.1359806, -5.3882895, -1.8760486, 1.9983547
1: -9.1011314, -6.2102652, -9.2140770, -6.2127647, -2.6195459, 2.7007694
2: -9.8907433, -7.0390263, -9.9462013, -7.0101213, -2.2801399, 2.3346996
3: -10.8593407, -8.3697014, -10.8163214, -8.2848644, -1.9232178, 1.8078022
4: -5.4984097, -3.5380495, -5.5424709, -3.5428865, -1.5744238, 1.6208234
5: -8.8549881, -6.1960635, -8.8605442, -6.2012596, -1.5705810, 1.5817223
6: -12.9006557, -9.7190580, -12.9705677, -9.7582445, -1.9676499, 2.0827248
7: 0.4047103, 2.7254429, 0.4816909, 2.8425953, -1.9972544, 1.7668386
8: -3.6371660, -0.9752846, -3.6956553, -1.0225992, -2.1527004, 2.3055491
9: 0.2750106, 2.2577384, 0.1708477, 2.2597508, -1.9213209, 2.0486040

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9896252
time: 3.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9896253
time: 4.03 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.1185226, -5.3804321, -8.1278324, -5.3859034, -1.9008536, 1.9450622
1: -9.2041674, -6.2211103, -9.2123814, -6.2292414, -2.5953865, 2.7166162
2: -9.9270658, -7.0027356, -9.9230490, -7.0082626, -2.3018298, 2.2990098
3: -10.8158751, -8.2981367, -10.8154984, -8.2866716, -1.9201226, 1.8127253
4: -5.5398026, -3.5432072, -5.5434432, -3.5452466, -1.6059608, 1.5972514
5: -8.8585453, -6.2013283, -8.8569717, -6.2010608, -1.5740066, 1.5749578
6: -12.9567490, -9.7586384, -12.9636412, -9.7602673, -2.0186410, 2.0449889
7: 0.4731579, 2.8109775, 0.4953136, 2.8283672, -1.9361968, 1.7967048
8: -3.6815100, -1.0242414, -3.6953039, -1.0277319, -2.1843133, 2.2822766
9: 0.1756313, 2.2563004, 0.1732543, 2.2550933, -1.9645433, 2.0231023

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9928398, upper bound: 0.9808605
time: 3.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0240459, upper bound: 1.0205391
time: 3.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1195583, -5.3815231, -8.1359806, -5.3882895, -1.9013453, 1.9744391
1: -9.2042065, -6.2155561, -9.2140770, -6.2127647, -2.6324191, 2.7266111
2: -9.9309359, -7.0035620, -9.9462013, -7.0101213, -2.3099051, 2.3502994
3: -10.8164215, -8.2979689, -10.8163214, -8.2848644, -1.9217749, 1.8129742
4: -5.5390544, -3.5419912, -5.5424709, -3.5428865, -1.6151824, 1.5997729
5: -8.8593807, -6.2013531, -8.8605442, -6.2012596, -1.5732803, 1.5848548
6: -12.9571419, -9.7579002, -12.9705677, -9.7582445, -2.0210948, 2.0581400
7: 0.4664931, 2.8125119, 0.4816909, 2.8425953, -1.9805822, 1.8007975
8: -3.6847596, -1.0267463, -3.6956553, -1.0225992, -2.1981797, 2.3202639
9: 0.1758590, 2.2576392, 0.1708477, 2.2597508, -1.9696660, 2.0453653

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9896253
time: 3.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0268038, upper bound: 1.0268046
time: 3.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.0754862, -5.3450303, -8.1657238, -5.3779998, -1.8839860, 2.0038769
1: -9.1010914, -6.2154675, -9.2257586, -6.2208891, -2.6070395, 2.7053967
2: -9.8871136, -7.0382557, -9.9391594, -6.9665422, -2.3120031, 2.3021712
3: -10.8589172, -8.3698845, -10.8301706, -8.2667694, -1.9431200, 1.8228834
4: -5.4986420, -3.5391738, -5.5552292, -3.5171974, -1.5949082, 1.6315098
5: -8.8541164, -6.1960359, -8.8832455, -6.1919756, -1.5813789, 1.6009102
6: -12.9001904, -9.7195282, -12.9709625, -9.7532539, -1.9768863, 2.0804229
7: 0.4048052, 2.7239385, 0.4491997, 2.8368356, -1.9633727, 1.7965829
8: -3.6335325, -0.9703298, -3.7104492, -0.9946251, -2.1716452, 2.2839952
9: 0.2742224, 2.2564554, 0.1650854, 2.2624879, -1.9240556, 2.0340080

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9928397, upper bound: 0.9863948
time: 3.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9928397, upper bound: 0.9863941
time: 3.45 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.0765009, -5.3470984, -8.1740389, -5.3803868, -1.8860555, 2.0295219
1: -9.1011314, -6.2102652, -9.2274456, -6.2044249, -2.6285124, 2.7160220
2: -9.8907433, -7.0390263, -9.9623280, -6.9684019, -2.3198094, 2.3521528
3: -10.8593407, -8.3697014, -10.8309956, -8.2649622, -1.9438052, 1.8227577
4: -5.4984097, -3.5380495, -5.5542521, -3.5148377, -1.6017742, 1.6333785
5: -8.8549881, -6.1960635, -8.8868237, -6.1921759, -1.5805073, 1.6077256
6: -12.9006557, -9.7190580, -12.9778900, -9.7512283, -1.9755349, 2.0917082
7: 0.4047103, 2.7254429, 0.4355721, 2.8510728, -2.0067301, 1.7985744
8: -3.6371660, -0.9752846, -3.7109237, -0.9894915, -2.1846113, 2.3206029
9: 0.2750106, 2.2577384, 0.1626736, 2.2671447, -1.9286427, 2.0564051

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9951599
time: 3.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9951596
time: 3.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.1185226, -5.3804321, -8.1657238, -5.3779998, -1.9108963, 1.9806018
1: -9.2041674, -6.2211103, -9.2257586, -6.2208891, -2.6043968, 2.7318664
2: -9.9270658, -7.0027356, -9.9391594, -6.9665422, -2.3414989, 2.3164759
3: -10.8158751, -8.2981367, -10.8301706, -8.2667694, -1.9407082, 1.8277423
4: -5.5398026, -3.5432072, -5.5552292, -3.5171974, -1.6336875, 1.6098285
5: -8.8585453, -6.2013283, -8.8832455, -6.1919756, -1.5839343, 1.6009765
6: -12.9567490, -9.7586384, -12.9709625, -9.7532539, -2.0278201, 2.0539751
7: 0.4731579, 2.8109775, 0.4491997, 2.8368356, -1.9456491, 1.8282955
8: -3.6815100, -1.0242414, -3.7104492, -0.9946251, -2.2109933, 2.2972913
9: 0.1756313, 2.2563004, 0.1650854, 2.2624879, -1.9718623, 2.0309205

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9928397, upper bound: 0.9863941
time: 3.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0240459, upper bound: 1.0260763
time: 3.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1195583, -5.3815231, -8.1740389, -5.3803868, -1.9113884, 2.0085044
1: -9.2042065, -6.2155561, -9.2274456, -6.2044249, -2.6414537, 2.7418637
2: -9.9309359, -7.0035620, -9.9623280, -6.9684019, -2.3495746, 2.3678541
3: -10.8164215, -8.2979689, -10.8309956, -8.2649622, -1.9423623, 1.8279943
4: -5.5390544, -3.5419912, -5.5542521, -3.5148377, -1.6429129, 1.6123285
5: -8.8593807, -6.2013531, -8.8868237, -6.1921759, -1.5832067, 1.6108084
6: -12.9571419, -9.7579002, -12.9778900, -9.7512283, -2.0303149, 2.0671232
7: 0.4664931, 2.8125119, 0.4355721, 2.8510728, -1.9900579, 1.8329897
8: -3.6847596, -1.0267463, -3.7109237, -0.9894915, -2.2246976, 2.3353181
9: 0.1758590, 2.2576392, 0.1626736, 2.2671447, -1.9769878, 2.0531659

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9951596
time: 3.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0268039, upper bound: 1.0323418
time: 3.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.1129456, -5.3370485, -8.1278324, -5.3859034, -1.9102960, 1.9818215
1: -9.1145344, -6.2068591, -9.2123814, -6.2292414, -2.6132631, 2.6993570
2: -9.9038439, -6.9965477, -9.9230490, -7.0082626, -2.2903996, 2.3243852
3: -10.8733978, -8.3507786, -10.8154984, -8.2866716, -1.9375911, 1.8275349
4: -5.5103736, -3.5112011, -5.5434432, -3.5452466, -1.5799427, 1.6465325
5: -8.8803196, -6.1869078, -8.8569717, -6.2010608, -1.5973191, 1.5849030
6: -12.9076729, -9.7121344, -12.9636412, -9.7602673, -1.9780884, 2.0803187
7: 0.3578620, 2.7324290, 0.4953136, 2.8283672, -1.9861186, 1.7744374
8: -3.6485810, -0.9370360, -3.6953039, -1.0277319, -2.1549969, 2.2952869
9: 0.2660697, 2.2640386, 0.1732543, 2.2550933, -1.9242268, 2.0337682

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9983771, upper bound: 0.9808604
time: 3.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9983771, upper bound: 0.9808605
time: 3.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.1139622, -5.3391218, -8.1359806, -5.3882895, -1.9123688, 2.0083973
1: -9.1145782, -6.2016153, -9.2140770, -6.2127647, -2.6347008, 2.7100101
2: -9.9075317, -6.9973192, -9.9462013, -7.0101213, -2.2982478, 2.3743057
3: -10.8738174, -8.3505964, -10.8163214, -8.2848644, -1.9382715, 1.8274024
4: -5.5101018, -3.5100796, -5.5424709, -3.5428865, -1.5867105, 1.6484261
5: -8.8811913, -6.1869326, -8.8605442, -6.2012596, -1.5964518, 1.5916789
6: -12.9081459, -9.7116623, -12.9705677, -9.7582445, -1.9764786, 2.0916469
7: 0.3577180, 2.7339430, 0.4816909, 2.8425953, -2.0312741, 1.7762651
8: -3.6522570, -0.9419827, -3.6956553, -1.0225992, -2.1678791, 2.3337543
9: 0.2668936, 2.2653403, 0.1708477, 2.2597508, -1.9287572, 2.0562005

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0040017, upper bound: 0.9896251
time: 4.05 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0040017, upper bound: 0.9896251
time: 4.00 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.1563854, -5.3725295, -8.1278324, -5.3859034, -1.9369402, 1.9550643
1: -9.2175360, -6.2127151, -9.2123814, -6.2292414, -2.6103859, 2.7255392
2: -9.9432602, -6.9610167, -9.9230490, -7.0082626, -2.3192859, 2.3386273
3: -10.8305378, -8.2782297, -10.8154984, -8.2866716, -1.9350662, 1.8323896
4: -5.5515904, -3.5151625, -5.5434432, -3.5452466, -1.6183376, 1.6245532
5: -8.8848171, -6.1922450, -8.8569717, -6.2010608, -1.5999641, 1.5848699
6: -12.9640589, -9.7516232, -12.9636412, -9.7602673, -2.0273428, 2.0526116
7: 0.4269991, 2.8193116, 0.4953136, 2.8283672, -1.9666679, 1.8061101
8: -3.6968007, -0.9911561, -3.6953039, -1.0277319, -2.1993308, 2.3099849
9: 0.1674142, 2.2637248, 0.1732543, 2.2550933, -1.9723940, 2.0305181

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9983771, upper bound: 0.9808605
time: 3.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0295831, upper bound: 1.0205390
time: 3.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1574202, -5.3736181, -8.1359806, -5.3882895, -1.9378266, 1.9844422
1: -9.2175760, -6.2071295, -9.2140770, -6.2127647, -2.6474185, 2.7355771
2: -9.9471817, -6.9618435, -9.9462013, -7.0101213, -2.3273792, 2.3899174
3: -10.8310823, -8.2780609, -10.8163214, -8.2848644, -1.9367166, 1.8326356
4: -5.5508451, -3.5139492, -5.5424709, -3.5428865, -1.6274605, 1.6270771
5: -8.8856544, -6.1922679, -8.8605442, -6.2012596, -1.5992408, 1.5947669
6: -12.9644604, -9.7508850, -12.9705677, -9.7582445, -2.0298028, 2.0657647
7: 0.4203987, 2.8208578, 0.4816909, 2.8425953, -2.0129874, 1.8102167
8: -3.7000813, -0.9936533, -3.6956553, -1.0225992, -2.2132344, 2.3494279
9: 0.1676426, 2.2650800, 0.1708477, 2.2597508, -1.9774551, 2.0527921

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0040017, upper bound: 0.9896251
time: 3.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0323403, upper bound: 1.0268047
time: 3.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.1129456, -5.3370485, -8.1657238, -5.3779998, -1.8976164, 1.9953771
1: -9.1145344, -6.2068591, -9.2257586, -6.2208891, -2.6472778, 2.7398372
2: -9.9038439, -6.9965477, -9.9391594, -6.9665422, -2.3022904, 2.3140049
3: -10.8733978, -8.3507786, -10.8301706, -8.2667694, -1.9634986, 1.8478127
4: -5.5103736, -3.5112011, -5.5552292, -3.5171974, -1.5882993, 1.6401167
5: -8.8803196, -6.1869078, -8.8832455, -6.1919756, -1.5919600, 1.5955853
6: -12.9076729, -9.7121344, -12.9709625, -9.7532539, -1.9897590, 2.0936778
7: 0.3578620, 2.7324290, 0.4491997, 2.8368356, -1.9802942, 1.7904167
8: -3.6485810, -0.9370360, -3.7104492, -0.9946251, -2.1722603, 2.3018999
9: 0.2660697, 2.2640386, 0.1650854, 2.2624879, -1.9349222, 2.0445356

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0000726, upper bound: 0.9828385
time: 3.45 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0000726, upper bound: 0.9828385
time: 3.52 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.1139622, -5.3391218, -8.1740389, -5.3803868, -1.8996944, 2.0218837
1: -9.1145782, -6.2016153, -9.2274456, -6.2044249, -2.6687498, 2.7504964
2: -9.9075317, -6.9973192, -9.9623280, -6.9684019, -2.3101387, 2.3639851
3: -10.8738174, -8.3505964, -10.8309956, -8.2649622, -1.9641814, 1.8476839
4: -5.5101018, -3.5100796, -5.5542521, -3.5148377, -1.5950670, 1.6419873
5: -8.8811913, -6.1869326, -8.8868237, -6.1921759, -1.5910897, 1.6023924
6: -12.9081459, -9.7116623, -12.9778900, -9.7512283, -1.9884076, 2.1050286
7: 0.3577180, 2.7339430, 0.4355721, 2.8510728, -2.0235624, 1.7921901
8: -3.6522570, -0.9419827, -3.7109237, -0.9894915, -2.1850705, 2.3385220
9: 0.2668936, 2.2653403, 0.1626736, 2.2671447, -1.9393930, 2.0669284

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0056974, upper bound: 0.9916032
time: 3.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0056974, upper bound: 0.9916032
time: 3.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.1563854, -5.3725295, -8.1657238, -5.3779998, -1.9244733, 1.9686728
1: -9.2175360, -6.2127151, -9.2257586, -6.2208891, -2.6446085, 2.7660174
2: -9.9432602, -6.9610167, -9.9391594, -6.9665422, -2.3311830, 2.3283186
3: -10.8305378, -8.2782297, -10.8301706, -8.2667694, -1.9609761, 1.8527300
4: -5.5515904, -3.5151625, -5.5552292, -3.5171974, -1.6270752, 1.6181331
5: -8.8848171, -6.1922450, -8.8832455, -6.1919756, -1.5946045, 1.5955997
6: -12.9640589, -9.7516232, -12.9709625, -9.7532539, -2.0408783, 2.0659685
7: 0.4269991, 2.8193116, 0.4491997, 2.8368356, -1.9612608, 1.8223917
8: -3.6968007, -0.9911561, -3.7104492, -0.9946251, -2.2168102, 2.3147807
9: 0.1674142, 2.2637248, 0.1650854, 2.2624879, -1.9830050, 2.0415869

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0000726, upper bound: 0.9828385
time: 3.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0311077, upper bound: 1.0224095
time: 3.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1574202, -5.3736181, -8.1740389, -5.3803868, -1.9249773, 1.9979811
1: -9.2175760, -6.2071295, -9.2274456, -6.2044249, -2.6816649, 2.7760596
2: -9.9471817, -6.9618435, -9.9623280, -6.9684019, -2.3392763, 2.3796954
3: -10.8310823, -8.2780609, -10.8309956, -8.2649622, -1.9626293, 1.8529789
4: -5.5508451, -3.5139492, -5.5542521, -3.5148377, -1.6362004, 1.6206365
5: -8.8856544, -6.1922679, -8.8868237, -6.1921759, -1.5938783, 1.6054325
6: -12.9644604, -9.7508850, -12.9778900, -9.7512283, -2.0433717, 2.0791442
7: 0.4203987, 2.8208578, 0.4355721, 2.8510728, -2.0056663, 1.8266373
8: -3.7000813, -0.9936533, -3.7109237, -0.9894915, -2.2306280, 2.3528252
9: 0.1676426, 2.2650800, 0.1626736, 2.2671447, -1.9880900, 2.0638213

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0056974, upper bound: 0.9916032
time: 3.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0338659, upper bound: 1.0286871
time: 4.16 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.94 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9928398, upper bound: 0.9808605
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9928398, upper bound: 0.9808605
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9896252
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9896253
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9928398, upper bound: 0.9808605
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0240459, upper bound: 1.0205391
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9896253
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0268038, upper bound: 1.0268046
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9928397, upper bound: 0.9863948
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9928397, upper bound: 0.9863941
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9951599
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9951596
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9928397, upper bound: 0.9863941
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0240459, upper bound: 1.0260763
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9951596
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0268039, upper bound: 1.0323418
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9983771, upper bound: 0.9808604
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9983771, upper bound: 0.9808605
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0040017, upper bound: 0.9896251
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0040017, upper bound: 0.9896251
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -0.9983771, upper bound: 0.9808605
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0295831, upper bound: 1.0205390
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0040017, upper bound: 0.9896251
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0323403, upper bound: 1.0268047
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0000726, upper bound: 0.9828385
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0000726, upper bound: 0.9828385
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0056974, upper bound: 0.9916032
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0056974, upper bound: 0.9916032
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0000726, upper bound: 0.9828385
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0311077, upper bound: 1.0224095
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0056974, upper bound: 0.9916032
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.94
Output dim: 7, lower bound: -1.0338659, upper bound: 1.0286871

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.0774212, -5.3410282, -8.0997047, -5.3864698, -1.8746314, 1.9459138
1: -9.1012020, -6.2079577, -9.1619177, -6.2377377, -2.5949421, 2.6301208
2: -9.8916731, -7.0308886, -9.9045763, -7.0254879, -2.2616558, 2.2765303
3: -10.8603325, -8.3695946, -10.8062315, -8.3338385, -1.8693953, 1.8010528
4: -5.5002260, -3.5368199, -5.5229120, -3.5497594, -1.5669408, 1.6034260
5: -8.8562021, -6.1959600, -8.8504658, -6.2015972, -1.5680432, 1.5698295
6: -12.9008265, -9.7182255, -12.9347706, -9.7607231, -1.9692650, 2.0330853
7: 0.3875179, 2.7263870, 0.5020494, 2.7802234, -1.9042668, 1.7617908
8: -3.6383677, -0.9676361, -3.6636119, -1.0370736, -2.1367083, 2.2100456
9: 0.2698810, 2.2579274, 0.2205977, 2.2501810, -1.9232373, 1.9894428

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9928398, upper bound: 0.9808605
time: 3.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9928398, upper bound: 0.9808605
time: 3.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1264572, -5.3859043, -1.9022422, 1.9513760
1: -9.2042723, -6.2131233, -9.2114172, -6.2293577, -2.6039324, 2.7320223
2: -9.9321079, -6.9953933, -9.9228325, -7.0086541, -2.3052349, 2.3104768
3: -10.8173695, -8.2978544, -10.8153305, -8.2880392, -1.9207940, 1.8128746
4: -5.5411634, -3.5408163, -5.5428314, -3.5453486, -1.6110120, 1.5988750
5: -8.8605442, -6.2012568, -8.8568869, -6.2011023, -1.5767951, 1.5773444
6: -12.9573345, -9.7571383, -12.9623785, -9.7602797, -2.0206041, 2.0488343
7: 0.4528751, 2.8134508, 0.4954782, 2.8258114, -1.9584475, 1.7985210
8: -3.6860209, -1.0203171, -3.6927991, -1.0278463, -2.1879811, 2.2887285
9: 0.1713096, 2.2579260, 0.1741972, 2.2550156, -1.9768367, 2.0344868

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9928398, upper bound: 0.9808605
time: 3.49 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9928398, upper bound: 0.9808605
time: 3.43 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.0774212, -5.3410282, -8.1069078, -5.3888545, -1.8759727, 1.9684587
1: -9.1012020, -6.2079577, -9.1635952, -6.2211266, -2.6108522, 2.6303153
2: -9.8916731, -7.0308886, -9.9274359, -7.0272870, -2.2669849, 2.3177772
3: -10.8603325, -8.3695946, -10.8070993, -8.3306084, -1.8698483, 1.8007760
4: -5.5002260, -3.5368199, -5.5224175, -3.5473707, -1.5708947, 1.6066070
5: -8.8562021, -6.1959600, -8.8541508, -6.2017918, -1.5674672, 1.5759959
6: -12.9008265, -9.7182255, -12.9416685, -9.7587290, -1.9673266, 2.0424383
7: 0.3875179, 2.7263870, 0.4883223, 2.7938151, -1.9354658, 1.7623014
8: -3.6383677, -0.9676361, -3.6598425, -1.0326567, -2.1470599, 2.2444162
9: 0.2698810, 2.2579274, 0.2188430, 2.2548800, -1.9279141, 2.0075879

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9928398, upper bound: 0.9808605
time: 3.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9984645, upper bound: 0.9896252
time: 3.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1346645, -5.3882909, -1.9020443, 1.9755802
1: -9.2042723, -6.2131233, -9.2131090, -6.2128825, -2.6375313, 2.7331934
2: -9.9321079, -6.9953933, -9.9459782, -7.0105171, -2.3101850, 2.3537941
3: -10.8173695, -8.2978544, -10.8161516, -8.2862291, -1.9217739, 1.8129368
4: -5.5411634, -3.5408163, -5.5418587, -3.5429926, -1.6169925, 1.6008072
5: -8.8605442, -6.2012568, -8.8604479, -6.2013016, -1.5762324, 1.5860772
6: -12.9573345, -9.7571383, -12.9695978, -9.7582579, -2.0231557, 2.0588808
7: 0.4528751, 2.8134508, 0.4818535, 2.8400831, -1.9909506, 1.8014045
8: -3.6860209, -1.0203171, -3.6935215, -1.0227137, -2.1993017, 2.3224311
9: 0.1713096, 2.2579260, 0.1717920, 2.2596722, -1.9819174, 2.0512252

Time for backsubstitution: 14.60 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.0185980796813965
rel_dist={7: [-1.0634246550232, 1.0634268200891506]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7908457, upper bound: 0.7927001
time: 8.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7927020, upper bound: 0.7927021
time: 4.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.63 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.63
Output dim: 7, lower bound: -0.7908457, upper bound: 0.7927001
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.63
Output dim: 7, lower bound: -0.7927020, upper bound: 0.7927021

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.1321526, -5.3759112, -8.1438475, -5.3693533, -1.7392712, 1.7367940
1: -9.2126160, -6.2120185, -9.2193661, -6.2081175, -2.5798807, 2.5805340
2: -9.9340887, -6.9920025, -9.9449568, -6.9793901, -2.1701241, 2.1695905
3: -10.8188086, -8.2860575, -10.8242207, -8.2769356, -1.7807388, 1.7751293
4: -5.5464420, -3.5399156, -5.5527210, -3.5308518, -1.4664822, 1.4643035
5: -8.8613005, -6.2009053, -8.8692360, -6.1943703, -1.3987799, 1.3996751
6: -12.9650011, -9.7570105, -12.9680328, -9.7536488, -1.8219295, 1.8212924
7: 0.4514637, 2.8336329, 0.4370689, 2.8406568, -1.8311238, 1.8380723
8: -3.7049961, -1.0193210, -3.7145891, -1.0091510, -2.1191883, 2.1241274
9: 0.1637008, 2.2586241, 0.1598777, 2.2622950, -1.9713230, 1.9715652

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7908418, upper bound: 0.7908454
time: 4.18 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7908418, upper bound: 0.7927058
time: 4.45 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.1700544, -5.3680086, -8.1700554, -5.3680096, -1.7506766, 1.7776303
1: -9.2260036, -6.2035923, -9.2260036, -6.2035933, -2.5969296, 2.6175146
2: -9.9503212, -6.9502831, -9.9503222, -6.9502802, -2.2148056, 2.1843901
3: -10.8334780, -8.2661514, -10.8334799, -8.2661505, -1.8062472, 1.8109612
4: -5.5582314, -3.5118756, -5.5582309, -3.5118744, -1.4974518, 1.4766459
5: -8.8875713, -6.1918206, -8.8875732, -6.1918216, -1.4109087, 1.4276528
6: -12.9723415, -9.7499933, -12.9723434, -9.7499933, -1.8341618, 1.8378420
7: 0.4052863, 2.8421242, 0.4052849, 2.8421261, -1.8460064, 1.8785605
8: -3.7202139, -0.9862318, -3.7202153, -0.9862299, -2.1606874, 2.1421752
9: 0.1555148, 2.2660573, 0.1555160, 2.2660570, -1.9856052, 1.9827895

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7927058, upper bound: 0.7908399
time: 10.40 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7927058, upper bound: 0.7927003
time: 6.44 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 31.44 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 31.44
Output dim: 7, lower bound: -0.7908418, upper bound: 0.7908454
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.44
Output dim: 7, lower bound: -0.7908418, upper bound: 0.7927058
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.44
Output dim: 7, lower bound: -0.7927058, upper bound: 0.7908399
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.44
Output dim: 7, lower bound: -0.7927058, upper bound: 0.7927003

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.1321526, -5.3759112, -8.1321526, -5.3759112, -1.7310824, 1.7310824
1: -9.2126160, -6.2120185, -9.2126160, -6.2120185, -2.5709095, 2.5709095
2: -9.9340887, -6.9920025, -9.9340887, -6.9920025, -2.1576724, 2.1576724
3: -10.8188086, -8.2860575, -10.8188086, -8.2860575, -1.7704926, 1.7704926
4: -5.5464420, -3.5399156, -5.5464420, -3.5399156, -1.4575620, 1.4575620
5: -8.8613005, -6.2009053, -8.8613005, -6.2009053, -1.3917542, 1.3917544
6: -12.9650011, -9.7570105, -12.9650011, -9.7570105, -1.8173075, 1.8173075
7: 0.4514637, 2.8336329, 0.4514637, 2.8336329, -1.8236742, 1.8236747
8: -3.7049961, -1.0193210, -3.7049961, -1.0193210, -2.1136823, 2.1136823
9: 0.1637008, 2.2586241, 0.1637008, 2.2586241, -1.9672709, 1.9672709

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7762763, upper bound: 0.7745390
time: 4.25 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7846235, upper bound: 0.7846221
time: 7.48 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.1321526, -5.3759112, -8.1690254, -5.3682451, -1.7404966, 1.7609546
1: -9.2126160, -6.2120185, -9.2255898, -6.2057867, -2.5775776, 2.5856433
2: -9.9340887, -6.9920025, -9.9497595, -6.9531641, -2.1946173, 2.1741662
3: -10.8188086, -8.2860575, -10.8331442, -8.2686729, -1.7894135, 1.7850640
4: -5.5464420, -3.5399156, -5.5577683, -3.5126173, -1.4839115, 1.4695258
5: -8.8613005, -6.2009053, -8.8872538, -6.1932797, -1.3998885, 1.4172764
6: -12.9650011, -9.7570105, -12.9721251, -9.7507410, -1.8242521, 1.8257382
7: 0.4514637, 2.8336329, 0.4056778, 2.8413494, -1.8320527, 1.8463194
8: -3.7049961, -1.0193210, -3.7187366, -0.9864254, -2.1347027, 2.1275301
9: 0.1637008, 2.2586241, 0.1566916, 2.2655618, -1.9742832, 1.9741464

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7762763, upper bound: 0.7763606
time: 4.25 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7846235, upper bound: 0.7864839
time: 6.94 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.1690254, -5.3682451, -8.1321526, -5.3759112, -1.7609549, 1.7404966
1: -9.2255898, -6.2057867, -9.2126160, -6.2120185, -2.5856428, 2.5775771
2: -9.9497595, -6.9531641, -9.9340887, -6.9920025, -2.1741662, 2.1946173
3: -10.8331442, -8.2686729, -10.8188086, -8.2860575, -1.7850637, 1.7894137
4: -5.5577683, -3.5126173, -5.5464420, -3.5399156, -1.4695258, 1.4839115
5: -8.8872538, -6.1932797, -8.8613005, -6.2009053, -1.4172764, 1.3998885
6: -12.9721251, -9.7507410, -12.9650011, -9.7570105, -1.8257380, 1.8242521
7: 0.4056778, 2.8413494, 0.4514637, 2.8336329, -1.8463197, 1.8320527
8: -3.7187366, -0.9864254, -3.7049961, -1.0193210, -2.1275306, 2.1347024
9: 0.1566916, 2.2655618, 0.1637008, 2.2586241, -1.9741464, 1.9742832

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7762759, upper bound: 0.7745373
time: 6.30 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7864845, upper bound: 0.7846223
time: 9.59 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.1700544, -5.3680086, -8.1700544, -5.3680086, -1.7506762, 1.7506762
1: -9.2260036, -6.2035923, -9.2260036, -6.2035923, -2.6175137, 2.6175132
2: -9.9503212, -6.9502831, -9.9503212, -6.9502831, -2.1843896, 2.1843896
3: -10.8334780, -8.2661514, -10.8334780, -8.2661514, -1.8109598, 1.8109601
4: -5.5582314, -3.5118756, -5.5582314, -3.5118756, -1.4766459, 1.4766459
5: -8.8875713, -6.1918206, -8.8875713, -6.1918206, -1.4109092, 1.4109089
6: -12.9723415, -9.7499933, -12.9723415, -9.7499933, -1.8378410, 1.8378410
7: 0.4052863, 2.8421242, 0.4052863, 2.8421242, -1.8460064, 1.8460064
8: -3.7202139, -0.9862318, -3.7202139, -0.9862318, -2.1421747, 2.1421742
9: 0.1555148, 2.2660573, 0.1555148, 2.2660573, -1.9856048, 1.9856048

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7780983, upper bound: 0.7745375
time: 4.70 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7864852, upper bound: 0.7846235
time: 5.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.38 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 25.38
Output dim: 7, lower bound: -0.7762763, upper bound: 0.7745390
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.38
Output dim: 7, lower bound: -0.7846235, upper bound: 0.7846221
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 25.38
Output dim: 7, lower bound: -0.7762763, upper bound: 0.7763606
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.38
Output dim: 7, lower bound: -0.7846235, upper bound: 0.7864839
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 25.38
Output dim: 7, lower bound: -0.7762759, upper bound: 0.7745373
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.38
Output dim: 7, lower bound: -0.7864845, upper bound: 0.7846223
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 25.38
Output dim: 7, lower bound: -0.7780983, upper bound: 0.7745375
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.38
Output dim: 7, lower bound: -0.7864852, upper bound: 0.7846235

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1242085, -5.3759174, -1.6908092, 1.7233725
1: -9.2042723, -6.2131233, -9.2070360, -6.2127581, -2.4275079, 2.5521364
2: -9.9321079, -6.9953933, -9.9327641, -6.9942718, -2.1522832, 2.1422849
3: -10.8173695, -8.2978544, -10.8178425, -8.2939482, -1.7642412, 1.6611755
4: -5.5411634, -3.5408163, -5.5429106, -3.5405183, -1.4564233, 1.4486632
5: -8.8605442, -6.2012568, -8.8607922, -6.2011423, -1.3870225, 1.3906341
6: -12.9573345, -9.7571383, -12.9597015, -9.7570934, -1.7845459, 1.8060534
7: 0.4528751, 2.8134508, 0.4524088, 2.8193593, -1.8174109, 1.7084126
8: -3.6860209, -1.0203171, -3.6916680, -1.0199876, -2.0282006, 2.1085920
9: 0.1713096, 2.2579260, 0.1690652, 2.2581570, -1.9051771, 1.9620738

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7804149, upper bound: 0.7782577
time: 4.69 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7815085, upper bound: 0.7815077
time: 4.90 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1610603, -5.3682504, -1.7002568, 1.7535784
1: -9.2042723, -6.2131233, -9.2199945, -6.2065229, -2.4342546, 2.5668778
2: -9.9321079, -6.9953933, -9.9484482, -6.9554334, -2.1889505, 2.1588612
3: -10.8173695, -8.2978544, -10.8321743, -8.2765656, -1.7831640, 1.6758318
4: -5.5411634, -3.5408163, -5.5542359, -3.5132205, -1.4826627, 1.4606295
5: -8.8605442, -6.2012568, -8.8867455, -6.1935167, -1.3951573, 1.4145396
6: -12.9573345, -9.7571383, -12.9668016, -9.7508259, -1.7931294, 1.8144908
7: 0.4528751, 2.8134508, 0.4066353, 2.8270431, -1.8257942, 1.7329538
8: -3.6860209, -1.0203171, -3.7055116, -0.9870949, -2.0477376, 2.1224465
9: 0.1713096, 2.2579260, 0.1620153, 2.2651019, -1.9122829, 1.9689007

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7804148, upper bound: 0.7801444
time: 4.70 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7815084, upper bound: 0.7833712
time: 4.70 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.1572819, -5.3682537, -8.1242085, -5.3759174, -1.7205687, 1.7327833
1: -9.2172356, -6.2068872, -9.2070360, -6.2127581, -2.4418912, 2.5587721
2: -9.9477959, -6.9565554, -9.9327641, -6.9942718, -2.1687927, 2.1786094
3: -10.8316946, -8.2804689, -10.8178425, -8.2939482, -1.7788019, 1.6792057
4: -5.5524883, -3.5135200, -5.5429106, -3.5405183, -1.4681072, 1.4749875
5: -8.8864937, -6.1936321, -8.8607922, -6.2011423, -1.4125280, 1.3987632
6: -12.9644318, -9.7508659, -12.9597015, -9.7570934, -1.7927256, 1.8129981
7: 0.4071054, 2.8211405, 0.4524088, 2.8193593, -1.8400784, 1.7167349
8: -3.6998682, -0.9874277, -3.6916680, -1.0199876, -2.0420060, 2.1296487
9: 0.1642604, 2.2648733, 0.1690652, 2.2581570, -1.9119272, 1.9690933

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7823008, upper bound: 0.7782526
time: 5.27 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7833711, upper bound: 0.7815112
time: 4.36 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.1583109, -5.3680158, -8.1620913, -5.3680143, -1.7103944, 1.7429562
1: -9.2176418, -6.2046900, -9.2204046, -6.2043295, -2.4738021, 2.5986996
2: -9.9483604, -6.9536762, -9.9490099, -6.9525518, -2.1790180, 2.1690340
3: -10.8320303, -8.2779474, -10.8325090, -8.2740440, -1.8046966, 1.7007785
4: -5.5529518, -3.5127769, -5.5546989, -3.5124776, -1.4756322, 1.4677243
5: -8.8868132, -6.1921716, -8.8870659, -6.1920552, -1.4061594, 1.4098213
6: -12.9646530, -9.7501202, -12.9670238, -9.7500782, -1.8063760, 1.8265848
7: 0.4067149, 2.8217978, 0.4062428, 2.8277011, -1.8396931, 1.7312756
8: -3.7013464, -0.9872322, -3.7070031, -0.9869022, -2.0566769, 2.1370969
9: 0.1630850, 2.2653689, 0.1608398, 2.2655969, -1.9234762, 1.9803462

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7823015, upper bound: 0.7782529
time: 5.40 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7833719, upper bound: 0.7815114
time: 4.47 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.60 seconds
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.7804149, upper bound: 0.7782577
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.7815085, upper bound: 0.7815077
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.7804148, upper bound: 0.7801444
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.7815084, upper bound: 0.7833712
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.7823008, upper bound: 0.7782526
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.7833711, upper bound: 0.7815112
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.7823015, upper bound: 0.7782529
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.7833719, upper bound: 0.7815114

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.1171379, -5.3836794, -8.1278324, -5.3859034, -1.6750312, 1.7150097
1: -9.2040901, -6.2266092, -9.2123814, -6.2292414, -2.3868504, 2.5119920
2: -9.9234972, -7.0080194, -9.9230490, -7.0082626, -2.1251736, 2.1154470
3: -10.8148022, -8.2983351, -10.8154984, -8.2866716, -1.7660084, 1.6587567
4: -5.5388260, -3.5449216, -5.5434432, -3.5452466, -1.4432125, 1.4421520
5: -8.8571625, -6.2013798, -8.8569717, -6.2010608, -1.3794732, 1.3801756
6: -12.9563265, -9.7596760, -12.9636412, -9.7602673, -1.7723947, 1.8005102
7: 0.4875984, 2.8092122, 0.4953136, 2.8283672, -1.7798200, 1.6587062
8: -3.6783128, -1.0270600, -3.6953039, -1.0277319, -2.0055280, 2.0925956
9: 0.1785997, 2.2551515, 0.1732543, 2.2550933, -1.8606534, 1.9208393

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7698522, upper bound: 0.7659550
time: 4.30 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7804149, upper bound: 0.7782577
time: 4.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1189594, -5.3855314, -8.1359806, -5.3882895, -1.6773109, 1.7481222
1: -9.2041607, -6.2172370, -9.2140770, -6.2127647, -2.4221721, 2.5286903
2: -9.9301319, -7.0094061, -9.9462013, -7.0101213, -2.1368694, 2.1720481
3: -10.8157463, -8.2980461, -10.8163214, -8.2848644, -1.7681470, 1.6586378
4: -5.5375481, -3.5428033, -5.5424709, -3.5428865, -1.4541435, 1.4450750
5: -8.8585978, -6.2014251, -8.8605442, -6.2012596, -1.3786612, 1.3905094
6: -12.9570160, -9.7584114, -12.9705677, -9.7582445, -1.7749128, 1.8159821
7: 0.4760098, 2.8118625, 0.4816909, 2.8425953, -1.8342886, 1.6626959
8: -3.6838837, -1.0313354, -3.6956553, -1.0225992, -2.0233655, 2.1336150
9: 0.1790102, 2.2574463, 0.1708477, 2.2597508, -1.8644996, 1.9452577

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7729494, upper bound: 0.7702977
time: 4.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7815085, upper bound: 0.7815117
time: 4.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1189594, -5.3855314, -8.1730261, -5.3806233, -1.6867619, 1.7765446
1: -9.2041607, -6.2172370, -9.2270384, -6.2066035, -2.4288783, 2.5433183
2: -9.9301319, -7.0094061, -9.9617815, -6.9712844, -2.1739388, 2.1886172
3: -10.8157463, -8.2980461, -10.8306580, -8.2674837, -1.7870612, 1.6733010
4: -5.5375481, -3.5428033, -5.5537877, -3.5155802, -1.4794433, 1.4570351
5: -8.8585978, -6.2014251, -8.8865051, -6.1936345, -1.3868032, 1.4143610
6: -12.9570160, -9.7584114, -12.9776716, -9.7519741, -1.7834969, 1.8244112
7: 0.4760098, 2.8118625, 0.4359670, 2.8502998, -1.8426576, 1.6869137
8: -3.6838837, -1.0313354, -3.7094388, -0.9896870, -2.0430794, 2.1474271
9: 0.1790102, 2.2574463, 0.1638482, 2.2666557, -1.8714223, 1.9519882

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7729495, upper bound: 0.7721250
time: 4.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7815084, upper bound: 0.7833751
time: 4.47 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.1539707, -5.3760118, -8.1278324, -5.3859034, -1.7041874, 1.7244196
1: -9.2170506, -6.2204208, -9.2123814, -6.2292414, -2.4012337, 2.5185642
2: -9.9391088, -6.9691820, -9.9230490, -7.0082626, -2.1416349, 2.1524453
3: -10.8291292, -8.2809544, -10.8154984, -8.2866716, -1.7805648, 1.6767817
4: -5.5501485, -3.5176158, -5.5434432, -3.5452466, -1.4549961, 1.4684958
5: -8.8831148, -6.1937532, -8.8569717, -6.2010608, -1.4047780, 1.3883076
6: -12.9634094, -9.7534056, -12.9636412, -9.7602673, -1.7805729, 1.8074491
7: 0.4418921, 2.8168850, 0.4953136, 2.8283672, -1.8026850, 1.6670141
8: -3.6921101, -0.9941654, -3.6953039, -1.0277319, -2.0192742, 2.1138542
9: 0.1715095, 2.2620711, 0.1732543, 2.2550933, -1.8673782, 1.9278541

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7698519, upper bound: 0.7659548
time: 4.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7823008, upper bound: 0.7782528
time: 5.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1557932, -5.3778653, -8.1359806, -5.3882895, -1.7070093, 1.7575336
1: -9.2171211, -6.2110043, -9.2140770, -6.2127647, -2.4365563, 2.5353165
2: -9.9458103, -6.9705696, -9.9462013, -7.0101213, -2.1533704, 2.2051730
3: -10.8300705, -8.2806616, -10.8163214, -8.2848644, -1.7827015, 1.6766679
4: -5.5488749, -3.5155029, -5.5424709, -3.5428865, -1.4658284, 1.4713945
5: -8.8845482, -6.1937943, -8.8605442, -6.2012596, -1.4039946, 1.3986404
6: -12.9641132, -9.7521439, -12.9705677, -9.7582445, -1.7831068, 1.8229239
7: 0.4303064, 2.8195508, 0.4816909, 2.8425953, -1.8586214, 1.6710210
8: -3.6977229, -0.9984298, -3.6956553, -1.0225992, -2.0371633, 2.1565111
9: 0.1719699, 2.2643905, 0.1708477, 2.2597508, -1.8712144, 1.9522867

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7748122, upper bound: 0.7702977
time: 4.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7833711, upper bound: 0.7815117
time: 4.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.1549959, -5.3757758, -8.1657238, -5.3779998, -1.6945810, 1.7345738
1: -9.2174587, -6.2182355, -9.2257586, -6.2208891, -2.4330740, 2.5583744
2: -9.9396544, -6.9663014, -9.9391594, -6.9665422, -2.1518559, 2.1421132
3: -10.8294621, -8.2784338, -10.8301706, -8.2667694, -1.8064661, 1.6983688
4: -5.5506139, -3.5168757, -5.5552292, -3.5171974, -1.4625177, 1.4612522
5: -8.8834324, -6.1922922, -8.8832455, -6.1919756, -1.3986149, 1.3993645
6: -12.9636269, -9.7526617, -12.9709625, -9.7532539, -1.7941976, 1.8210516
7: 0.4415021, 2.8175378, 0.4491997, 2.8368356, -1.8023605, 1.6815529
8: -3.6935816, -0.9939709, -3.7104492, -0.9946251, -2.0340347, 2.1211123
9: 0.1703370, 2.2625625, 0.1650854, 2.2624879, -1.8787832, 1.9390612

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7719135, upper bound: 0.7659547
time: 4.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7823015, upper bound: 0.7782525
time: 5.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1568193, -5.3776298, -8.1740389, -5.3803868, -1.6968966, 1.7676187
1: -9.2175293, -6.2088127, -9.2274456, -6.2044249, -2.4684224, 2.5751410
2: -9.9463673, -6.9676871, -9.9623280, -6.9684019, -2.1635938, 2.1988015
3: -10.8304024, -8.2781410, -10.8309956, -8.2649622, -1.8086047, 1.6982491
4: -5.5493383, -3.5147612, -5.5542521, -3.5148377, -1.4733529, 1.4641280
5: -8.8848696, -6.1923370, -8.8868237, -6.1921759, -1.3978033, 1.4096332
6: -12.9643364, -9.7513971, -12.9778900, -9.7512283, -1.7967596, 1.8365471
7: 0.4299140, 2.8202066, 0.4355721, 2.8510728, -1.8565483, 1.6857047
8: -3.6992035, -0.9982343, -3.7109237, -0.9894915, -2.0518236, 2.1622195
9: 0.1707970, 2.2648854, 0.1626736, 2.2671447, -1.8826461, 1.9634447

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7751180, upper bound: 0.7707149
time: 4.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7833719, upper bound: 0.7815090
time: 4.54 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.90 seconds
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7698522, upper bound: 0.7659550
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7804149, upper bound: 0.7782577
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7729494, upper bound: 0.7702977
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7815085, upper bound: 0.7815117
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7729495, upper bound: 0.7721250
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7815084, upper bound: 0.7833751
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7698519, upper bound: 0.7659548
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7823008, upper bound: 0.7782528
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7748122, upper bound: 0.7702977
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7833711, upper bound: 0.7815117
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7719135, upper bound: 0.7659547
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7823015, upper bound: 0.7782525
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7751180, upper bound: 0.7707149
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.90
Output dim: 7, lower bound: -0.7833719, upper bound: 0.7815090

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1199322, -5.3859081, -1.6773944, 1.7205534
1: -9.2042723, -6.2131233, -9.2067995, -6.2299166, -2.3979807, 2.5346866
2: -9.9321079, -6.9953933, -9.9217920, -7.0105305, -2.1285872, 2.1338620
3: -10.8173695, -8.2978544, -10.8145380, -8.2945633, -1.7636461, 1.6584091
4: -5.5411634, -3.5408163, -5.5399108, -3.5458331, -1.4511166, 1.4397082
5: -8.8605442, -6.2012568, -8.8564739, -6.2012978, -1.3821764, 1.3835204
6: -12.9573345, -9.7571383, -12.9584217, -9.7603436, -1.7751279, 1.8021245
7: 0.4528751, 2.8134508, 0.4962621, 2.8138552, -1.8140125, 1.6613560
8: -3.6860209, -1.0203171, -3.6822262, -1.0283875, -2.0114307, 2.1006637
9: 0.1713096, 2.2579260, 0.1783473, 2.2546439, -1.8797021, 1.9370203

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7804149, upper bound: 0.7782559
time: 4.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7804149, upper bound: 0.7782536
time: 5.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1283751, -5.3882971, -1.6784649, 1.7447438
1: -9.2042723, -6.2131233, -9.2084942, -6.2134495, -2.4271307, 2.5361519
2: -9.9321079, -6.9953933, -9.9449120, -7.0124087, -2.1348691, 2.1771193
3: -10.8173695, -8.2978544, -10.8153439, -8.2926216, -1.7646246, 1.6579592
4: -5.5411634, -3.5408163, -5.5389366, -3.5434954, -1.4566679, 1.4414663
5: -8.8605442, -6.2012568, -8.8599987, -6.2014956, -1.3816142, 1.3918693
6: -12.9573345, -9.7571383, -12.9657345, -9.7583227, -1.7778306, 1.8122163
7: 0.4528751, 2.8134508, 0.4826341, 2.8280730, -1.8467975, 1.6632643
8: -3.6860209, -1.0203171, -3.6833334, -1.0232644, -2.0248966, 2.1333995
9: 0.1713096, 2.2579260, 0.1761668, 2.2592940, -1.8847766, 1.9516869

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7804149, upper bound: 0.7782536
time: 5.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7815085, upper bound: 0.7815078
time: 4.92 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.1204453, -5.3759203, -8.1654034, -5.3806286, -1.6879134, 1.7736256
1: -9.2042723, -6.2131233, -9.2214565, -6.2072854, -2.4338241, 2.5507603
2: -9.9321079, -6.9953933, -9.9605007, -6.9735708, -2.1716847, 2.1936994
3: -10.8173695, -8.2978544, -10.8296738, -8.2751112, -1.7835379, 1.6726151
4: -5.5411634, -3.5408163, -5.5502524, -3.5161881, -1.4828477, 1.4534268
5: -8.8605442, -6.2012568, -8.8859615, -6.1938705, -1.3897572, 1.4156811
6: -12.9573345, -9.7571383, -12.9727736, -9.7520523, -1.7864151, 1.8206518
7: 0.4528751, 2.8134508, 0.4369206, 2.8357759, -1.8551674, 1.6874781
8: -3.6860209, -1.0203171, -3.6971126, -0.9903536, -2.0446649, 2.1472144
9: 0.1713096, 2.2579260, 0.1691271, 2.2662041, -1.8917046, 1.9584541

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7804148, upper bound: 0.7801444
time: 4.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7815084, upper bound: 0.7833751
time: 4.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.1572819, -5.3682537, -8.1199322, -5.3859081, -1.7067223, 1.7299643
1: -9.2172356, -6.2068872, -9.2067995, -6.2299166, -2.4123640, 2.5413222
2: -9.9477959, -6.9565554, -9.9217920, -7.0105305, -2.1450968, 2.1701341
3: -10.8316946, -8.2804689, -10.8145380, -8.2945633, -1.7782059, 1.6764395
4: -5.5524883, -3.5135200, -5.5399108, -3.5458331, -1.4628000, 1.4660325
5: -8.8864937, -6.1936321, -8.8564739, -6.2012978, -1.4075937, 1.3916497
6: -12.9644318, -9.7508659, -12.9584217, -9.7603436, -1.7833080, 1.8090687
7: 0.4071054, 2.8211405, 0.4962621, 2.8138552, -1.8366494, 1.6696782
8: -3.6998682, -0.9874277, -3.6822262, -1.0283875, -2.0252371, 2.1216769
9: 0.1642604, 2.2648733, 0.1783473, 2.2546439, -1.8864527, 1.9440398

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7823008, upper bound: 0.7782526
time: 5.47 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7823008, upper bound: 0.7782526
time: 5.50 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.1572819, -5.3682537, -8.1283751, -5.3882971, -1.7082062, 1.7541547
1: -9.2172356, -6.2068872, -9.2084942, -6.2134495, -2.4415131, 2.5427871
2: -9.9477959, -6.9565554, -9.9449120, -7.0124087, -2.1513786, 2.2115746
3: -10.8316946, -8.2804689, -10.8153439, -8.2926216, -1.7791843, 1.6759894
4: -5.5524883, -3.5135200, -5.5389366, -3.5434954, -1.4683514, 1.4677906
5: -8.8864937, -6.1936321, -8.8599987, -6.2014956, -1.4069839, 1.3999984
6: -12.9644318, -9.7508659, -12.9657345, -9.7583227, -1.7860107, 1.8191609
7: 0.4071054, 2.8211405, 0.4826341, 2.8280730, -1.8713369, 1.6715865
8: -3.6998682, -0.9874277, -3.6833334, -1.0232644, -2.0387030, 2.1557269
9: 0.1642604, 2.2648733, 0.1761668, 2.2592940, -1.8915272, 1.9587064

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7823008, upper bound: 0.7782527
time: 5.57 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7833711, upper bound: 0.7815097
time: 4.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.1583109, -5.3680158, -8.1577940, -5.3780050, -1.6969800, 1.7401180
1: -9.2176418, -6.2046900, -9.2201681, -6.2215605, -2.4441938, 2.5812345
2: -9.9483604, -6.9536762, -9.9379158, -6.9688144, -2.1553211, 2.1605392
3: -10.8320303, -8.2779474, -10.8292036, -8.2746601, -1.8041067, 1.6980081
4: -5.5529518, -3.5127769, -5.5516944, -3.5177841, -1.4703155, 1.4587879
5: -8.8868132, -6.1921716, -8.8827486, -6.1922112, -1.4013233, 1.4027076
6: -12.9646530, -9.7501202, -12.9657192, -9.7533283, -1.7969494, 1.8226700
7: 0.4067149, 2.8217978, 0.4501619, 2.8222370, -1.8362589, 1.6842074
8: -3.7013464, -0.9872322, -3.6974850, -0.9952846, -2.0399389, 2.1291718
9: 0.1630850, 2.2653689, 0.1701400, 2.2620437, -1.8978577, 1.9553108

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7823015, upper bound: 0.7782523
time: 5.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7823015, upper bound: 0.7782524
time: 5.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.1583109, -5.3680158, -8.1664171, -5.3803911, -1.6980500, 1.7642455
1: -9.2176418, -6.2046900, -9.2218647, -6.2051058, -2.4733677, 2.5825806
2: -9.9483604, -6.9536762, -9.9610491, -6.9706907, -2.1616020, 2.2038836
3: -10.8320303, -8.2779474, -10.8300085, -8.2725906, -1.8050866, 1.6975610
4: -5.5529518, -3.5127769, -5.5507164, -3.5154464, -1.4758687, 1.4605222
5: -8.8868132, -6.1921716, -8.8862801, -6.1924114, -1.4007587, 1.4109797
6: -12.9646530, -9.7501202, -12.9730024, -9.7513075, -1.7996621, 1.8327799
7: 0.4067149, 2.8217978, 0.4365277, 2.8365505, -1.8690629, 1.6862330
8: -3.7013464, -0.9872322, -3.6985941, -0.9901595, -2.0533590, 2.1619601
9: 0.1630850, 2.2653689, 0.1679513, 2.2666936, -1.9029570, 1.9698987

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 2153
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 1949
type: B, layer: 3, pos: 2809
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 1760
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 212
type: B, layer: 3, pos: 2482
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7823015, upper bound: 0.7782524
time: 5.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7833719, upper bound: 0.7815085
time: 4.75 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 25.24 seconds
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7804149, upper bound: 0.7782559
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7804149, upper bound: 0.7782536
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7804149, upper bound: 0.7782536
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7815085, upper bound: 0.7815078
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7804148, upper bound: 0.7801444
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7815084, upper bound: 0.7833751
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7823008, upper bound: 0.7782526
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7823008, upper bound: 0.7782526
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7823008, upper bound: 0.7782527
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7833711, upper bound: 0.7815097
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7823015, upper bound: 0.7782523
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7823015, upper bound: 0.7782524
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7823015, upper bound: 0.7782524
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.24
Output dim: 7, lower bound: -0.7833719, upper bound: 0.7815085

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.1171379, -5.3836794, -8.1278324, -5.3859034, -1.6750312, 1.7150097
1: -9.2040901, -6.2266092, -9.2123814, -6.2292414, -2.3868504, 2.5119920
2: -9.9234972, -7.0080194, -9.9230490, -7.0082626, -2.1251736, 2.1154470
3: -10.8148022, -8.2983351, -10.8154984, -8.2866716, -1.7660084, 1.6587567
4: -5.5388260, -3.5449216, -5.5434432, -3.5452466, -1.4432125, 1.4421520
5: -8.8571625, -6.2013798, -8.8569717, -6.2010608, -1.3794732, 1.3801756
6: -12.9563265, -9.7596760, -12.9636412, -9.7602673, -1.7723947, 1.8005102
7: 0.4875984, 2.8092122, 0.4953136, 2.8283672, -1.7798200, 1.6587062
8: -3.6783128, -1.0270600, -3.6953039, -1.0277319, -2.0055280, 2.0925956
9: 0.1785997, 2.2551515, 0.1732543, 2.2550933, -1.8606534, 1.9208393

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7698522, upper bound: 0.7659549
time: 4.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7804149, upper bound: 0.7782559
time: 4.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1189594, -5.3855314, -8.1359806, -5.3882895, -1.6773109, 1.7481222
1: -9.2041607, -6.2172370, -9.2140770, -6.2127647, -2.4221721, 2.5286903
2: -9.9301319, -7.0094061, -9.9462013, -7.0101213, -2.1368694, 2.1720481
3: -10.8157463, -8.2980461, -10.8163214, -8.2848644, -1.7681470, 1.6586378
4: -5.5375481, -3.5428033, -5.5424709, -3.5428865, -1.4541435, 1.4450750
5: -8.8585978, -6.2014251, -8.8605442, -6.2012596, -1.3786612, 1.3905094
6: -12.9570160, -9.7584114, -12.9705677, -9.7582445, -1.7749128, 1.8159821
7: 0.4760098, 2.8118625, 0.4816909, 2.8425953, -1.8342886, 1.6626959
8: -3.6838837, -1.0313354, -3.6956553, -1.0225992, -2.0233655, 2.1336150
9: 0.1790102, 2.2574463, 0.1708477, 2.2597508, -1.8644996, 1.9452577

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7698523, upper bound: 0.7659548
time: 4.25 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7804149, upper bound: 0.7782528
time: 4.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.1171379, -5.3836794, -8.1278324, -5.3859034, -1.6750312, 1.7150097
1: -9.2040901, -6.2266092, -9.2123814, -6.2292414, -2.3868504, 2.5119920
2: -9.9234972, -7.0080194, -9.9230490, -7.0082626, -2.1251736, 2.1154470
3: -10.8148022, -8.2983351, -10.8154984, -8.2866716, -1.7660084, 1.6587567
4: -5.5388260, -3.5449216, -5.5434432, -3.5452466, -1.4432125, 1.4421520
5: -8.8571625, -6.2013798, -8.8569717, -6.2010608, -1.3794732, 1.3801756
6: -12.9563265, -9.7596760, -12.9636412, -9.7602673, -1.7723947, 1.8005102
7: 0.4875984, 2.8092122, 0.4953136, 2.8283672, -1.7798200, 1.6587062
8: -3.6783128, -1.0270600, -3.6953039, -1.0277319, -2.0055280, 2.0925956
9: 0.1785997, 2.2551515, 0.1732543, 2.2550933, -1.8606534, 1.9208393

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7698523, upper bound: 0.7659550
time: 4.27 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7804149, upper bound: 0.7782536
time: 4.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1189594, -5.3855314, -8.1359806, -5.3882895, -1.6773109, 1.7481222
1: -9.2041607, -6.2172370, -9.2140770, -6.2127647, -2.4221721, 2.5286903
2: -9.9301319, -7.0094061, -9.9462013, -7.0101213, -2.1368694, 2.1720481
3: -10.8157463, -8.2980461, -10.8163214, -8.2848644, -1.7681470, 1.6586378
4: -5.5375481, -3.5428033, -5.5424709, -3.5428865, -1.4541435, 1.4450750
5: -8.8585978, -6.2014251, -8.8605442, -6.2012596, -1.3786612, 1.3905094
6: -12.9570160, -9.7584114, -12.9705677, -9.7582445, -1.7749128, 1.8159821
7: 0.4760098, 2.8118625, 0.4816909, 2.8425953, -1.8342886, 1.6626959
8: -3.6838837, -1.0313354, -3.6956553, -1.0225992, -2.0233655, 2.1336150
9: 0.1790102, 2.2574463, 0.1708477, 2.2597508, -1.8644996, 1.9452577

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7729494, upper bound: 0.7702977
time: 4.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7815085, upper bound: 0.7815096
time: 4.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1189594, -5.3855314, -8.1730261, -5.3806233, -1.6867619, 1.7765446
1: -9.2041607, -6.2172370, -9.2270384, -6.2066035, -2.4288783, 2.5433183
2: -9.9301319, -7.0094061, -9.9617815, -6.9712844, -2.1739388, 2.1886172
3: -10.8157463, -8.2980461, -10.8306580, -8.2674837, -1.7870612, 1.6733010
4: -5.5375481, -3.5428033, -5.5537877, -3.5155802, -1.4794433, 1.4570351
5: -8.8585978, -6.2014251, -8.8865051, -6.1936345, -1.3868032, 1.4143610
6: -12.9570160, -9.7584114, -12.9776716, -9.7519741, -1.7834969, 1.8244112
7: 0.4760098, 2.8118625, 0.4359670, 2.8502998, -1.8426576, 1.6869137
8: -3.6838837, -1.0313354, -3.7094388, -0.9896870, -2.0430794, 2.1474271
9: 0.1790102, 2.2574463, 0.1638482, 2.2666557, -1.8714223, 1.9519882

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 2153
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1949
type: A, layer: 3, pos: 2809
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 1760
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 2482
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 170

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7729495, upper bound: 0.7721250
time: 4.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7815084, upper bound: 0.7833751
time: 4.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.1539707, -5.3760118, -8.1278324, -5.3859034, -1.7041874, 1.7244196
1: -9.2170506, -6.2204208, -9.2123814, -6.2292414, -2.4012337, 2.5185642
2: -9.9391088, -6.9691820, -9.9230490, -7.0082626, -2.1416349, 2.1524453
3: -10.8291292, -8.2809544, -10.8154984, -8.2866716, -1.7805648, 1.6767817
4: -5.5501485, -3.5176158, -5.5434432, -3.5452466, -1.4549961, 1.4684958
5: -8.8831148, -6.1937532, -8.8569717, -6.2010608, -1.4047780, 1.3883076
6: -12.9634094, -9.7534056, -12.9636412, -9.7602673, -1.7805729, 1.8074491
7: 0.4418921, 2.8168850, 0.4953136, 2.8283672, -1.8026850, 1.6670141
8: -3.6921101, -0.9941654, -3.6953039, -1.0277319, -2.0192742, 2.1138542
9: 0.1715095, 2.2620711, 0.1732543, 2.2550933, -1.8673782, 1.9278541

Time for backsubstitution: 14.46 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.8785624504089355
rel_dist={7: [-0.7927072900922123, 0.7927031607168487]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 2412.14 seconds
