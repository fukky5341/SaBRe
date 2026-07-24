## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.579334386
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.3477392, 4.3477392)
1: (-7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.7961807, 3.7961807)
2: (-10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807)
3: (-12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1542416, 3.1542416)
4: (5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894)
5: (-8.9787197, -5.6989894, -8.9787197, -5.6989894, -3.2797303, 3.2797303)
6: (-12.5030499, -8.9509478, -12.5030499, -8.9509478, -3.3759890, 3.3759892)
7: (-5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9533715)
8: (-1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666)
9: (-6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219)

## BASE Result
execution time: IAR + LP analysis = 14.10 + 33.79 = 47.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -2.5504220, upper bound: 2.5504211


# Binary Search by BASE starts (time budget: 3552.10 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.402289390563965
rel_dist={4: [-1.981378173882315, 1.9813780589154808]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=3.2979745864868164
rel_dist={4: [-1.6115652775740719, 1.611567303353243]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=3.17470645904541
rel_dist={4: [-1.3305644078822736, 1.3305637963562233]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=3.2363405227661133
rel_dist={4: [-1.4769317174548258, 1.4769330892355903]}

## Binary Search Result
Binary search time: 202.22 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3349.89 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0912072, upper bound: 2.0629733
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0629732, upper bound: 2.0912076
time: 4.16 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.56 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.56
Output dim: 4, lower bound: -2.0912072, upper bound: 2.0629733
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.56
Output dim: 4, lower bound: -2.0629732, upper bound: 2.0912076

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0642600, 4.0645900
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6917329, 3.6911349
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0953984, 3.0934227
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9958696, 2.9953218
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.8257203, 2.8267572
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 495

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0911767, upper bound: 2.0532332
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0534473, upper bound: 2.0404324
time: 5.22 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0645900, 4.0642591
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6911349, 3.6917324
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0934229, 3.0953984
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9953213, 2.9958696
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.8267570, 2.8257208
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 495

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0404325, upper bound: 2.0534477
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0532330, upper bound: 2.0911778
time: 4.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.85 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.85
Output dim: 4, lower bound: -2.0911767, upper bound: 2.0532332
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.85
Output dim: 4, lower bound: -2.0534473, upper bound: 2.0404324
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.85
Output dim: 4, lower bound: -2.0404325, upper bound: 2.0534477
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.85
Output dim: 4, lower bound: -2.0532330, upper bound: 2.0911778

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0734220, 4.0770006
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6273599, 3.6585665
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1323714, 3.1207170
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9885607, 2.9744177
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.8453951, 2.8412859
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0911763, upper bound: 2.0532329
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0911763, upper bound: 2.0532328
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0766702, 4.0737514
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6579404, 3.6267614
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1226931, 3.1303945
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9749651, 2.9880133
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.8402491, 2.8444171
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0534469, upper bound: 2.0404326
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0534469, upper bound: 2.0404322
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0737519, 4.0766697
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6267610, 3.6579399
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1303949, 3.1226928
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9880133, 2.9749656
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.8444166, 2.8402495
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0404321, upper bound: 2.0534474
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0404321, upper bound: 2.0534475
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0770001, 4.0734210
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6585660, 3.6273594
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1207170, 3.1323709
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9744177, 2.9885609
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.8412857, 2.8453956
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0532326, upper bound: 2.0911776
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0404342, upper bound: 2.0534470
time: 5.20 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -2.0911763, upper bound: 2.0532329
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -2.0911763, upper bound: 2.0532328
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -2.0534469, upper bound: 2.0404326
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -2.0534469, upper bound: 2.0404322
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -2.0404321, upper bound: 2.0534474
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -2.0404321, upper bound: 2.0534475
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -2.0532326, upper bound: 2.0911776
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -2.0404342, upper bound: 2.0534470

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0077553, 4.0425253
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6294184, 3.6601233
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1199150, 3.0883811
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9834490, 2.9672003
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7749386, 2.7932470
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9504776, 2.9499688
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 921

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 611

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9516099, upper bound: 1.9187650
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9516099, upper bound: 1.9187650
time: 6.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0389462, 4.0113335
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6289167, 3.6606250
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1000352, 3.1082609
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9813433, 2.9693053
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7973566, 2.7708287
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9409542, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2622

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0904439, upper bound: 2.0528817
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0908255, upper bound: 2.0525010
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0110035, 4.0392766
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6599989, 3.6283183
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1102366, 3.0980585
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9698534, 2.9807956
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7697926, 2.7963781
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9401572
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1690

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0308781, upper bound: 2.0129747
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0274632, upper bound: 2.0174776
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0421944, 4.0080853
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6594973, 3.6288199
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0903568, 3.1179383
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9677477, 2.9829006
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7922106, 2.7739599
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9491549, 2.9496799
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 417

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0510473, upper bound: 2.0384204
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0513894, upper bound: 2.0380820
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0080853, 4.0421944
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6288195, 3.6594968
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1179380, 3.0903568
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9829006, 2.9677479
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7739601, 2.7922106
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9496803, 2.9491544
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0265417, upper bound: 2.0403746
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0272973, upper bound: 2.0395802
time: 5.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0392761, 4.0110030
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6283178, 3.6599979
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0980587, 3.1102366
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9807959, 2.9698529
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7963781, 2.7697923
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9401579, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 668

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0386133, upper bound: 2.0395948
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0265943, upper bound: 2.0516191
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0113335, 4.0389462
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6606245, 3.6289163
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1082611, 3.1000352
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9693050, 2.9813433
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7708282, 2.7973566
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9409540
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1704

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2384

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9900328, upper bound: 2.0279632
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9900328, upper bound: 2.0279632
time: 4.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0425262, 4.0077543
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6601229, 3.6294174
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0883808, 3.1199150
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9672003, 2.9834485
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7932472, 2.7749383
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9499693, 2.9504769
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 317

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1396

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.8774261, upper bound: 1.9095496
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.8774261, upper bound: 1.9095496
time: 4.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -1.9516099, upper bound: 1.9187650
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -1.9516099, upper bound: 1.9187650
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -2.0904439, upper bound: 2.0528817
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -2.0908255, upper bound: 2.0525010
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -2.0308781, upper bound: 2.0129747
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -2.0274632, upper bound: 2.0174776
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -2.0510473, upper bound: 2.0384204
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -2.0513894, upper bound: 2.0380820
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -2.0265417, upper bound: 2.0403746
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -2.0272973, upper bound: 2.0395802
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -2.0386133, upper bound: 2.0395948
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -2.0265943, upper bound: 2.0516191
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -1.9900328, upper bound: 2.0279632
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -1.9900328, upper bound: 2.0279632
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -1.8774261, upper bound: 1.9095496
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 4, lower bound: -1.8774261, upper bound: 1.9095496

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0069065, 4.0453238
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6269665, 3.6575460
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1185627, 3.0746672
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9817753, 2.9579883
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7740660, 2.7931001
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9500904, 2.9492908
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 3105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2572

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9338948, upper bound: 1.9171574
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9500429, upper bound: 1.9161863
time: 4.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0077553, 4.0416784
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6294184, 3.6576724
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1199150, 3.0870285
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9834490, 2.9655266
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7747917, 2.7932470
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9497986, 2.9499688
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1839

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9307793, upper bound: 1.8883393
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9320428, upper bound: 1.8987531
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0364647, 4.0095043
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6308908, 3.6627369
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0965238, 3.1105905
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9941082, 2.9827476
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7930312, 2.7687833
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9397383, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0670684, upper bound: 2.0316459
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0670684, upper bound: 2.0316459
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0371161, 4.0088525
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6310282, 3.6625991
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1023645, 3.1047497
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9947853, 2.9820697
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7953110, 2.7665038
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9418907, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2320

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0905723, upper bound: 2.0465701
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0848890, upper bound: 2.0522461
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0008726, 4.0296521
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6439767, 3.6148734
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1122642, 3.1007440
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9809327, 2.9994369
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7653627, 2.8021243
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.8575649, 2.8447878
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1396

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1395

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9907428, upper bound: 1.9872560
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0017012, upper bound: 1.9785693
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0013781, 4.0274458
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6443753, 3.6122966
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1126547, 3.1000860
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9855466, 2.9918761
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7712359, 2.7919486
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.8633080, 2.8306718
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 317

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0253110, upper bound: 2.0054840
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0154637, upper bound: 2.0153255
time: 4.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0371456, 4.0010395
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6385288, 3.6162863
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0863843, 3.1122921
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9659338, 2.9800012
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7939095, 2.7723999
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9453750, 2.9450319
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 765

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 709

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0493444, upper bound: 2.0354664
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0480759, upper bound: 2.0367963
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0351486, 4.0030351
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6469622, 3.6078520
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0847106, 3.1139662
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9648485, 2.9810867
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7906504, 2.7756586
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9445062, 2.9459012
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1690

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1432

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0200147, upper bound: 2.0246666
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0379481, upper bound: 2.0066772
time: 5.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.9966955, 4.0273061
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6136990, 3.6453633
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1173244, 3.0897284
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9820757, 2.9668963
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7659388, 2.7869732
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9231744, 2.9224422
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 668

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9303836, upper bound: 1.9449962
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9303836, upper bound: 1.9449962
time: 5.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0080853, 4.0308051
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6288195, 3.6443758
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1173096, 3.0903568
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9820490, 2.9677479
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7687225, 2.7922106
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9496803, 2.9226489
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2383

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9684910, upper bound: 1.9753773
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9684910, upper bound: 1.9753773
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0397167, 4.0116196
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6229076, 3.6541319
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0964503, 3.1090250
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9687285, 2.9594772
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7993355, 2.7745953
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9418316, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 921

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1402

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0357566, upper bound: 2.0393058
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0383235, upper bound: 2.0365766
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0398941, 4.0114427
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6224518, 3.6545873
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0968471, 3.1086285
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9704204, 2.9577863
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.8011808, 2.7727494
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9439259, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9988714, upper bound: 2.0241315
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9987134, upper bound: 2.0242826
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0289631, 4.0375295
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6591935, 3.6282258
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1344161, 3.0982161
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9680004, 2.9795973
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7757111, 2.7969751
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9383872
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2594

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9821538, upper bound: 2.0230125
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9850933, upper bound: 2.0200910
time: 5.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0099173, 4.0389462
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6599336, 3.6289163
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1064420, 3.1000352
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9693050, 2.9800386
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7704468, 2.7973566
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9409540
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1101

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9896329, upper bound: 2.0229922
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9828786, upper bound: 2.0275703
time: 4.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0404329, 4.0068989
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6617260, 3.6287465
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0879698, 3.1209884
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9662590, 2.9869473
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7898617, 2.7722082
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9504766, 2.9503529
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1509

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2594

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.8697119, upper bound: 1.9057305
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.8736455, upper bound: 1.9018343
time: 4.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0416689, 4.0077543
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6594524, 3.6294174
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0883808, 3.1195035
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9672003, 2.9825082
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7932472, 2.7715528
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9498453, 2.9504769
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2570

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.8698557, upper bound: 1.8902753
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.8582053, upper bound: 1.9019026
time: 4.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9338948, upper bound: 1.9171574
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9500429, upper bound: 1.9161863
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9307793, upper bound: 1.8883393
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9320428, upper bound: 1.8987531
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -2.0670684, upper bound: 2.0316459
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -2.0670684, upper bound: 2.0316459
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -2.0905723, upper bound: 2.0465701
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -2.0848890, upper bound: 2.0522461
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9907428, upper bound: 1.9872560
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -2.0017012, upper bound: 1.9785693
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -2.0253110, upper bound: 2.0054840
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -2.0154637, upper bound: 2.0153255
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -2.0493444, upper bound: 2.0354664
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -2.0480759, upper bound: 2.0367963
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -2.0200147, upper bound: 2.0246666
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -2.0379481, upper bound: 2.0066772
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9303836, upper bound: 1.9449962
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9303836, upper bound: 1.9449962
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9684910, upper bound: 1.9753773
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9684910, upper bound: 1.9753773
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -2.0357566, upper bound: 2.0393058
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -2.0383235, upper bound: 2.0365766
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9988714, upper bound: 2.0241315
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9987134, upper bound: 2.0242826
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9821538, upper bound: 2.0230125
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9850933, upper bound: 2.0200910
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9896329, upper bound: 2.0229922
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.9828786, upper bound: 2.0275703
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.8697119, upper bound: 1.9057305
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.8736455, upper bound: 1.9018343
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.8698557, upper bound: 1.8902753
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.35
Output dim: 4, lower bound: -1.8582053, upper bound: 1.9019026

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6752176, 3.7181969
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.5696154, 3.5861177
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0157051, 2.9911318
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9749565, 2.9626946
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7771049, 2.7944613
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.8482723, 2.8176742
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.56 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.402289390563965
rel_dist={4: [-2.091210306910777, 2.0912106090343423]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 495

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7188988, upper bound: 1.7188962
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7188967, upper bound: 1.7405292
time: 5.11 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.06 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.06
Output dim: 4, lower bound: -1.7188988, upper bound: 1.7188962
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.06
Output dim: 4, lower bound: -1.7188967, upper bound: 1.7405292

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.7154493, 3.7173049
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4096947, 3.4278698
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8975348, 2.8920050
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3551989, 3.3543196
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7591810, 2.7514124
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.5187140, 2.5157728
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7747307, 2.7803373
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6567502, 2.6609159

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7405254, upper bound: 1.7187690
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7110575, upper bound: 1.7188964
time: 5.22 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.7173052, 3.7154486
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4278698, 3.4096951
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8920050, 2.8975348
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3543196, 3.3551989
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7514124, 2.7591813
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.5157728, 2.5187135
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7803373, 2.7747307
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6609159, 2.6567502

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6250
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6250

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7188963, upper bound: 1.7405288
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7188963, upper bound: 1.7405288
time: 5.44 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 4, lower bound: -1.7405254, upper bound: 1.7187690
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 4, lower bound: -1.7110575, upper bound: 1.7188964
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 4, lower bound: -1.7188963, upper bound: 1.7405288
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 4, lower bound: -1.7188963, upper bound: 1.7405288

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.7162023, 3.7182469
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4079080, 3.4257407
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.9032178, 2.8965580
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3561521, 3.3554978
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7594881, 2.7514060
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.5151606, 2.5128121
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7770505, 2.7822011
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6574173, 2.6617484

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7405250, upper bound: 1.7187687
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7405250, upper bound: 1.7187686
time: 4.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.7163911, 3.7180579
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4075665, 3.4253826
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.9020882, 2.8976870
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3559155, 3.3552728
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7591753, 2.7517190
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.5146012, 2.5122199
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7765946, 2.7817357
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6575799, 2.6615825

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7110570, upper bound: 1.7188960
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7110570, upper bound: 1.7188960
time: 4.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6516385, 3.6676059
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4297123, 3.4112511
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8710294, 2.8651991
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3543301, 3.3551788
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7453966, 2.7519629
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4453154, 2.4610667
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7540555, 2.7430072
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6649752, 2.6576347

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7188938, upper bound: 1.7110571
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7187686, upper bound: 1.7405251
time: 7.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6694627, 3.6497822
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4294262, 3.4115376
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8596697, 2.8765590
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3542995, 3.3552098
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7441940, 2.7531657
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4581261, 2.4482563
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7486138, 2.7484488
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6618004, 2.6608095

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7188938, upper bound: 1.7110571
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7187686, upper bound: 1.7405251
time: 7.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.86 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 4, lower bound: -1.7405250, upper bound: 1.7187687
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 4, lower bound: -1.7405250, upper bound: 1.7187686
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 4, lower bound: -1.7110570, upper bound: 1.7188960
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 4, lower bound: -1.7110570, upper bound: 1.7188960
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 4, lower bound: -1.7188938, upper bound: 1.7110571
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 4, lower bound: -1.7187686, upper bound: 1.7405251
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 4, lower bound: -1.7188938, upper bound: 1.7110571
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 4, lower bound: -1.7187686, upper bound: 1.7405251

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6505356, 3.6704040
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4097509, 3.4272971
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8822412, 2.8642220
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3561621, 3.3554759
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7534733, 2.7441883
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4447031, 2.4551654
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7507687, 2.7504776
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6614752, 2.6626320

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1782

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6880716, upper bound: 1.6666961
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6880716, upper bound: 1.6666961
time: 6.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6683578, 3.6525803
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4094648, 3.4275842
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8708816, 2.8755820
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3561316, 3.3555074
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7522707, 2.7453914
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4575138, 2.4423549
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7453270, 2.7559192
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6583004, 2.6658063

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 765

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2622

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7292886, upper bound: 1.7056790
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7272299, upper bound: 1.7077236
time: 5.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6507244, 3.6702151
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4094095, 3.4269395
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8811121, 2.8653512
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3559256, 3.3552518
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7531595, 2.7445014
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4441442, 2.4545732
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7503128, 2.7500122
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6616383, 2.6624660

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2236

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1753

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7080884, upper bound: 1.7125645
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7047341, upper bound: 1.7159392
time: 5.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6685486, 3.6523914
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4091234, 3.4272256
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8697519, 2.8767109
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3558950, 3.3552823
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7519569, 2.7457042
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4569550, 2.4417627
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7448711, 2.7554538
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6584635, 2.6656408

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2809

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1396

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6069574, upper bound: 1.6072178
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6069574, upper bound: 1.6072178
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6523914, 3.6685476
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4272261, 3.4091229
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8767109, 2.8697519
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3552828, 3.3558946
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7457037, 2.7519572
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4417629, 2.4569547
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7554541, 2.7448709
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6656408, 2.6584635

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1690

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7077007, upper bound: 1.6992517
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7069630, upper bound: 1.6999906
time: 13.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6525803, 3.6683588
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4275846, 3.4094644
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8755817, 2.8708816
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3555079, 3.3561311
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7453909, 2.7522702
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4423552, 2.4575138
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7559195, 2.7453263
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6658063, 2.6583004

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2334

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6928428, upper bound: 1.7150022
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6928428, upper bound: 1.7150022
time: 5.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6702156, 3.6507239
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4269400, 3.4094095
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8653512, 2.8811119
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3552523, 3.3559260
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7445011, 2.7531600
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4545727, 2.4441440
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7500124, 2.7503126
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6624660, 2.6616383

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 3105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7034811, upper bound: 1.7032790
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7111276, upper bound: 1.6956325
time: 5.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6704044, 3.6505346
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4272966, 3.4097514
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8642220, 2.8822415
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3554764, 3.3561625
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7441883, 2.7534730
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4551649, 2.4447033
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7504778, 2.7507679
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6626320, 2.6614752

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2383

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1396

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6071313, upper bound: 1.6264881
time: 6.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6071313, upper bound: 1.6264881
time: 6.04 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.6880716, upper bound: 1.6666961
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.6880716, upper bound: 1.6666961
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.7292886, upper bound: 1.7056790
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.7272299, upper bound: 1.7077236
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.7080884, upper bound: 1.7125645
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.7047341, upper bound: 1.7159392
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.6069574, upper bound: 1.6072178
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.6069574, upper bound: 1.6072178
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.7077007, upper bound: 1.6992517
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.7069630, upper bound: 1.6999906
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.6928428, upper bound: 1.7150022
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.6928428, upper bound: 1.7150022
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.7034811, upper bound: 1.7032790
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.7111276, upper bound: 1.6956325
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.6071313, upper bound: 1.6264881
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.6071313, upper bound: 1.6264881

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6499472, 3.6701126
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4086981, 3.4262657
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8813210, 2.8645837
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3557587, 3.3544326
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7516327, 2.7433329
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4446354, 2.4555788
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7501006, 2.7495401
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6633000, 2.6623130

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2384

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6821508, upper bound: 1.6595017
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6808785, upper bound: 1.6606869
time: 5.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6502438, 3.6704040
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4097509, 3.4262443
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8822412, 2.8633015
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3561621, 3.3550715
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7526178, 2.7441883
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4447031, 2.4550972
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7507687, 2.7498105
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6611567, 2.6626320

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2341

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 317

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6776779, upper bound: 1.6567352
time: 6.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6776181, upper bound: 1.6567686
time: 5.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6349020, 3.6174669
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4062757, 3.4144754
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8672023, 2.8718488
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3414965, 3.3416128
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7338629, 2.7344146
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4540715, 2.4398146
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7416639, 2.7527053
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6381707, 2.6427326

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2314

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7089405, upper bound: 1.7053054
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7289242, upper bound: 1.6853577
time: 5.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6332455, 3.6191235
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3963556, 3.4243956
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8671484, 2.8719027
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3422375, 3.3408723
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7412939, 2.7269835
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4549732, 2.4389129
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7421131, 2.7522562
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6352267, 2.6456766

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2320

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7074126, upper bound: 1.6847791
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7072940, upper bound: 1.6859420
time: 6.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6177983, 3.6336083
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3797870, 3.3991842
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8880887, 2.8820066
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3495665, 3.3472676
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7487736, 2.7385483
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4127398, 2.4202030
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7459722, 2.7470813
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6609349, 2.6635327

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 709

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 669

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6315835, upper bound: 1.6380606
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6315835, upper bound: 1.6380606
time: 8.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6145959, 3.6372895
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3821998, 3.3973169
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8988638, 2.8723278
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3479424, 3.3491292
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7472067, 2.7404790
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4105158, 2.4231689
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7473807, 2.7457883
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6630034, 2.6617627

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2369

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6929850, upper bound: 1.7147520
time: 8.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7035513, upper bound: 1.6996037
time: 5.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6674433, 3.6515355
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4099369, 3.4265547
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8693409, 2.8772402
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3559790, 3.3550754
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7510171, 2.7474566
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4535694, 2.4394794
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7451744, 2.7553298
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6591229, 2.6654100

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 411

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2570

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6025147, upper bound: 1.6058269
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6055665, upper bound: 1.6027572
time: 5.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6676912, 3.6523914
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4084530, 3.4272256
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8697519, 2.8762999
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3556871, 3.3552823
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7519569, 2.7447639
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4569550, 2.4383771
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7447472, 2.7554538
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6582332, 2.6656408

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2371

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1451

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5512894, upper bound: 1.5534586
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5534500, upper bound: 1.5512173
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6505051, 3.6669102
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4281006, 3.4075789
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8758311, 2.8693199
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3545790, 3.3537550
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7452488, 2.7517955
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4415164, 2.4564209
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7547188, 2.7420163
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6652927, 2.6573758

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1978

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2314

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6876087, upper bound: 1.6989635
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7074107, upper bound: 1.6791805
time: 6.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6507549, 3.6685476
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4256821, 3.4091229
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8762789, 2.8697519
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3552828, 3.3551912
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7455425, 2.7519572
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4412293, 2.4569547
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7525997, 2.7448709
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6656408, 2.6581159

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6456613, upper bound: 1.6386547
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6456613, upper bound: 1.6386547
time: 6.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6511602, 3.6651893
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4298267, 3.4074135
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8672147, 2.8605411
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3546791, 3.3560610
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7364192, 2.7438335
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4408436, 2.4553249
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7529111, 2.7415562
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6632900, 2.6564426

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2369

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2314

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6723140, upper bound: 1.7147069
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6925628, upper bound: 1.6945844
time: 5.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6525803, 3.6669388
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4255333, 3.4094644
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8652415, 2.8708816
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3555079, 3.3553038
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7369542, 2.7522702
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4423552, 2.4560025
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7559195, 2.7423186
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6658063, 2.6557841

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6790917, upper bound: 1.7012657
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6792103, upper bound: 1.7009979
time: 6.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6390038, 3.6216531
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4088755, 3.3933320
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8571911, 2.8715243
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3480949, 3.3514829
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7279382, 2.7293801
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4522824, 2.4369574
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7450004, 2.7431364
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6587081, 2.6526966

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2314

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7000525, upper bound: 1.6988521
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6991185, upper bound: 1.6997779
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6411438, 3.6195130
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4108610, 3.3913460
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8557634, 2.8729520
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3508081, 3.3487692
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7207217, 2.7365966
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4473863, 2.4418535
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7428365, 2.7453008
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6535244, 2.6578803

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1983

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7087630, upper bound: 1.6901487
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7049405, upper bound: 1.6932730
time: 5.54 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 27.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6821508, upper bound: 1.6595017
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6808785, upper bound: 1.6606869
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6776779, upper bound: 1.6567352
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6776181, upper bound: 1.6567686
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.7089405, upper bound: 1.7053054
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.7289242, upper bound: 1.6853577
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.7074126, upper bound: 1.6847791
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.7072940, upper bound: 1.6859420
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6315835, upper bound: 1.6380606
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6315835, upper bound: 1.6380606
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6929850, upper bound: 1.7147520
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.7035513, upper bound: 1.6996037
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6025147, upper bound: 1.6058269
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6055665, upper bound: 1.6027572
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.5512894, upper bound: 1.5534586
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.5534500, upper bound: 1.5512173
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6876087, upper bound: 1.6989635
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.7074107, upper bound: 1.6791805
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6456613, upper bound: 1.6386547
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6456613, upper bound: 1.6386547
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6723140, upper bound: 1.7147069
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6925628, upper bound: 1.6945844
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6790917, upper bound: 1.7012657
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6792103, upper bound: 1.7009979
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.7000525, upper bound: 1.6988521
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.6991185, upper bound: 1.6997779
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.7087630, upper bound: 1.6901487
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.39
Output dim: 4, lower bound: -1.7049405, upper bound: 1.6932730
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.39
Output dim: 4, lower bound: -1.6071313, upper bound: 1.6264881
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.39
Output dim: 4, lower bound: -1.6071313, upper bound: 1.6264881
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=3.359607696533203
rel_dist={4: [-1.74054476319957, 1.7405458869370563]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6250
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6250

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115648, upper bound: 1.6115668
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115668, upper bound: 1.6115667
time: 5.10 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.36 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.36
Output dim: 4, lower bound: -1.6115648, upper bound: 1.6115668
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.36
Output dim: 4, lower bound: -1.6115668, upper bound: 1.6115667

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5215483, 3.5349159
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3992033, 3.3989882
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7887115, 2.7949197
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7661748, 2.7576547
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2979774, 3.2979541
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6935301, 2.6926281
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3214426, 2.3310504
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7156959, 2.7116146
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5823317, 2.5799503

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 495

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115511, upper bound: 1.5954346
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954347, upper bound: 1.6115534
time: 5.15 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5349169, 3.5215480
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3989878, 3.3992038
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7949200, 2.7887113
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7576547, 2.7661746
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2979546, 3.2979774
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6926279, 2.6935303
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3310504, 2.3214426
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7116141, 2.7156954
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5799503, 2.5823317

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 495

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115636, upper bound: 1.6004872
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6004851, upper bound: 1.6115641
time: 4.89 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.67 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.67
Output dim: 4, lower bound: -1.6115511, upper bound: 1.5954346
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.67
Output dim: 4, lower bound: -1.5954347, upper bound: 1.6115534
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.67
Output dim: 4, lower bound: -1.6115636, upper bound: 1.6004872
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.67
Output dim: 4, lower bound: -1.6004851, upper bound: 1.6115641

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5307088, 3.5454693
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3384018, 3.3518167
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7976170, 2.7849495
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2933483, 3.2926650
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6765852, 2.6698568
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3381777, 2.3455803
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6819930, 2.6821165
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6028948, 2.6036377

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115499, upper bound: 1.5953406
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5895300, upper bound: 1.5954348
time: 4.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5321012, 3.5440769
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3520317, 3.3381858
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7934694, 2.7890971
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2926884, 3.2933249
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6707592, 2.6756835
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3359728, 2.3477857
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6861978, 2.6779118
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2211523, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6060185, 2.6005139

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954326, upper bound: 1.5895323
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5895323, upper bound: 1.5954325
time: 8.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5356684, 3.5224423
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4006872, 3.4006462
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7950339, 2.7888074
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7630544, 2.7707276
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2989082, 3.2991004
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6909885, 2.6916556
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3274989, 2.3183353
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7138200, 2.7175593
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5806165, 2.5831199

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 495

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 495

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115499, upper bound: 1.5953406
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954326, upper bound: 1.5895323
time: 5.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5358105, 3.5223005
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4004316, 3.4009018
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7950158, 2.7888255
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7622075, 2.7715745
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2990770, 3.2989321
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6907530, 2.6918905
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3279433, 2.3178911
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7134776, 2.7179008
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5807390, 2.5829978

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 495

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 495

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5895300, upper bound: 1.5954348
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5953408, upper bound: 1.6115502
time: 4.81 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.86 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 4, lower bound: -1.6115499, upper bound: 1.5953406
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 4, lower bound: -1.5895300, upper bound: 1.5954348
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 4, lower bound: -1.5954326, upper bound: 1.5895323
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 4, lower bound: -1.5895323, upper bound: 1.5954325
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 4, lower bound: -1.6115499, upper bound: 1.5953406
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 4, lower bound: -1.5954326, upper bound: 1.5895323
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 4, lower bound: -1.5895300, upper bound: 1.5954348
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.86
Output dim: 4, lower bound: -1.5953408, upper bound: 1.6115502

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5314617, 3.5463638
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3365297, 3.3496890
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8030167, 2.7895024
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2943010, 3.2937860
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6768141, 2.6698511
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3346252, 2.3424716
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6841993, 2.6839805
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6035604, 2.6044278

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2333

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1384

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6046476, upper bound: 1.5876043
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6038081, upper bound: 1.5884416
time: 4.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5316029, 3.5462217
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3362722, 3.3494201
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8021698, 2.7903492
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2941236, 3.2936177
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6765795, 2.6700859
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3342056, 2.3420274
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6838570, 2.6836314
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6036825, 2.6043034

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2642

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1396

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5060938, upper bound: 1.5076668
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5060938, upper bound: 1.5076668
time: 5.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5328541, 3.5449715
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3496351, 3.3360581
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7988691, 2.7936499
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2936411, 3.2941003
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6709881, 2.6756778
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3324194, 2.3438134
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6877127, 2.6797752
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2205467, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6066847, 2.6013017

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 310

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1753

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5932043, upper bound: 1.5843940
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5902973, upper bound: 1.5872994
time: 5.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5329952, 3.5448294
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3499041, 3.3363142
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7980223, 2.7944970
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2938099, 3.2942777
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6707535, 2.6759124
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3328638, 2.3442328
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6880617, 2.6801171
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2204618, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6068087, 2.6011796

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5951781, upper bound: 1.6101382
time: 6.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5939284, upper bound: 1.6113880
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5448303, 3.5329957
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3363142, 3.3499036
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7944970, 2.7980223
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2942781, 3.2938094
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6759119, 2.6707532
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3442326, 2.3328638
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6801176, 2.6880617
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2204618
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6011796, 2.6068087

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 913

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 310

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6072666, upper bound: 1.5864470
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6026295, upper bound: 1.5911006
time: 5.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5462227, 3.5316033
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3494196, 3.3362727
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7903490, 2.8021698
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2936182, 3.2941232
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6700859, 2.6765800
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3420277, 2.3342056
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6836319, 2.6838570
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6043034, 2.6036825

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2622

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5844954, upper bound: 1.5777271
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5836102, upper bound: 1.5786134
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5449715, 3.5328541
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3360586, 3.3496351
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7936497, 2.7988691
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2941008, 3.2936411
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6756773, 2.6709881
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3438129, 2.3324196
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6797762, 2.6877127
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2205467
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6013017, 2.6066847

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 611

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 709

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5888625, upper bound: 1.5941267
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5883265, upper bound: 1.5947197
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5463638, 3.5314617
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3496885, 3.3365293
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7895021, 2.8030169
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2937860, 3.2943006
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6698513, 2.6768146
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3424711, 2.3346250
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6839809, 2.6841984
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6044278, 2.6035604

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2320

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2570

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5916446, upper bound: 1.6104920
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5942887, upper bound: 1.6078529
time: 5.20 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.6046476, upper bound: 1.5876043
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.6038081, upper bound: 1.5884416
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.5060938, upper bound: 1.5076668
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.5060938, upper bound: 1.5076668
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.5932043, upper bound: 1.5843940
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.5902973, upper bound: 1.5872994
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.5951781, upper bound: 1.6101382
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.5939284, upper bound: 1.6113880
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.6072666, upper bound: 1.5864470
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.6026295, upper bound: 1.5911006
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.5844954, upper bound: 1.5777271
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.5836102, upper bound: 1.5786134
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.5888625, upper bound: 1.5941267
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.5883265, upper bound: 1.5947197
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.5916446, upper bound: 1.6104920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.25
Output dim: 4, lower bound: -1.5942887, upper bound: 1.6078529

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5314245, 3.5468936
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3359785, 3.3487172
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8020020, 2.7885284
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2917676, 3.2911448
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6767950, 2.6698589
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3344245, 2.3400855
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6803064, 2.6812425
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6015339, 2.6024418

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 409

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1395

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5622936, upper bound: 1.5648478
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5831868, upper bound: 1.5603402
time: 5.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5319920, 3.5463266
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3355589, 3.3491368
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8020430, 2.7884872
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2916589, 3.2912526
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6768227, 2.6698310
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3322387, 2.3422713
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6814604, 2.6800880
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6015744, 2.6024013

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5514667, upper bound: 1.5355163
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5514667, upper bound: 1.5355163
time: 4.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.4999280, 3.5092850
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3200126, 3.3078361
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8058457, 2.8078856
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2868757, 3.2861161
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6662102, 2.6697247
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3010159, 2.3101847
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6837249, 2.6768446
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.1874580, 3.2017188
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6059809, 2.6019258

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1384

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5925834, upper bound: 1.5791369
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5820634, upper bound: 1.5837629
time: 5.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.4975266, 3.5120459
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3218226, 3.3064356
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8139272, 2.8006265
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2856569, 3.2875123
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6650352, 2.6711726
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.2993469, 2.3124092
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6847816, 2.6758747
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.1926460, 3.1971345
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6075320, 2.6005983

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2320

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 403

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5889454, upper bound: 1.5834933
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5864074, upper bound: 1.5859852
time: 7.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.4557071, 3.4738054
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.2844386, 3.2575493
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7276506, 2.7251253
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2672167, 3.2674937
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6414790, 2.6506717
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4016666, 2.4139557
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7201734, 2.7156801
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.1502852, 3.1580763
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6335883, 2.6249785

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2383

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5665917, upper bound: 1.5815078
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5665917, upper bound: 1.5815078
time: 4.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.4619708, 3.4675398
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.2711387, 3.2708483
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7286468, 2.7241254
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2670259, 3.2676845
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6455112, 2.6466384
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4025869, 2.4130325
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7236238, 2.7122283
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.1488309, 3.1595306
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6306081, 2.6279588

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1101

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5935915, upper bound: 1.6088025
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5905901, upper bound: 1.6110567
time: 5.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5447426, 3.5327306
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3254957, 3.3448443
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7600312, 2.7695765
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2789288, 3.2750401
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6758137, 2.6705825
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3418188, 2.3296621
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6692801, 2.6762152
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2177787
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5810719, 2.5888424

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1395

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5646820, upper bound: 1.5635742
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5855595, upper bound: 1.5599741
time: 6.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5445633, 3.5329089
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3312559, 3.3390851
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7660513, 2.7635581
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2755089, 3.2784605
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6757412, 2.6706543
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3410301, 2.3304496
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6682701, 2.6772261
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2163672
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5832133, 2.5867019

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5994725, upper bound: 1.5837030
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5987469, upper bound: 1.5886918
time: 5.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5123510, 3.4964905
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3437510, 3.3231645
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7866702, 2.7984500
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2789831, 3.2800441
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6535358, 2.6656032
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3388104, 2.3316653
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6800814, 2.6806431
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2144051, 3.2144165
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5834374, 2.5806093

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 611

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5250717, upper bound: 1.5241541
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5250717, upper bound: 1.5241541
time: 5.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5111084, 3.4977329
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3363123, 3.3306046
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7866292, 2.7984905
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2795391, 3.2794886
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6591091, 2.6600299
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3394866, 2.3309889
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6804180, 2.6803060
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2140083, 3.2148128
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5812297, 2.5828166

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5643603, upper bound: 1.5741328
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5792218, upper bound: 1.5593260
time: 4.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5473433, 3.5351677
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3355427, 3.3488126
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7736487, 2.7700861
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7910881, 2.7848198
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2991395, 3.2972374
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6401958, 2.6267080
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3434219, 2.3337958
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6810989, 2.6940365
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.1804023, 3.1763144
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5678141, 2.5760703

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2488

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5847998, upper bound: 1.5904466
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5851841, upper bound: 1.5900443
time: 5.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5473318, 3.5352259
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3352356, 3.3491645
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7793593, 2.7645612
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7796006, 2.7966011
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2976956, 3.2987566
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6313982, 2.6357684
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3453112, 2.3320279
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6864367, 2.6890364
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.1856503, 3.1713285
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5707905, 2.5731969

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5765533, upper bound: 1.5885023
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5821189, upper bound: 1.5829267
time: 8.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5454311, 3.5297499
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3682146, 3.3397789
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8028207, 2.8138082
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2883973, 3.2890873
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6570354, 2.6632051
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3426719, 2.3346264
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6889739, 2.6885815
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5945053, 2.5950923

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 669

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1395

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5626052, upper bound: 1.5888711
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5695034, upper bound: 1.5679910
time: 5.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5446510, 3.5305295
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3529387, 3.3550577
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8002934, 2.8163338
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2885728, 3.2889118
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6562419, 2.6639986
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3424735, 2.3348248
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6883626, 2.6891918
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5959587, 2.5936379

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2236

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5923076, upper bound: 1.6014964
time: 7.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5804332, upper bound: 1.5898374
time: 5.78 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 28.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5622936, upper bound: 1.5648478
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5831868, upper bound: 1.5603402
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5514667, upper bound: 1.5355163
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5514667, upper bound: 1.5355163
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5925834, upper bound: 1.5791369
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5820634, upper bound: 1.5837629
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5889454, upper bound: 1.5834933
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5864074, upper bound: 1.5859852
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5665917, upper bound: 1.5815078
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5665917, upper bound: 1.5815078
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5935915, upper bound: 1.6088025
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5905901, upper bound: 1.6110567
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5646820, upper bound: 1.5635742
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5855595, upper bound: 1.5599741
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5994725, upper bound: 1.5837030
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5987469, upper bound: 1.5886918
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5250717, upper bound: 1.5241541
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5250717, upper bound: 1.5241541
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5643603, upper bound: 1.5741328
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5792218, upper bound: 1.5593260
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5847998, upper bound: 1.5904466
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5851841, upper bound: 1.5900443
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5765533, upper bound: 1.5885023
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5821189, upper bound: 1.5829267
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5626052, upper bound: 1.5888711
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5695034, upper bound: 1.5679910
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5923076, upper bound: 1.6014964
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 4, lower bound: -1.5804332, upper bound: 1.5898374

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5113888, 3.5275416
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3398323, 3.3541408
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8367095, 2.8234162
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2449698, 3.2390723
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.5821624, 2.5881498
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3332958, 2.3411441
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6414280, 2.6428518
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2024913, 3.1985970
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.4572887, 2.4504995

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1782

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5194162, upper bound: 1.4956796
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5194162, upper bound: 1.4956796
time: 5.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5280781, 3.5428486
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3456821, 3.3319821
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7979283, 2.7925942
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2794743, 3.2785096
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6659694, 2.6707830
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3069735, 2.3184819
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6658249, 2.6611874
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2199397, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6062679, 2.6007190

Time for backsubstitution: 14.31 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=3.2979745864868164
rel_dist={4: [-1.6115673031162494, 1.6115673021877095]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2417.73 seconds
