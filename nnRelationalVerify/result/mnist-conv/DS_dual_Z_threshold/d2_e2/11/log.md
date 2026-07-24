## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.167531904


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4602838, 0.4602839)
1: (2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2606032, 0.2606031)
2: (-3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2680004, 0.2680004)
3: (-11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2803777, 0.2803777)
4: (-2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1896749, 0.1896749)
5: (-9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2679567, 0.2679568)
6: (-7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4301465, 0.4301466)
7: (-4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2142508, 0.2142508)
8: (-1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2695249, 0.2695248)
9: (-14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3115454, 0.3115455)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.50 + 33.11 = 56.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.1745120, upper bound: 0.1745124

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 4627

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1745030, upper bound: 0.1708731
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1708734, upper bound: 0.1745026
time: 3.08 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.29 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.29
Output dim: 1, lower bound: -0.1745030, upper bound: 0.1708731
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.29
Output dim: 1, lower bound: -0.1708734, upper bound: 0.1745026

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4559846, 0.4545482
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2500391, 0.2464415
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2678010, 0.2677350
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2677395, 0.2709427
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1865657, 0.1855336
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2562801, 0.2594404
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4073577, 0.4133233
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2118154, 0.2107937
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2694472, 0.2694666
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3039910, 0.3012532

Time for backsubstitution: 20.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 472

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1737490, upper bound: 0.1708699
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1745002, upper bound: 0.1701206
time: 3.05 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4545481, 0.4559846
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2464415, 0.2500391
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2677348, 0.2678013
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2709427, 0.2677395
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1855336, 0.1865658
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2594404, 0.2562801
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4133232, 0.4073576
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2107937, 0.2118154
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2694665, 0.2694473
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3012532, 0.3039910

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 472

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1701205, upper bound: 0.1744997
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1708704, upper bound: 0.1737486
time: 3.09 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.18 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.18
Output dim: 1, lower bound: -0.1737490, upper bound: 0.1708699
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.18
Output dim: 1, lower bound: -0.1745002, upper bound: 0.1701206
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.18
Output dim: 1, lower bound: -0.1701205, upper bound: 0.1744997
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.18
Output dim: 1, lower bound: -0.1708704, upper bound: 0.1737486

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4562550, 0.4535329
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2493867, 0.2466162
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2664349, 0.2680998
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2682153, 0.2691708
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1851421, 0.1859131
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2567146, 0.2578121
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4074337, 0.4130348
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2112855, 0.2109348
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2697282, 0.2684174
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3036025, 0.3013537

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1737485, upper bound: 0.1694676
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1723372, upper bound: 0.1701195
time: 5.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4549692, 0.4545482
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2500391, 0.2457892
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2678010, 0.2663684
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2659677, 0.2709427
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1865657, 0.1841098
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2546518, 0.2594404
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4070694, 0.4133233
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2118154, 0.2102637
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2683983, 0.2694666
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3039910, 0.3008649

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1744995, upper bound: 0.1687154
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1730937, upper bound: 0.1701202
time: 3.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4548185, 0.4549693
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2457892, 0.2502139
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2663686, 0.2681661
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2714185, 0.2659678
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1841097, 0.1869453
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2598748, 0.2546518
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4133995, 0.4070693
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2102637, 0.2119565
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2697477, 0.2683980
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3008649, 0.3040915

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1701199, upper bound: 0.1730933
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1687159, upper bound: 0.1744999
time: 3.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4535329, 0.4559846
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2464415, 0.2493867
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2677348, 0.2664347
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2691709, 0.2677395
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1855336, 0.1851420
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2578121, 0.2562801
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4130349, 0.4073576
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2107937, 0.2112854
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2684174, 0.2694473
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3012532, 0.3036027

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1708697, upper bound: 0.1723367
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1694680, upper bound: 0.1737488
time: 3.25 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.42 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.42
Output dim: 1, lower bound: -0.1737485, upper bound: 0.1694676
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.42
Output dim: 1, lower bound: -0.1723372, upper bound: 0.1701195
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.42
Output dim: 1, lower bound: -0.1744995, upper bound: 0.1687154
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.42
Output dim: 1, lower bound: -0.1730937, upper bound: 0.1701202
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.42
Output dim: 1, lower bound: -0.1701199, upper bound: 0.1730933
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.42
Output dim: 1, lower bound: -0.1687159, upper bound: 0.1744999
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.42
Output dim: 1, lower bound: -0.1708697, upper bound: 0.1723367
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.42
Output dim: 1, lower bound: -0.1694680, upper bound: 0.1737488

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4562564, 0.4535340
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2493894, 0.2466178
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2664304, 0.2680941
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2682056, 0.2691582
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1851422, 0.1859123
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2567061, 0.2578057
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4074280, 0.4130275
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2112782, 0.2109255
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2697206, 0.2684114
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3035986, 0.3013484

Time for backsubstitution: 21.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 3, pos: 1207

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1691861, upper bound: 0.1642897
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1684088, upper bound: 0.1650653
time: 3.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4562559, 0.4535342
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2493882, 0.2466191
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2664289, 0.2680955
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2682028, 0.2691612
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1851411, 0.1859133
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2567083, 0.2578036
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4074266, 0.4130292
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2112761, 0.2109276
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2697220, 0.2684097
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3035972, 0.3013496

Time for backsubstitution: 21.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 1207

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1677857, upper bound: 0.1656755
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1670084, upper bound: 0.1664505
time: 3.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4549708, 0.4545492
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2500420, 0.2457906
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2677972, 0.2663627
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2659583, 0.2709299
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1865658, 0.1841090
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2546433, 0.2594341
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4070635, 0.4133155
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2118083, 0.2102544
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2683904, 0.2694604
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3039868, 0.3008596

Time for backsubstitution: 21.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 3, pos: 1207

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699339, upper bound: 0.1635405
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1691561, upper bound: 0.1643158
time: 3.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4549704, 0.4545496
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2500407, 0.2457919
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2677960, 0.2663641
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2659552, 0.2709329
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1865648, 0.1841100
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2546455, 0.2594320
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4070618, 0.4133172
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2118061, 0.2102566
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2683921, 0.2694590
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3039856, 0.3008608

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 1207

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1685365, upper bound: 0.1649310
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1677593, upper bound: 0.1657074
time: 3.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4548202, 0.4549702
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2457919, 0.2502154
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2663641, 0.2681603
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2714088, 0.2659553
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1841100, 0.1869444
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2598664, 0.2546454
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4133937, 0.4070621
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2102566, 0.2119471
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2697399, 0.2683918
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3008606, 0.3040862

Time for backsubstitution: 21.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.46 seconds

### Candidate
type: DSZ, layer: 3, pos: 1207

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1657075, upper bound: 0.1677591
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1649318, upper bound: 0.1685366
time: 3.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4548197, 0.4549706
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2457907, 0.2502166
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2663627, 0.2681618
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2714057, 0.2659582
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1841090, 0.1869454
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2598685, 0.2546433
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4133921, 0.4070635
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2102544, 0.2119492
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2697415, 0.2683904
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3008597, 0.3040874

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.38 seconds

### Candidate
type: DSZ, layer: 3, pos: 1207

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1643164, upper bound: 0.1691564
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1635408, upper bound: 0.1699333
time: 3.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4535344, 0.4559857
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2464443, 0.2493882
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2677310, 0.2664289
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2691612, 0.2677270
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1855336, 0.1851411
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2578036, 0.2562738
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4130292, 0.4073501
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2107866, 0.2112761
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2684097, 0.2694414
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3012490, 0.3035973

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 1207

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1664512, upper bound: 0.1670086
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1656758, upper bound: 0.1677863
time: 3.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4535341, 0.4559859
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2464430, 0.2493895
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2677298, 0.2664304
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2691584, 0.2677299
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1855327, 0.1851422
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2578057, 0.2562717
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4130275, 0.4073517
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2107844, 0.2112782
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2684114, 0.2694395
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3012478, 0.3035985

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.38 seconds

### Candidate
type: DSZ, layer: 3, pos: 1207

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1650649, upper bound: 0.1684087
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1642899, upper bound: 0.1691864
time: 3.78 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.61 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1691861, upper bound: 0.1642897
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1684088, upper bound: 0.1650653
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1677857, upper bound: 0.1656755
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1670084, upper bound: 0.1664505
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1699339, upper bound: 0.1635405
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1691561, upper bound: 0.1643158
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1685365, upper bound: 0.1649310
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1677593, upper bound: 0.1657074
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1657075, upper bound: 0.1677591
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1649318, upper bound: 0.1685366
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1643164, upper bound: 0.1691564
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1635408, upper bound: 0.1699333
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1664512, upper bound: 0.1670086
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1656758, upper bound: 0.1677863
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1650649, upper bound: 0.1684087
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 1, lower bound: -0.1642899, upper bound: 0.1691864

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4552777, 0.4543698
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2496356, 0.2449661
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2662477, 0.2677028
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2633922, 0.2638674
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1783264, 0.1836186
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2566488, 0.2577517
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4059572, 0.4106246
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2089032, 0.2091440
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2697062, 0.2684069
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3032939, 0.3020920

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2327

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1689410, upper bound: 0.1596748
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1649588, upper bound: 0.1640441
time: 3.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4562564, 0.4525553
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2477379, 0.2466178
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2664304, 0.2679117
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2682056, 0.2643448
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1828484, 0.1859123
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2567061, 0.2577484
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4074280, 0.4115567
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2094969, 0.2109255
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2697160, 0.2684114
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3035986, 0.3010436

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 2327

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1681633, upper bound: 0.1608215
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1638164, upper bound: 0.1648199
time: 4.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4552772, 0.4543701
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2496344, 0.2449675
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2662463, 0.2677038
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2633893, 0.2638702
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1783254, 0.1836196
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2566509, 0.2577496
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4059558, 0.4106261
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2089010, 0.2091461
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2697077, 0.2684052
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3032925, 0.3020931

Time for backsubstitution: 22.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 2327

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1675391, upper bound: 0.1610628
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1635595, upper bound: 0.1654425
time: 3.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4539921, 0.4553851
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2502880, 0.2441391
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2676144, 0.2659714
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2611446, 0.2656391
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1797501, 0.1818153
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2545860, 0.2593801
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4055927, 0.4109126
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2094333, 0.2084730
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2683761, 0.2694559
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3036821, 0.3016031

Time for backsubstitution: 21.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2327

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1696905, upper bound: 0.1589264
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1657065, upper bound: 0.1632936
time: 3.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4549708, 0.4535706
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2483903, 0.2457906
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2677972, 0.2661803
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2659583, 0.2661166
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1842722, 0.1841090
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2546433, 0.2593768
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4070635, 0.4118448
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2100269, 0.2102544
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2683859, 0.2694604
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3039868, 0.3005548

Time for backsubstitution: 22.04 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.61 + 549.85 = 606.47 seconds
