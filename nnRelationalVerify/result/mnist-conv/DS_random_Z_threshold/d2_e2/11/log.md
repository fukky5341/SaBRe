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
execution time: IAR + RelationalAnalysis = 25.29 + 33.32 = 58.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.1745120, upper bound: 0.1745124

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 472

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4627

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1745030, upper bound: 0.1708731
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1708734, upper bound: 0.1745026
time: 2.99 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.91 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.91
Output dim: 1, lower bound: -0.1745030, upper bound: 0.1708731
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.91
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

Time for backsubstitution: 23.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 472

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1745024, upper bound: 0.1694709
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1730968, upper bound: 0.1708726
time: 2.97 seconds

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

Time for backsubstitution: 22.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 472

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1708728, upper bound: 0.1730964
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1694712, upper bound: 0.1745024
time: 3.07 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.07 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.07
Output dim: 1, lower bound: -0.1745024, upper bound: 0.1694709
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.07
Output dim: 1, lower bound: -0.1730968, upper bound: 0.1708726
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.07
Output dim: 1, lower bound: -0.1708728, upper bound: 0.1730964
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.07
Output dim: 1, lower bound: -0.1694712, upper bound: 0.1745024

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4559860, 0.4545492
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2500420, 0.2464432
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2677972, 0.2677298
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2677300, 0.2709299
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1865658, 0.1855326
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2562716, 0.2594341
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4073517, 0.4133155
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2118083, 0.2107844
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2694396, 0.2694604
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3039868, 0.3012478

Time for backsubstitution: 23.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 472

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 472

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1737485, upper bound: 0.1694676
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1744995, upper bound: 0.1687154
time: 2.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4559855, 0.4545496
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2500407, 0.2464443
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2677960, 0.2677310
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2677271, 0.2709329
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1865648, 0.1855336
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2562737, 0.2594320
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4073501, 0.4133172
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2118061, 0.2107866
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2694410, 0.2694590
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3039856, 0.3012490

Time for backsubstitution: 23.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 472

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 472

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1723372, upper bound: 0.1701195
time: 5.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1730937, upper bound: 0.1701202
time: 3.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4545498, 0.4559857
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2464443, 0.2500407
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2677310, 0.2677960
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2709329, 0.2677270
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1855336, 0.1865647
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2594321, 0.2562738
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4133172, 0.4073501
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2107866, 0.2118061
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2694591, 0.2694414
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3012490, 0.3039856

Time for backsubstitution: 23.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 472

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 472

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1701199, upper bound: 0.1730933
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1708697, upper bound: 0.1723367
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4545493, 0.4559859
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2464430, 0.2500418
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2677298, 0.2677972
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2709301, 0.2677299
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1855327, 0.1865658
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2594341, 0.2562717
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4133158, 0.4073517
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2107844, 0.2118083
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2694606, 0.2694395
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3012478, 0.3039868

Time for backsubstitution: 23.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 472

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 472

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1687159, upper bound: 0.1744999
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1694680, upper bound: 0.1737488
time: 3.16 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.06 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 1, lower bound: -0.1737485, upper bound: 0.1694676
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 1, lower bound: -0.1744995, upper bound: 0.1687154
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 1, lower bound: -0.1723372, upper bound: 0.1701195
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 1, lower bound: -0.1730937, upper bound: 0.1701202
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 1, lower bound: -0.1701199, upper bound: 0.1730933
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 1, lower bound: -0.1708697, upper bound: 0.1723367
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 1, lower bound: -0.1687159, upper bound: 0.1744999
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.06
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

Time for backsubstitution: 22.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2927

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1664044, upper bound: 0.1619853
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1664044, upper bound: 0.1619853
time: 4.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 23.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1808

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1807

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1744994, upper bound: 0.1672193
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1730041, upper bound: 0.1687156
time: 3.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 23.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1775

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2327

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1720710, upper bound: 0.1663415
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1678090, upper bound: 0.1706135
time: 3.18 seconds

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

Time for backsubstitution: 23.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 913

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1488

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1659089, upper bound: 0.1633635
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1661741, upper bound: 0.1632295
time: 3.43 seconds

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

Time for backsubstitution: 22.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1787

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2543

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1686560, upper bound: 0.1724337
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1694571, upper bound: 0.1716321
time: 3.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 23.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2119

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1691800, upper bound: 0.1714874
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1700527, upper bound: 0.1705077
time: 3.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 23.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1978

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1568537, upper bound: 0.1612094
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1568537, upper bound: 0.1612095
time: 3.57 seconds

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

Time for backsubstitution: 23.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 1488

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2488

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1595693, upper bound: 0.1638388
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1595693, upper bound: 0.1638388
time: 3.74 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.19 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1664044, upper bound: 0.1619853
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1664044, upper bound: 0.1619853
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1744994, upper bound: 0.1672193
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1730041, upper bound: 0.1687156
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1720710, upper bound: 0.1663415
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1678090, upper bound: 0.1706135
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1659089, upper bound: 0.1633635
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1661741, upper bound: 0.1632295
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1686560, upper bound: 0.1724337
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1694571, upper bound: 0.1716321
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1691800, upper bound: 0.1714874
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1700527, upper bound: 0.1705077
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1568537, upper bound: 0.1612094
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1568537, upper bound: 0.1612095
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1595693, upper bound: 0.1638388
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.19
Output dim: 1, lower bound: -0.1595693, upper bound: 0.1638388

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4549887, 0.4545702
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2500440, 0.2457872
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2679472, 0.2665257
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2660797, 0.2711394
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1865971, 0.1841130
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2546540, 0.2595075
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4070385, 0.4133135
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2122762, 0.2106498
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2679484, 0.2690618
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3040159, 0.3008975

Time for backsubstitution: 23.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2488

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2119

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1728015, upper bound: 0.1663715
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1736742, upper bound: 0.1653920
time: 3.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4549913, 0.4545677
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2500385, 0.2457927
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2679601, 0.2665131
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2661674, 0.2710516
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1865699, 0.1841403
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2547168, 0.2594448
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4070613, 0.4132904
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2122036, 0.2107224
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2679918, 0.2690179
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3040249, 0.3008887

Time for backsubstitution: 23.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2927
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1775

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1715394, upper bound: 0.1665552
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1706409, upper bound: 0.1670852
time: 3.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4632125, 0.4587210
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2505175, 0.2467610
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2639410, 0.2618165
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2658359, 0.2660197
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1790456, 0.1773556
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2565367, 0.2574255
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.3752637, 0.3792895
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2087594, 0.2055407
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2596918, 0.2576468
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.2974712, 0.2941422

Time for backsubstitution: 23.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1807
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1207
type: DSZ, layer: 3, pos: 318
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 159
type: DSZ, layer: 3, pos: 1787
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 61
type: DSZ, layer: 3, pos: 2927

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1646812, upper bound: 0.1589266
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1646812, upper bound: 0.1589266
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4614429, 0.4604903
1: 2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2495301, 0.2477485
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2601500, 0.2656074
3: -11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2650613, 0.2667940
4: -2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1765829, 0.1798178
5: -9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2563300, 0.2576321
6: -7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.3736854, 0.3808663
7: -4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2058891, 0.2084097
8: -1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2589592, 0.2583792
9: -14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.2963899, 0.2952228

Time for backsubstitution: 23.65 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.61 + 543.26 = 601.87 seconds
