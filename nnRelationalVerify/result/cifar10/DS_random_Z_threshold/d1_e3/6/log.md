## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 6)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.028245526200000003


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0361615, 0.0361615)
1: (-1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1307468, 0.1307468)
2: (-0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2118390, 0.2118390)
3: (-3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6015050, 0.6015050)
4: (-4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3087388, 0.3087388)
5: (-4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6962128, 0.6962130)
6: (-5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7997320, 0.7997320)
7: (-6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1995578, 0.1995578)
8: (-0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2736573, 0.2736573)
9: (-2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1492823, 0.1492823)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.34 + 28.78 = 37.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0282724, upper bound: 0.0282727

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 728

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 517

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282717, upper bound: 0.0282128
time: 21.61 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282116, upper bound: 0.0282714
time: 31.40 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 53.02 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 53.02
Output dim: 2, lower bound: -0.0282717, upper bound: 0.0282128
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 53.02
Output dim: 2, lower bound: -0.0282116, upper bound: 0.0282714

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0359993, 0.0359949
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1302670, 0.1302513
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108774, 0.2108349
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6007998, 0.6007774
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3089657, 0.3089464
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6951990, 0.6951673
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7970486, 0.7971079
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1972980, 0.1972255
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712864, 0.2713651
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1490805, 0.1490763

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 753

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2275

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282710, upper bound: 0.0282097
time: 17.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282689, upper bound: 0.0282112
time: 25.35 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0359949, 0.0359994
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1302513, 0.1302671
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108349, 0.2108774
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6007774, 0.6007998
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3089464, 0.3089656
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6951673, 0.6951990
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7971077, 0.7970486
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1972255, 0.1972981
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2713650, 0.2712862
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1490763, 0.1490805

Time for backsubstitution: 6.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2540

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2072

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282112, upper bound: 0.0282579
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281962, upper bound: 0.0282717
time: 53.23 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 62.73 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 62.73
Output dim: 2, lower bound: -0.0282710, upper bound: 0.0282097
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 62.73
Output dim: 2, lower bound: -0.0282689, upper bound: 0.0282112
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 62.73
Output dim: 2, lower bound: -0.0282112, upper bound: 0.0282579
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 62.73
Output dim: 2, lower bound: -0.0281962, upper bound: 0.0282717

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0358078, 0.0357985
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1282646, 0.1283090
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108384, 0.2107933
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6008017, 0.6007793
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086995, 0.3086742
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6952236, 0.6951888
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7968345, 0.7968992
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1966922, 0.1966057
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712443, 0.2713249
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1484211, 0.1484389

Time for backsubstitution: 6.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 753

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282692, upper bound: 0.0282071
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282695, upper bound: 0.0282073
time: 36.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0358029, 0.0358033
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1283246, 0.1282489
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108358, 0.2107960
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6008017, 0.6007793
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086935, 0.3086802
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6952202, 0.6951919
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7968402, 0.7968935
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1966783, 0.1966195
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712460, 0.2713233
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1484432, 0.1484169

Time for backsubstitution: 6.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3018

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3019

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282497, upper bound: 0.0281962
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282542, upper bound: 0.0281928
time: 3.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0356899, 0.0356821
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1267489, 0.1268807
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108123, 0.2108468
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6007159, 0.6007385
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3081974, 0.3081774
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6952708, 0.6953018
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7967520, 0.7966986
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1969311, 0.1970017
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2713778, 0.2712998
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1475000, 0.1475260

Time for backsubstitution: 6.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2602

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3456

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282015, upper bound: 0.0281485
time: 38.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281354, upper bound: 0.0282569
time: 24.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0356777, 0.0356944
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1268650, 0.1267646
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108043, 0.2108548
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6007161, 0.6007383
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3081582, 0.3082167
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6952701, 0.6953025
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7967577, 0.7966924
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1969290, 0.1970037
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2713785, 0.2712989
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1475218, 0.1475043

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3427

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281911, upper bound: 0.0282325
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281561, upper bound: 0.0282657
time: 10.04 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 19.90 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.90
Output dim: 2, lower bound: -0.0282692, upper bound: 0.0282071
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.90
Output dim: 2, lower bound: -0.0282695, upper bound: 0.0282073
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.90
Output dim: 2, lower bound: -0.0282497, upper bound: 0.0281962
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.90
Output dim: 2, lower bound: -0.0282542, upper bound: 0.0281928
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 19.90
Output dim: 2, lower bound: -0.0282015, upper bound: 0.0281485
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.90
Output dim: 2, lower bound: -0.0281354, upper bound: 0.0282569
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 19.90
Output dim: 2, lower bound: -0.0281911, upper bound: 0.0282325
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.90
Output dim: 2, lower bound: -0.0281561, upper bound: 0.0282657

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357874, 0.0357779
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1282287, 0.1282727
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108266, 0.2107816
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6002512, 0.6002474
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086967, 0.3086717
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6946707, 0.6946537
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7967579, 0.7968230
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1962813, 0.1962137
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712454, 0.2713253
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483338, 0.1483497

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2973

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 728

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282676, upper bound: 0.0282070
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282682, upper bound: 0.0282057
time: 3.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357872, 0.0357781
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1282284, 0.1282730
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108267, 0.2107815
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6002698, 0.6002288
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086967, 0.3086715
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6946883, 0.6946361
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7967579, 0.7968230
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1963002, 0.1961948
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712448, 0.2713259
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483319, 0.1483516

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2516

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 714

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282691, upper bound: 0.0282072
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282691, upper bound: 0.0282070
time: 3.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357191, 0.0357171
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1281436, 0.1280646
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2106556, 0.2106165
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5963330, 0.5963268
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3074352, 0.3074253
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6908708, 0.6908584
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7955408, 0.7956672
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1909460, 0.1909044
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711763, 0.2712559
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1484027, 0.1483772

Time for backsubstitution: 6.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3018

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3200

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282497, upper bound: 0.0281919
time: 50.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282458, upper bound: 0.0281964
time: 20.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357167, 0.0357195
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1281403, 0.1280679
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2106563, 0.2106158
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5963492, 0.5963106
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3074385, 0.3074219
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6908867, 0.6908422
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7956138, 0.7955942
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1909631, 0.1908872
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711787, 0.2712536
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1484035, 0.1483765

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3200

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 728

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281910
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282533, upper bound: 0.0281902
time: 5.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0352084, 0.0352177
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1255415, 0.1257137
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2088542, 0.2089842
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6010025, 0.6010697
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086241, 0.3086659
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6949785, 0.6950698
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7892356, 0.7889695
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1940476, 0.1942475
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2664944, 0.2662481
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1470236, 0.1470558

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 489

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 502

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281342, upper bound: 0.0282371
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281268, upper bound: 0.0282563
time: 55.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0356711, 0.0356862
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1265367, 0.1264489
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2098640, 0.2099596
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5989180, 0.5990001
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3079319, 0.3079847
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6924317, 0.6925569
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7966053, 0.7965368
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1908543, 0.1910879
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2683809, 0.2681984
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1473673, 0.1473531

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 714

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3019

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281369, upper bound: 0.0282507
time: 20.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281414, upper bound: 0.0282467
time: 55.09 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 81.96 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 81.96
Output dim: 2, lower bound: -0.0282676, upper bound: 0.0282070
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 81.96
Output dim: 2, lower bound: -0.0282682, upper bound: 0.0282057
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 81.96
Output dim: 2, lower bound: -0.0282691, upper bound: 0.0282072
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 81.96
Output dim: 2, lower bound: -0.0282691, upper bound: 0.0282070
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 81.96
Output dim: 2, lower bound: -0.0282497, upper bound: 0.0281919
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 81.96
Output dim: 2, lower bound: -0.0282458, upper bound: 0.0281964
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 81.96
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281910
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 81.96
Output dim: 2, lower bound: -0.0282533, upper bound: 0.0281902
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 81.96
Output dim: 2, lower bound: -0.0281342, upper bound: 0.0282371
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 81.96
Output dim: 2, lower bound: -0.0281268, upper bound: 0.0282563
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 81.96
Output dim: 2, lower bound: -0.0281369, upper bound: 0.0282507
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 81.96
Output dim: 2, lower bound: -0.0281414, upper bound: 0.0282467

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357720, 0.0357609
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1282282, 0.1282723
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108215, 0.2107766
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6002064, 0.6002049
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086734, 0.3086513
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6946244, 0.6946098
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7966335, 0.7967056
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1962617, 0.1961943
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712318, 0.2713120
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483312, 0.1483470

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2050

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3199

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282676, upper bound: 0.0281993
time: 22.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282610, upper bound: 0.0282077
time: 13.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357704, 0.0357624
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1282283, 0.1282723
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108216, 0.2107764
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6002088, 0.6002023
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086765, 0.3086482
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6946268, 0.6946074
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7966406, 0.7966985
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1962620, 0.1961940
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712320, 0.2713119
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483311, 0.1483471

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3019

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282682, upper bound: 0.0281680
time: 13.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282315, upper bound: 0.0282059
time: 12.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357268, 0.0357156
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1281303, 0.1281769
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108265, 0.2107813
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6002493, 0.6002086
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3087248, 0.3087003
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6946816, 0.6946295
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7965915, 0.7966620
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1962739, 0.1961683
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712296, 0.2713109
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483047, 0.1483245

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2514

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 754

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282643, upper bound: 0.0282021
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282649, upper bound: 0.0282024
time: 4.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357247, 0.0357177
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1281323, 0.1281749
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108265, 0.2107813
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6002498, 0.6002082
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3087257, 0.3086994
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6946821, 0.6946293
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7965972, 0.7966567
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1962737, 0.1961686
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712297, 0.2713108
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483048, 0.1483245

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3277

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 715

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282683, upper bound: 0.0282065
time: 11.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282683, upper bound: 0.0282073
time: 5.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357200, 0.0357181
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1281310, 0.1280507
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2106445, 0.2106051
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5963182, 0.5963116
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3074418, 0.3074320
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6908569, 0.6908443
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7955623, 0.7956882
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1908807, 0.1908378
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711403, 0.2712188
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483207, 0.1482908

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 502

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282496, upper bound: 0.0281923
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282483, upper bound: 0.0281923
time: 4.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357200, 0.0357180
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1281297, 0.1280519
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2106442, 0.2106054
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5963178, 0.5963120
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3074418, 0.3074320
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6908567, 0.6908445
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7955613, 0.7956889
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1908794, 0.1908392
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711394, 0.2712198
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483163, 0.1482952

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2964

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282448, upper bound: 0.0281960
time: 17.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282444, upper bound: 0.0281961
time: 13.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357013, 0.0357025
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1281399, 0.1280676
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2106512, 0.2106107
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5963044, 0.5962684
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3074152, 0.3074018
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6908410, 0.6907989
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7954893, 0.7954769
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1909435, 0.1908679
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711654, 0.2712404
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1484009, 0.1483738

Time for backsubstitution: 6.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 715

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282519, upper bound: 0.0281896
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282519, upper bound: 0.0281905
time: 3.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0356997, 0.0357041
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1281399, 0.1280676
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2106513, 0.2106106
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5963070, 0.5962658
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3074183, 0.3073987
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6908433, 0.6907963
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7954960, 0.7954698
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1909438, 0.1908676
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711655, 0.2712402
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1484008, 0.1483739

Time for backsubstitution: 6.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2973

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3231

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282518, upper bound: 0.0281655
time: 20.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282278, upper bound: 0.0281884
time: 3.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0352053, 0.0352145
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1255144, 0.1256892
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2087713, 0.2089102
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6008651, 0.6009436
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086072, 0.3086480
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6947646, 0.6948733
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7892280, 0.7889619
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1935871, 0.1938245
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2662817, 0.2660152
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1470111, 0.1470435

Time for backsubstitution: 6.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2602

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0280352, upper bound: 0.0282545
time: 18.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281266, upper bound: 0.0281648
time: 21.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0355872, 0.0356000
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1263558, 0.1262645
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2096839, 0.2097801
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5944493, 0.5945475
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3066738, 0.3067299
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6880825, 0.6882235
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7953067, 0.7953112
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1851219, 0.1853727
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2683114, 0.2681311
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1473267, 0.1473134

Time for backsubstitution: 6.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3214

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281368, upper bound: 0.0282159
time: 27.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281002, upper bound: 0.0282503
time: 34.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0355849, 0.0356023
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1263524, 0.1262679
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2096846, 0.2097794
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5944655, 0.5945313
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3066772, 0.3067266
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6880987, 0.6882075
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7953801, 0.7952383
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1851391, 0.1853555
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2683138, 0.2681289
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1473275, 0.1473126

Time for backsubstitution: 6.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281396, upper bound: 0.0282470
time: 17.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281403, upper bound: 0.0282462
time: 15.74 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 40.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282676, upper bound: 0.0281993
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282610, upper bound: 0.0282077
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282682, upper bound: 0.0281680
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282315, upper bound: 0.0282059
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282643, upper bound: 0.0282021
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282649, upper bound: 0.0282024
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282683, upper bound: 0.0282065
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282683, upper bound: 0.0282073
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282496, upper bound: 0.0281923
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282483, upper bound: 0.0281923
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282448, upper bound: 0.0281960
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282444, upper bound: 0.0281961
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282519, upper bound: 0.0281896
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282519, upper bound: 0.0281905
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282518, upper bound: 0.0281655
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0282278, upper bound: 0.0281884
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0280352, upper bound: 0.0282545
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0281266, upper bound: 0.0281648
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0281368, upper bound: 0.0282159
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0281002, upper bound: 0.0282503
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0281396, upper bound: 0.0282470
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 40.10
Output dim: 2, lower bound: -0.0281403, upper bound: 0.0282462

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357719, 0.0357608
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1282291, 0.1282721
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108194, 0.2107744
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6002011, 0.6001985
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086730, 0.3086510
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6946199, 0.6946040
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7966278, 0.7966990
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1962530, 0.1961846
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712280, 0.2713060
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483237, 0.1483356

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2514

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282658, upper bound: 0.0281996
time: 11.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282665, upper bound: 0.0281983
time: 3.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357719, 0.0357608
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1282281, 0.1282731
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108193, 0.2107745
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6001999, 0.6001997
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086730, 0.3086510
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6946189, 0.6946052
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7966268, 0.7966998
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1962520, 0.1961856
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712259, 0.2713081
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483199, 0.1483395

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2602

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 754

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282563, upper bound: 0.0282037
time: 7.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282568, upper bound: 0.0282030
time: 4.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357350, 0.0357274
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1279342, 0.1279780
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2103575, 0.2102959
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5994365, 0.5994084
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3077689, 0.3077208
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6938970, 0.6938510
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7970147, 0.7970703
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1956651, 0.1955661
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2713772, 0.2714553
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483015, 0.1483101

Time for backsubstitution: 6.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 715

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3277

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282610, upper bound: 0.0281679
time: 12.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282661, upper bound: 0.0281621
time: 3.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357130, 0.0357023
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1280731, 0.1281195
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2107813, 0.2107363
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5990591, 0.5990577
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086413, 0.3086175
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6934783, 0.6934645
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7959681, 0.7960355
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1954518, 0.1953875
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712212, 0.2713018
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483046, 0.1483244

Time for backsubstitution: 6.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2288

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2316

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282451, upper bound: 0.0281730
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282291, upper bound: 0.0281828
time: 5.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357135, 0.0357018
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1280729, 0.1281197
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2107815, 0.2107361
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5990984, 0.5990183
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086419, 0.3086169
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6935167, 0.6934261
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7959652, 0.7960382
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1954931, 0.1953462
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712204, 0.2713024
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483046, 0.1483244

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282434, upper bound: 0.0281992
time: 30.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282621, upper bound: 0.0281800
time: 6.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0356113, 0.0355996
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1270304, 0.1270793
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108328, 0.2107878
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6002502, 0.6002089
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3084842, 0.3084456
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6947341, 0.6946805
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7963719, 0.7964364
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1959386, 0.1958276
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712207, 0.2713019
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1478546, 0.1478752

Time for backsubstitution: 6.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2514

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3456

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282677, upper bound: 0.0281308
time: 8.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281593, upper bound: 0.0281963
time: 25.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0356066, 0.0356042
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1270368, 0.1270730
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108330, 0.2107876
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6002502, 0.6002089
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3084719, 0.3084587
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6947331, 0.6946814
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7963772, 0.7964312
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1959327, 0.1958343
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712208, 0.2713016
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1478555, 0.1478742

Time for backsubstitution: 6.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2964

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2050

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282675, upper bound: 0.0282026
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282650, upper bound: 0.0282064
time: 3.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357212, 0.0357195
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1281838, 0.1280990
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2106402, 0.2106010
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5963168, 0.5963100
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3074424, 0.3074328
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6908691, 0.6908565
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7955480, 0.7956729
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1908791, 0.1908363
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711469, 0.2712221
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483437, 0.1483083

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2288

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 546

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282485, upper bound: 0.0280877
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281457, upper bound: 0.0281906
time: 4.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357214, 0.0357193
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1281793, 0.1281035
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2106403, 0.2106009
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5963163, 0.5963100
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3074427, 0.3074326
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6908693, 0.6908562
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7955470, 0.7956736
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1908791, 0.1908363
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711436, 0.2712255
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483382, 0.1483139

Time for backsubstitution: 6.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2316

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3427

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282433, upper bound: 0.0281534
time: 7.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282082, upper bound: 0.0281877
time: 67.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0355868, 0.0355834
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1270219, 0.1269560
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2106577, 0.2106175
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5963056, 0.5962694
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3072182, 0.3071915
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6908934, 0.6908503
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7952645, 0.7952576
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1906058, 0.1905242
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711469, 0.2712221
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1479475, 0.1479213

Time for backsubstitution: 6.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281605, upper bound: 0.0281905
time: 9.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282518, upper bound: 0.0280998
time: 4.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0355822, 0.0355881
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1270282, 0.1269496
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2106580, 0.2106173
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5963056, 0.5962694
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3072051, 0.3072045
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6908925, 0.6908512
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7952697, 0.7952523
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1905997, 0.1905302
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711471, 0.2712219
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1479484, 0.1479204

Time for backsubstitution: 6.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 546

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2552

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282517, upper bound: 0.0281895
time: 10.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282515, upper bound: 0.0281899
time: 19.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0356999, 0.0357046
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1275292, 0.1274788
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2105485, 0.2105054
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5958641, 0.5958121
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3068972, 0.3068627
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6903298, 0.6902608
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7956638, 0.7956322
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1900762, 0.1899712
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711822, 0.2712510
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483467, 0.1483163

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3277

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282500, upper bound: 0.0281641
time: 15.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282507, upper bound: 0.0281634
time: 10.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0351506, 0.0351601
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1241249, 0.1242773
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2080296, 0.2081869
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5966089, 0.5968342
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3081286, 0.3082625
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6905622, 0.6908115
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7834797, 0.7834058
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1896834, 0.1900266
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2662889, 0.2660217
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1467766, 0.1468107

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2531

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2050

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0280343, upper bound: 0.0282516
time: 11.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0280318, upper bound: 0.0282547
time: 8.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0355522, 0.0355645
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1260530, 0.1259620
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2092033, 0.2093160
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5936570, 0.5937768
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3057463, 0.3058225
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6873276, 0.6874950
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7956779, 0.7956846
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1845322, 0.1848140
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2684544, 0.2682760
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1472905, 0.1472846

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3437

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0280814, upper bound: 0.0282514
time: 6.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281001, upper bound: 0.0282328
time: 26.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0354547, 0.0354652
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1247806, 0.1247092
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2097061, 0.2098017
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5944684, 0.5945343
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3065777, 0.3066003
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6882014, 0.6883100
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7948861, 0.7947431
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1847056, 0.1849209
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2682207, 0.2680354
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1466188, 0.1466064

Time for backsubstitution: 6.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2297

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2531

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281378, upper bound: 0.0282460
time: 6.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281384, upper bound: 0.0282455
time: 12.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0354477, 0.0354721
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1247937, 0.1246961
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2097069, 0.2098009
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5944684, 0.5945343
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3065507, 0.3066272
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6882010, 0.6883103
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7948847, 0.7947440
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1847045, 0.1849221
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2682203, 0.2680358
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1466213, 0.1466039

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 488

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2514

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281383, upper bound: 0.0282372
time: 20.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281310, upper bound: 0.0282448
time: 3.74 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 31.03 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282658, upper bound: 0.0281996
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282665, upper bound: 0.0281983
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282563, upper bound: 0.0282037
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282568, upper bound: 0.0282030
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282610, upper bound: 0.0281679
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282661, upper bound: 0.0281621
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282451, upper bound: 0.0281730
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282291, upper bound: 0.0281828
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282434, upper bound: 0.0281992
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282621, upper bound: 0.0281800
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282677, upper bound: 0.0281308
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0281593, upper bound: 0.0281963
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282675, upper bound: 0.0282026
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282650, upper bound: 0.0282064
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282485, upper bound: 0.0280877
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0281457, upper bound: 0.0281906
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282433, upper bound: 0.0281534
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282082, upper bound: 0.0281877
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0281605, upper bound: 0.0281905
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282518, upper bound: 0.0280998
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282517, upper bound: 0.0281895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282515, upper bound: 0.0281899
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282500, upper bound: 0.0281641
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0282507, upper bound: 0.0281634
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0280343, upper bound: 0.0282516
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0280318, upper bound: 0.0282547
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0280814, upper bound: 0.0282514
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0281001, upper bound: 0.0282328
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0281378, upper bound: 0.0282460
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0281384, upper bound: 0.0282455
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0281383, upper bound: 0.0282372
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 31.03
Output dim: 2, lower bound: -0.0281310, upper bound: 0.0282448

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0356417, 0.0356237
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1266573, 0.1267134
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108411, 0.2107969
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6002040, 0.6002012
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3085735, 0.3085246
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6947234, 0.6947074
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7961342, 0.7962042
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1958194, 0.1957498
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711353, 0.2712131
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1476149, 0.1476294

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2316

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2050

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282650, upper bound: 0.0281958
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282624, upper bound: 0.0281995
time: 3.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0356347, 0.0356306
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1266704, 0.1267004
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108420, 0.2107961
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6002040, 0.6002012
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3085467, 0.3085515
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6947231, 0.6947076
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7961328, 0.7962054
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1958182, 0.1957510
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711350, 0.2712134
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1476174, 0.1476269

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 754

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3413

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282653, upper bound: 0.0281615
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282284, upper bound: 0.0281968
time: 3.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357581, 0.0357475
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1281709, 0.1282157
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2107742, 0.2107297
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5990095, 0.5990484
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3085896, 0.3085682
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6934156, 0.6934404
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7960038, 0.7960744
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1954299, 0.1954048
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712178, 0.2712994
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483198, 0.1483393

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281649, upper bound: 0.0282031
time: 36.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282562, upper bound: 0.0281122
time: 3.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357587, 0.0357470
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1281707, 0.1282160
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2107745, 0.2107295
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5990489, 0.5990090
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3085903, 0.3085676
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6934540, 0.6934021
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7960010, 0.7960768
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1954711, 0.1953636
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712172, 0.2713001
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1483198, 0.1483393

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 546

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2602

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281975
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282521, upper bound: 0.0281982
time: 3.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357020, 0.0356974
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1278563, 0.1279000
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2103348, 0.2102814
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5993483, 0.5993140
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3077753, 0.3077199
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6938102, 0.6937587
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7969971, 0.7970490
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1956630, 0.1955599
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2713718, 0.2714437
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1482781, 0.1482834

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3427

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3231

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282596, upper bound: 0.0281422
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282355, upper bound: 0.0281662
time: 3.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0357050, 0.0356944
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1278561, 0.1279000
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2103429, 0.2102732
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5993421, 0.5993202
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3077682, 0.3077270
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6938045, 0.6937644
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7969933, 0.7970526
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1956589, 0.1955639
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2713656, 0.2714499
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1482748, 0.1482867

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 489

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 754

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282614, upper bound: 0.0281585
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282619, upper bound: 0.0281577
time: 3.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0356600, 0.0356534
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1280437, 0.1280901
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2107527, 0.2107027
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.5987957, 0.5986812
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3085600, 0.3085362
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6931419, 0.6930255
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7953758, 0.7953732
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1951787, 0.1950165
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2711792, 0.2712617
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1482991, 0.1483189

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2531

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2602

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282580, upper bound: 0.0281773
time: 3.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282574, upper bound: 0.0281772
time: 3.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0351469, 0.0351180
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1258634, 0.1258719
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2089702, 0.2088298
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6005816, 0.6004957
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3089725, 0.3088722
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6945009, 0.6943870
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7886434, 0.7889210
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1931845, 0.1929441
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2661688, 0.2664180
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1473843, 0.1473987

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2316

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2602

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282637, upper bound: 0.0281271
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282630, upper bound: 0.0281267
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0354581, 0.0354498
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1249912, 0.1250269
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108155, 0.2107703
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6002519, 0.6002104
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3081685, 0.3081397
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6947844, 0.6947289
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7961679, 0.7962251
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1954213, 0.1953061
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712090, 0.2712895
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1472156, 0.1472336

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 502

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2973

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282590, upper bound: 0.0281965
time: 38.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282601, upper bound: 0.0281939
time: 7.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0354522, 0.0354558
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1250014, 0.1250273
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2108157, 0.2107701
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6002519, 0.6002104
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3081530, 0.3081557
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6947806, 0.6947329
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7961717, 0.7962217
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1954044, 0.1953232
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2712086, 0.2712898
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1472153, 0.1472343

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 3456
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2316

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2552

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282647, upper bound: 0.0282043
time: 68.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282645, upper bound: 0.0282052
time: 74.12 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 149.38 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282650, upper bound: 0.0281958
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282624, upper bound: 0.0281995
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282653, upper bound: 0.0281615
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282284, upper bound: 0.0281968
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0281649, upper bound: 0.0282031
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282562, upper bound: 0.0281122
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281975
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282521, upper bound: 0.0281982
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282596, upper bound: 0.0281422
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282355, upper bound: 0.0281662
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282614, upper bound: 0.0281585
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282619, upper bound: 0.0281577
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282580, upper bound: 0.0281773
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282574, upper bound: 0.0281772
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282637, upper bound: 0.0281271
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282630, upper bound: 0.0281267
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282590, upper bound: 0.0281965
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282601, upper bound: 0.0281939
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282647, upper bound: 0.0282043
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 149.38
Output dim: 2, lower bound: -0.0282645, upper bound: 0.0282052
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 149.38
Output dim: 2, lower bound: -0.0282485, upper bound: 0.0280877
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 149.38
Output dim: 2, lower bound: -0.0282518, upper bound: 0.0280998
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 149.38
Output dim: 2, lower bound: -0.0282517, upper bound: 0.0281895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 149.38
Output dim: 2, lower bound: -0.0282515, upper bound: 0.0281899
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 149.38
Output dim: 2, lower bound: -0.0282500, upper bound: 0.0281641
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 149.38
Output dim: 2, lower bound: -0.0282507, upper bound: 0.0281634
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 149.38
Output dim: 2, lower bound: -0.0280343, upper bound: 0.0282516
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 149.38
Output dim: 2, lower bound: -0.0280318, upper bound: 0.0282547
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 149.38
Output dim: 2, lower bound: -0.0280814, upper bound: 0.0282514
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 149.38
Output dim: 2, lower bound: -0.0281378, upper bound: 0.0282460

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 37.12 + 1773.88 = 1811.01 seconds
