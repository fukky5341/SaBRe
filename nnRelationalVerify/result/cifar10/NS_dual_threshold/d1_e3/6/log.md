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
execution time: IAR + RelationalAnalysis = 7.53 + 29.38 = 36.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0282724, upper bound: 0.0282727

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3456
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: B, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 546

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282713, upper bound: 0.0281695
time: 12.09 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282713, upper bound: 0.0282723
time: 26.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 38.78 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 38.78
Output dim: 2, lower bound: -0.0282713, upper bound: 0.0281695
NS_A2, status: Status.UNKNOWN, split count: 1, time: 38.78
Output dim: 2, lower bound: -0.0282713, upper bound: 0.0282723

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 1.6286296, 1.8220252, 1.6286290, 1.8220793, -0.0357856, 0.0357330
1: -1.5907009, -0.9892200, -1.5907340, -0.9892183, -0.1304107, 0.1304453
2: -0.1710155, 0.2519290, -0.1710206, 0.2520331, -0.2110926, 0.2110299
3: -3.9573860, -2.7509122, -3.9573913, -2.7509120, -0.6014493, 0.6014519
4: -4.2525039, -3.2467523, -4.2525034, -3.2466607, -0.3080937, 0.3080136
5: -4.5817533, -3.3530216, -4.5817599, -3.3529532, -0.6956310, 0.6955931
6: -5.3589525, -3.6091208, -5.3593044, -3.6091208, -0.7968552, 0.7971585
7: -6.2234945, -5.0624032, -6.2234950, -5.0622363, -0.1981198, 0.1979160
8: -0.8317333, -0.2983226, -0.8318503, -0.2983227, -0.2729049, 0.2729931
9: -2.3948431, -1.9256269, -2.3949237, -1.9256275, -0.1486220, 0.1486828

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3456
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3456

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282707, upper bound: 0.0280667
time: 5.94 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282708, upper bound: 0.0281686
time: 57.22 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 1.6281203, 1.8224572, 1.6286249, 1.8224572, -0.0366388, 0.0357630
1: -1.5909625, -0.9887190, -1.5909631, -0.9892013, -0.1304308, 0.1313228
2: -0.1722020, 0.2527471, -0.1710544, 0.2527486, -0.2130070, 0.2110999
3: -3.9575393, -2.7509103, -3.9574201, -2.7509112, -0.6015964, 0.6015139
4: -4.2534103, -3.2459192, -4.2525048, -3.2459168, -0.3094708, 0.3080735
5: -4.5824680, -3.3525584, -4.5817838, -3.3525596, -0.6970029, 0.6956900
6: -5.3614950, -3.6063557, -5.3615360, -3.6091208, -0.7970731, 0.8019680
7: -6.2251530, -5.0609388, -6.2235026, -5.0609355, -0.2012565, 0.1980189
8: -0.8326685, -0.2973220, -0.8326674, -0.2983216, -0.2729760, 0.2742677
9: -2.3954995, -1.9249188, -2.3955050, -1.9256268, -0.1486704, 0.1499934

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3456
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3456

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282707, upper bound: 0.0281677
time: 19.84 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282707, upper bound: 0.0282723
time: 3.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.69 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 29.69
Output dim: 2, lower bound: -0.0282707, upper bound: 0.0280667
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 29.69
Output dim: 2, lower bound: -0.0282708, upper bound: 0.0281686
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 29.69
Output dim: 2, lower bound: -0.0282707, upper bound: 0.0281677
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 29.69
Output dim: 2, lower bound: -0.0282707, upper bound: 0.0282723

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: 1.6287028, 1.8215942, 1.6286305, 1.8217307, -0.0353791, 0.0353052
1: -1.5905089, -0.9902829, -1.5907339, -0.9900958, -0.1292985, 0.1293896
2: -0.1704974, 0.2496880, -0.1709776, 0.2502140, -0.2089612, 0.2088737
3: -3.9573576, -2.7512717, -3.9572425, -2.7511957, -0.6011372, 0.6009203
4: -4.2523541, -3.2476935, -4.2524338, -3.2474599, -0.3074466, 0.3071760
5: -4.5814600, -3.3538842, -4.5816002, -3.3536644, -0.6948440, 0.6947248
6: -5.3521948, -3.6097040, -5.3538356, -3.6091208, -0.7894752, 0.7898932
7: -6.2228637, -5.0656567, -6.2234921, -5.0649238, -0.1952028, 0.1949250
8: -0.8267707, -0.2994072, -0.8278544, -0.2983229, -0.2679470, 0.2679009
9: -2.3948631, -1.9259229, -2.3948998, -1.9258701, -0.1481403, 0.1482599

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3019

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282515, upper bound: 0.0280508
time: 3.44 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0280509
time: 16.20 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 1.6286296, 1.8220236, 1.6286292, 1.8220781, -0.0357844, 0.0352664
1: -1.5907009, -0.9892261, -1.5907340, -0.9892219, -0.1304057, 0.1292725
2: -0.1710125, 0.2519260, -0.1710178, 0.2520305, -0.2110566, 0.2088662
3: -3.9573801, -2.7510200, -3.9573865, -2.7510033, -0.6013026, 0.6014814
4: -4.2525034, -3.2467823, -4.2525034, -3.2466841, -0.3079542, 0.3079867
5: -4.5817475, -3.3530240, -4.5817547, -3.3529553, -0.6955545, 0.6950622
6: -5.3587570, -3.6091208, -5.3591495, -3.6091208, -0.7892137, 0.7971275
7: -6.2234945, -5.0624084, -6.2234950, -5.0622406, -0.1981154, 0.1946625
8: -0.8317329, -0.2983228, -0.8318497, -0.2983227, -0.2680207, 0.2729921
9: -2.3948429, -1.9256535, -2.3949237, -1.9256480, -0.1486218, 0.1482103

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3019

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0281482
time: 25.35 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0281535
time: 3.75 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: 1.6281933, 1.8220266, 1.6286262, 1.8221087, -0.0362314, 0.0353354
1: -1.5907719, -0.9897801, -1.5909631, -0.9900793, -0.1293187, 0.1302677
2: -0.1716839, 0.2505061, -0.1710116, 0.2509297, -0.2108705, 0.2089434
3: -3.9575107, -2.7512698, -3.9572713, -2.7511947, -0.6012821, 0.6009818
4: -4.2532616, -3.2468600, -4.2524347, -3.2467165, -0.3088111, 0.3072363
5: -4.5821743, -3.3534212, -4.5816250, -3.3532708, -0.6962106, 0.6948212
6: -5.3547368, -3.6069388, -5.3560667, -3.6091208, -0.7896936, 0.7947026
7: -6.2245216, -5.0641923, -6.2235003, -5.0636230, -0.1983374, 0.1950284
8: -0.8277056, -0.2984064, -0.8286718, -0.2983215, -0.2680175, 0.2691759
9: -2.3955204, -1.9252142, -2.3954809, -1.9258695, -0.1481888, 0.1495725

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3019

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282515, upper bound: 0.0281539
time: 3.77 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0281536
time: 4.21 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: 1.6281204, 1.8224556, 1.6286249, 1.8224562, -0.0366375, 0.0352966
1: -1.5909625, -0.9887249, -1.5909631, -0.9892055, -0.1304258, 0.1301516
2: -0.1721989, 0.2527440, -0.1710523, 0.2527464, -0.2129710, 0.2089359
3: -3.9575338, -2.7510180, -3.9574165, -2.7510021, -0.6014483, 0.6015437
4: -4.2534103, -3.2459486, -4.2525048, -3.2459402, -0.3093321, 0.3080472
5: -4.5824618, -3.3525615, -4.5817800, -3.3525605, -0.6969271, 0.6951580
6: -5.3612981, -3.6063557, -5.3613820, -3.6091208, -0.7894311, 0.8019370
7: -6.2251530, -5.0609446, -6.2235026, -5.0609398, -0.2012522, 0.1947656
8: -0.8326678, -0.2973223, -0.8326671, -0.2983216, -0.2680920, 0.2742667
9: -2.3954997, -1.9249458, -2.3955050, -1.9256476, -0.1486703, 0.1495245

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3019

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282515, upper bound: 0.0282564
time: 7.83 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0282563
time: 26.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 40.64 seconds
NS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 40.64
Output dim: 2, lower bound: -0.0282515, upper bound: 0.0280508
NS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 40.64
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0280509
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 40.64
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0281482
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 40.64
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0281535
NS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 40.64
Output dim: 2, lower bound: -0.0282515, upper bound: 0.0281539
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 40.64
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0281536
NS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 40.64
Output dim: 2, lower bound: -0.0282515, upper bound: 0.0282564
NS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 40.64
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0282563

## BFS NS instance: NS_A1_A1_A1

### Backsubstitution after applying NS history:
0: 1.6287460, 1.8215500, 1.6286314, 1.8216909, -0.0352959, 0.0352590
1: -1.5905893, -0.9906122, -1.5907339, -0.9903702, -0.1289289, 0.1290070
2: -0.1700167, 0.2496769, -0.1705873, 0.2502141, -0.2084108, 0.2083316
3: -3.9553380, -2.7515035, -3.9554987, -2.7511966, -0.5981536, 0.5960013
4: -4.2519441, -3.2477407, -4.2520747, -3.2474713, -0.3067446, 0.3059434
5: -4.5794544, -3.3541145, -4.5798664, -3.3536651, -0.6919055, 0.6899580
6: -5.3506413, -3.6104059, -5.3524466, -3.6091208, -0.7880776, 0.7882215
7: -6.2212381, -5.0658579, -6.2220716, -5.0649238, -0.1922918, 0.1895472
8: -0.8267174, -0.2994400, -0.8278080, -0.2983245, -0.2678977, 0.2678307
9: -2.3948026, -1.9259384, -2.3948517, -1.9258704, -0.1480780, 0.1481872

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3413

## Relational analysis of NS_A1_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0280139
time: 3.11 seconds

## Relational analysis of NS_A1_A1_A1_A2

### Relational analysis result of NS_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282504, upper bound: 0.0280502
time: 3.53 seconds

## BFS NS instance: NS_A1_A1_A2

### Backsubstitution after applying NS history:
0: 1.6287031, 1.8215895, 1.6286308, 1.8217263, -0.0353729, 0.0352201
1: -1.5905089, -0.9905401, -1.5907339, -0.9903339, -0.1291897, 0.1290745
2: -0.1702508, 0.2496881, -0.1707527, 0.2502140, -0.2086246, 0.2087269
3: -3.9558239, -2.7512722, -3.9559004, -2.7511959, -0.5964675, 0.6007869
4: -4.2520108, -3.2476952, -4.2521324, -3.2474608, -0.3061850, 0.3071430
5: -4.5799279, -3.3538842, -4.5802593, -3.3536651, -0.6902838, 0.6945882
6: -5.3516183, -3.6097040, -5.3533273, -3.6091208, -0.7878780, 0.7893970
7: -6.2215710, -5.0656562, -6.2223625, -5.0649233, -0.1894874, 0.1948800
8: -0.8267523, -0.2994076, -0.8278385, -0.2983232, -0.2678769, 0.2678807
9: -2.3948555, -1.9259232, -2.3948932, -1.9258701, -0.1480954, 0.1482555

Time for backsubstitution: 5.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3019

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3413

## Relational analysis of NS_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282179, upper bound: 0.0280497
time: 3.60 seconds

## Relational analysis of NS_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0280493
time: 3.47 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: 1.6286306, 1.8219841, 1.6286726, 1.8220339, -0.0357382, 0.0351832
1: -1.5907009, -0.9895002, -1.5908146, -0.9895517, -0.1300230, 0.1289028
2: -0.1706223, 0.2519256, -0.1705375, 0.2520193, -0.2105147, 0.2083159
3: -3.9556365, -2.7510219, -3.9553671, -2.7512345, -0.5963833, 0.5984986
4: -4.2521458, -3.2467933, -4.2520938, -3.2467306, -0.3067219, 0.3072844
5: -4.5800142, -3.3530240, -4.5797491, -3.3531861, -0.6907878, 0.6921237
6: -5.3573689, -3.6091208, -5.3575959, -3.6098237, -0.7875423, 0.7957288
7: -6.2220731, -5.0624084, -6.2218695, -5.0624428, -0.1927375, 0.1917512
8: -0.8316860, -0.2983243, -0.8317964, -0.2983555, -0.2679505, 0.2729430
9: -2.3947945, -1.9256539, -2.3948641, -1.9256637, -0.1485491, 0.1481480

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3413

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282179, upper bound: 0.0281485
time: 3.35 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0281481
time: 22.38 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: 1.6286304, 1.8220192, 1.6286297, 1.8220731, -0.0356993, 0.0352602
1: -1.5907009, -0.9894639, -1.5907340, -0.9894797, -0.1300906, 0.1291635
2: -0.1707875, 0.2519260, -0.1707717, 0.2520306, -0.2109100, 0.2085295
3: -3.9560370, -2.7510195, -3.9558527, -2.7510035, -0.6011689, 0.5968126
4: -4.2522025, -3.2467833, -4.2521601, -3.2466850, -0.3079212, 0.3067247
5: -4.5804071, -3.3530240, -4.5802226, -3.3529551, -0.6954184, 0.6905017
6: -5.3582487, -3.6091208, -5.3585711, -3.6091208, -0.7887173, 0.7955303
7: -6.2223649, -5.0624084, -6.2222018, -5.0622406, -0.1980705, 0.1889471
8: -0.8317168, -0.2983232, -0.8318316, -0.2983230, -0.2680008, 0.2729220
9: -2.3948357, -1.9256539, -2.3949158, -1.9256479, -0.1486174, 0.1481654

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3019

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3413

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0281147
time: 17.57 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0281531
time: 7.39 seconds

## BFS NS instance: NS_A2_A1_A1

### Backsubstitution after applying NS history:
0: 1.6282365, 1.8219820, 1.6286272, 1.8220690, -0.0361483, 0.0352892
1: -1.5908521, -0.9901097, -1.5909631, -0.9903538, -0.1289489, 0.1298851
2: -0.1712029, 0.2504950, -0.1706214, 0.2509298, -0.2103200, 0.2084014
3: -3.9554918, -2.7515011, -3.9555283, -2.7511961, -0.5982990, 0.5960624
4: -4.2528505, -3.2469075, -4.2520766, -3.2467275, -0.3081088, 0.3060039
5: -4.5801682, -3.3536518, -4.5798907, -3.3532710, -0.6932724, 0.6900545
6: -5.3531852, -3.6076407, -5.3546782, -3.6091208, -0.7882950, 0.7930316
7: -6.2228966, -5.0643945, -6.2220788, -5.0636230, -0.1954264, 0.1896504
8: -0.8276520, -0.2984395, -0.8286256, -0.2983234, -0.2679688, 0.2691059
9: -2.3954604, -1.9252299, -2.3954329, -1.9258701, -0.1481266, 0.1495000

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198
type: B, layer: 1, pos: 3019

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3413

## Relational analysis of NS_A2_A1_A1_A1

### Relational analysis result of NS_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0281164
time: 4.56 seconds

## Relational analysis of NS_A2_A1_A1_A2

### Relational analysis result of NS_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0281508
time: 56.78 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: 1.6281937, 1.8220215, 1.6286268, 1.8221042, -0.0362252, 0.0352503
1: -1.5907719, -0.9900376, -1.5909631, -0.9903173, -0.1292096, 0.1299525
2: -0.1714374, 0.2505062, -0.1707865, 0.2509297, -0.2105337, 0.2087969
3: -3.9559774, -2.7512701, -3.9559302, -2.7511947, -0.5966129, 0.6008486
4: -4.2529168, -3.2468600, -4.2521343, -3.2467179, -0.3075491, 0.3072031
5: -4.5806417, -3.3534222, -4.5802836, -3.3532710, -0.6916504, 0.6946853
6: -5.3541603, -3.6069388, -5.3555593, -3.6091208, -0.7880964, 0.7942067
7: -6.2232294, -5.0641928, -6.2223701, -5.0636225, -0.1926219, 0.1949835
8: -0.8276871, -0.2984072, -0.8286558, -0.2983220, -0.2679479, 0.2691562
9: -2.3955123, -1.9252144, -2.3954742, -1.9258702, -0.1481438, 0.1495681

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3413

## Relational analysis of NS_A2_A1_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282179, upper bound: 0.0281535
time: 3.81 seconds

## Relational analysis of NS_A2_A1_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0281534
time: 4.01 seconds

## BFS NS instance: NS_A2_A2_A1

### Backsubstitution after applying NS history:
0: 1.6281635, 1.8224113, 1.6286259, 1.8224167, -0.0365544, 0.0352504
1: -1.5910429, -0.9890541, -1.5909631, -0.9894800, -0.1300560, 0.1297692
2: -0.1717183, 0.2527327, -0.1706618, 0.2527462, -0.2124210, 0.2083933
3: -3.9555144, -2.7512496, -3.9556723, -2.7510033, -0.5984669, 0.5966237
4: -4.2530003, -3.2459950, -4.2521472, -3.2459509, -0.3086302, 0.3068146
5: -4.5804563, -3.3527918, -4.5800462, -3.3525612, -0.6939893, 0.6903913
6: -5.3597460, -3.6070585, -5.3599925, -3.6091208, -0.7880330, 0.8002656
7: -6.2235274, -5.0611467, -6.2220817, -5.0609393, -0.1983413, 0.1893878
8: -0.8326147, -0.2973551, -0.8326209, -0.2983229, -0.2680428, 0.2741965
9: -2.3954396, -1.9249614, -2.3954568, -1.9256480, -0.1486080, 0.1494518

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3413

## Relational analysis of NS_A2_A2_A1_A1

### Relational analysis result of NS_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0282178
time: 20.74 seconds

## Relational analysis of NS_A2_A2_A1_A2

### Relational analysis result of NS_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0282552
time: 16.24 seconds

## BFS NS instance: NS_A2_A2_A2

### Backsubstitution after applying NS history:
0: 1.6281207, 1.8224506, 1.6286258, 1.8224516, -0.0366313, 0.0352115
1: -1.5909625, -0.9889822, -1.5909631, -0.9894434, -0.1303167, 0.1298365
2: -0.1719525, 0.2527439, -0.1708273, 0.2527464, -0.2126344, 0.2087890
3: -3.9560001, -2.7510185, -3.9560723, -2.7510023, -0.5967805, 0.6014094
4: -4.2530665, -3.2459493, -4.2522049, -3.2459412, -0.3080704, 0.3080140
5: -4.5809302, -3.3525612, -4.5804381, -3.3525615, -0.6923671, 0.6950221
6: -5.3607216, -3.6063557, -5.3608737, -3.6091208, -0.7878346, 0.8014406
7: -6.2238598, -5.0609446, -6.2223730, -5.0609393, -0.1955370, 0.1947208
8: -0.8326495, -0.2973225, -0.8326513, -0.2983220, -0.2680218, 0.2742468
9: -2.3954916, -1.9249458, -2.3954985, -1.9256480, -0.1486254, 0.1495202

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3413

## Relational analysis of NS_A2_A2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282179, upper bound: 0.0282560
time: 3.20 seconds

## Relational analysis of NS_A2_A2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282549, upper bound: 0.0282554
time: 22.47 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.72 seconds
NS_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0280139
NS_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282504, upper bound: 0.0280502
NS_A1_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282179, upper bound: 0.0280497
NS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0280493
NS_A1_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282179, upper bound: 0.0281485
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0281481
NS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0281147
NS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0281531
NS_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0281164
NS_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0281508
NS_A2_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282179, upper bound: 0.0281535
NS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0281534
NS_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0282178
NS_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0282552
NS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282179, upper bound: 0.0282560
NS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 2, lower bound: -0.0282549, upper bound: 0.0282554

## BFS NS instance: NS_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: 1.6287557, 1.8215498, 1.6286396, 1.8216913, -0.0352795, 0.0352449
1: -1.5905893, -0.9907575, -1.5907339, -0.9904999, -0.1287880, 0.1288424
2: -0.1699909, 0.2492965, -0.1705651, 0.2498888, -0.2080197, 0.2078918
3: -3.9553225, -2.7521572, -3.9554849, -2.7517567, -0.5975764, 0.5953317
4: -4.2513585, -3.2477422, -4.2515659, -3.2474718, -0.3061324, 0.3054194
5: -4.5794392, -3.3552184, -4.5798540, -3.3546095, -0.6909456, 0.6888400
6: -5.3505893, -3.6104059, -5.3524027, -3.6091208, -0.7880204, 0.7881724
7: -6.2212372, -5.0676837, -6.2220702, -5.0664916, -0.1906990, 0.1876911
8: -0.8250946, -0.2994404, -0.8264199, -0.2983247, -0.2662517, 0.2664222
9: -2.3948016, -1.9259987, -2.3948507, -1.9259216, -0.1480187, 0.1481181

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 489

## Relational analysis of NS_A1_A1_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0279891
time: 20.77 seconds

## Relational analysis of NS_A1_A1_A1_A1_A2

### Relational analysis result of NS_A1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0280088
time: 7.92 seconds

## BFS NS instance: NS_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: 1.6287498, 1.8215830, 1.6286381, 1.8216909, -0.0352833, 0.0352795
1: -1.5908376, -0.9906022, -1.5907339, -0.9903719, -0.1291814, 0.1289034
2: -0.1707621, 0.2496712, -0.1705868, 0.2502092, -0.2090038, 0.2079184
3: -3.9566832, -2.7514498, -3.9554961, -2.7512050, -0.5994868, 0.5955772
4: -4.2521248, -3.2464731, -4.2520742, -3.2474709, -0.3067083, 0.3072209
5: -4.5817528, -3.3539591, -4.5798631, -3.3536777, -0.6941900, 0.6893446
6: -5.3506174, -3.6103377, -5.3524141, -3.6091208, -0.7880385, 0.7882822
7: -6.2249641, -5.0659084, -6.2220712, -5.0649672, -0.1960897, 0.1878614
8: -0.8269454, -0.2960488, -0.8278071, -0.2983245, -0.2670717, 0.2712196
9: -2.3949594, -1.9259343, -2.3948517, -1.9258840, -0.1482374, 0.1481502

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 489

## Relational analysis of NS_A1_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282243, upper bound: 0.0280482
time: 9.73 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0280470
time: 3.17 seconds

## BFS NS instance: NS_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: 1.6287099, 1.8215895, 1.6286346, 1.8217590, -0.0353934, 0.0352077
1: -1.5905089, -0.9905417, -1.5909824, -0.9903237, -0.1290860, 0.1293270
2: -0.1702504, 0.2496831, -0.1714977, 0.2502083, -0.2082115, 0.2093202
3: -3.9558213, -2.7512803, -3.9572446, -2.7511425, -0.5960443, 0.6021205
4: -4.2520103, -3.2476947, -4.2523136, -3.2461939, -0.3074623, 0.3071097
5: -4.5799246, -3.3538961, -4.5825572, -3.3535109, -0.6896706, 0.6968729
6: -5.3515840, -3.6097040, -5.3533025, -3.6090527, -0.7879398, 0.7893587
7: -6.2215705, -5.0656996, -6.2260880, -5.0649734, -0.1878018, 0.1986782
8: -0.8267511, -0.2994076, -0.8280669, -0.2949324, -0.2712657, 0.2670555
9: -2.3948555, -1.9259372, -2.3950496, -1.9258660, -0.1480583, 0.1484151

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 489

## Relational analysis of NS_A1_A1_A2_B2_B1

### Relational analysis result of NS_A1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282287, upper bound: 0.0280472
time: 57.06 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2

### Relational analysis result of NS_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0280483
time: 10.04 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 1.6286373, 1.8219841, 1.6286767, 1.8220668, -0.0357588, 0.0351706
1: -1.5907009, -0.9895017, -1.5910628, -0.9895419, -0.1299194, 0.1291553
2: -0.1706217, 0.2519207, -0.1712826, 0.2520136, -0.2101015, 0.2089089
3: -3.9556336, -2.7510295, -3.9567125, -2.7511814, -0.5959601, 0.5998323
4: -4.2521448, -3.2467928, -4.2522769, -3.2454634, -0.3079991, 0.3072487
5: -4.5800123, -3.3530364, -4.5820470, -3.3530321, -0.6901731, 0.6944084
6: -5.3573356, -3.6091208, -5.3575721, -3.6097560, -0.7876024, 0.7956901
7: -6.2220731, -5.0624528, -6.2255950, -5.0624928, -0.1910518, 0.1955492
8: -0.8316852, -0.2983243, -0.8320241, -0.2949644, -0.2713397, 0.2721175
9: -2.3947945, -1.9256680, -2.3950205, -1.9256589, -0.1485119, 0.1483077

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 489

## Relational analysis of NS_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282288, upper bound: 0.0281466
time: 10.31 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281450
time: 3.39 seconds

## BFS NS instance: NS_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 1.6286401, 1.8220193, 1.6286380, 1.8220731, -0.0356829, 0.0352460
1: -1.5907009, -0.9896097, -1.5907340, -0.9896092, -0.1299497, 0.1289991
2: -0.1707614, 0.2515455, -0.1707494, 0.2517051, -0.2105189, 0.2080896
3: -3.9560208, -2.7516742, -3.9558394, -2.7515640, -0.6005921, 0.5961442
4: -4.2516179, -3.2467840, -4.2516508, -3.2466855, -0.3073097, 0.3062005
5: -4.5803928, -3.3541288, -4.5802097, -3.3539004, -0.6944575, 0.6893845
6: -5.3581953, -3.6091208, -5.3585272, -3.6091208, -0.7886600, 0.7954805
7: -6.2223630, -5.0642333, -6.2222009, -5.0638080, -0.1964777, 0.1870910
8: -0.8300937, -0.2983235, -0.8304423, -0.2983232, -0.2663553, 0.2715138
9: -2.3948348, -1.9257138, -2.3949149, -1.9256997, -0.1485581, 0.1480962

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3019

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 489

## Relational analysis of NS_A1_A2_B2_A1_A1

### Relational analysis result of NS_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0280927
time: 37.63 seconds

## Relational analysis of NS_A1_A2_B2_A1_A2

### Relational analysis result of NS_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281122
time: 16.37 seconds

## BFS NS instance: NS_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 1.6286340, 1.8220521, 1.6286365, 1.8220731, -0.0356868, 0.0352807
1: -1.5909492, -0.9894542, -1.5907340, -0.9894809, -0.1303430, 0.1290600
2: -0.1715326, 0.2519202, -0.1707716, 0.2520257, -0.2115032, 0.2081164
3: -3.9573820, -2.7509665, -3.9558508, -2.7510121, -0.6025026, 0.5963898
4: -4.2523866, -3.2455156, -4.2521596, -3.2466850, -0.3078911, 0.3080024
5: -4.5827060, -3.3528700, -4.5802202, -3.3529677, -0.6977022, 0.6898897
6: -5.3582239, -3.6090527, -5.3585377, -3.6091208, -0.7886786, 0.7955910
7: -6.2260904, -5.0624580, -6.2222023, -5.0622835, -0.2018686, 0.1872613
8: -0.8319440, -0.2949324, -0.8318307, -0.2983230, -0.2671757, 0.2763109
9: -2.3949924, -1.9256494, -2.3949158, -1.9256620, -0.1487769, 0.1481282

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 489

## Relational analysis of NS_A1_A2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282288, upper bound: 0.0281505
time: 25.74 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281512
time: 15.34 seconds

## BFS NS instance: NS_A2_A1_A1_A1

### Backsubstitution after applying NS history:
0: 1.6282462, 1.8219819, 1.6286355, 1.8220690, -0.0361318, 0.0352751
1: -1.5908521, -0.9902552, -1.5909631, -0.9904829, -0.1288080, 0.1297204
2: -0.1711770, 0.2501144, -0.1705988, 0.2506044, -0.2099290, 0.2079616
3: -3.9554763, -2.7521553, -3.9555149, -2.7517557, -0.5977213, 0.5953939
4: -4.2522655, -3.2469075, -4.2515674, -3.2467284, -0.3074970, 0.3054794
5: -4.5801539, -3.3547554, -4.5798793, -3.3542156, -0.6923125, 0.6889373
6: -5.3531322, -3.6076407, -5.3546333, -3.6091208, -0.7882383, 0.7929819
7: -6.2228956, -5.0662198, -6.2220774, -5.0651903, -0.1938336, 0.1877944
8: -0.8260293, -0.2984397, -0.8272362, -0.2983235, -0.2663230, 0.2676972
9: -2.3954589, -1.9252907, -2.3954320, -1.9259216, -0.1480671, 0.1494306

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198
type: B, layer: 1, pos: 3019

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 489

## Relational analysis of NS_A2_A1_A1_A1_A1

### Relational analysis result of NS_A2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0280926
time: 24.35 seconds

## Relational analysis of NS_A2_A1_A1_A1_A2

### Relational analysis result of NS_A2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0281127
time: 34.01 seconds

## BFS NS instance: NS_A2_A1_A1_A2

### Backsubstitution after applying NS history:
0: 1.6282405, 1.8220152, 1.6286336, 1.8220690, -0.0361356, 0.0353097
1: -1.5911003, -0.9900998, -1.5909631, -0.9903554, -0.1292014, 0.1297815
2: -0.1719480, 0.2504890, -0.1706209, 0.2509249, -0.2109130, 0.2079885
3: -3.9568372, -2.7514474, -3.9555264, -2.7512038, -0.5996323, 0.5956390
4: -4.2530317, -3.2456393, -4.2520761, -3.2467275, -0.3080739, 0.3072813
5: -4.5824666, -3.3534961, -4.5798883, -3.3532829, -0.6955571, 0.6894414
6: -5.3531604, -3.6075730, -5.3546457, -3.6091208, -0.7882564, 0.7930915
7: -6.2266221, -5.0644441, -6.2220788, -5.0636663, -0.1992241, 0.1879647
8: -0.8278804, -0.2950488, -0.8286238, -0.2983234, -0.2671431, 0.2724946
9: -2.3956170, -1.9252260, -2.3954334, -1.9258840, -0.1482859, 0.1494629

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 489

## Relational analysis of NS_A2_A1_A1_A2_A1

### Relational analysis result of NS_A2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0281257
time: 11.27 seconds

## Relational analysis of NS_A2_A1_A1_A2_A2

### Relational analysis result of NS_A2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0281506
time: 3.75 seconds

## BFS NS instance: NS_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: 1.6282004, 1.8220215, 1.6286306, 1.8221370, -0.0362457, 0.0352378
1: -1.5907719, -0.9900393, -1.5912116, -0.9903073, -0.1291060, 0.1302050
2: -0.1714368, 0.2505011, -0.1715319, 0.2509241, -0.2101207, 0.2093901
3: -3.9559758, -2.7512784, -3.9572737, -2.7511418, -0.5961890, 0.6021824
4: -4.2529163, -3.2468605, -4.2523155, -3.2454503, -0.3088267, 0.3071697
5: -4.5806389, -3.3534338, -4.5825825, -3.3531170, -0.6910367, 0.6969699
6: -5.3541269, -3.6069388, -5.3555355, -3.6090527, -0.7881567, 0.7941680
7: -6.2232289, -5.0642357, -6.2260957, -5.0636725, -0.1909363, 0.1987815
8: -0.8276862, -0.2984072, -0.8288835, -0.2949313, -0.2713370, 0.2683311
9: -2.3955131, -1.9252286, -2.3956308, -1.9258657, -0.1481066, 0.1497277

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 489

## Relational analysis of NS_A2_A1_A2_B2_A1

### Relational analysis result of NS_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281264
time: 14.09 seconds

## Relational analysis of NS_A2_A1_A2_B2_A2

### Relational analysis result of NS_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281508
time: 11.67 seconds

## BFS NS instance: NS_A2_A2_A1_A1

### Backsubstitution after applying NS history:
0: 1.6281734, 1.8224112, 1.6286340, 1.8224164, -0.0365380, 0.0352363
1: -1.5910429, -0.9891992, -1.5909631, -0.9896090, -0.1299152, 0.1296046
2: -0.1716922, 0.2523523, -0.1706398, 0.2524208, -0.2120301, 0.2079536
3: -3.9554980, -2.7519035, -3.9556584, -2.7515645, -0.5978889, 0.5959547
4: -4.2524157, -3.2459967, -4.2516379, -3.2459526, -0.3080189, 0.3062904
5: -4.5804415, -3.3538952, -4.5800333, -3.3535061, -0.6930289, 0.6892741
6: -5.3596935, -3.6070585, -5.3599491, -3.6091208, -0.7879758, 0.8002160
7: -6.2235260, -5.0629716, -6.2220802, -5.0625076, -0.1967486, 0.1875318
8: -0.8309919, -0.2973557, -0.8312312, -0.2983236, -0.2663969, 0.2727880
9: -2.3954382, -1.9250214, -2.3954561, -1.9256995, -0.1485485, 0.1493826

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 489

## Relational analysis of NS_A2_A2_A1_A1_A1

### Relational analysis result of NS_A2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0281953
time: 5.45 seconds

## Relational analysis of NS_A2_A2_A1_A1_A2

### Relational analysis result of NS_A2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0282166
time: 3.62 seconds

## BFS NS instance: NS_A2_A2_A1_A2

### Backsubstitution after applying NS history:
0: 1.6281674, 1.8224442, 1.6286325, 1.8224167, -0.0365418, 0.0352709
1: -1.5912912, -0.9890440, -1.5909631, -0.9894817, -0.1303084, 0.1296655
2: -0.1724634, 0.2527270, -0.1706616, 0.2527411, -0.2130145, 0.2079803
3: -3.9568586, -2.7511959, -3.9556704, -2.7510114, -0.5998001, 0.5962008
4: -4.2531838, -3.2447276, -4.2521472, -3.2459507, -0.3086004, 0.3080920
5: -4.5827546, -3.3526378, -4.5800428, -3.3525729, -0.6962736, 0.6897788
6: -5.3597202, -3.6069906, -5.3599606, -3.6091208, -0.7879944, 0.8003263
7: -6.2272534, -5.0611963, -6.2220817, -5.0609832, -0.2021392, 0.1877021
8: -0.8328416, -0.2939638, -0.8326194, -0.2983229, -0.2672170, 0.2775854
9: -2.3955956, -1.9249573, -2.3954568, -1.9256617, -0.1487674, 0.1494147

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 489

## Relational analysis of NS_A2_A2_A1_A2_B1

### Relational analysis result of NS_A2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282243, upper bound: 0.0282534
time: 5.99 seconds

## Relational analysis of NS_A2_A2_A1_A2_B2

### Relational analysis result of NS_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0282527
time: 3.28 seconds

## BFS NS instance: NS_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: 1.6281292, 1.8224506, 1.6286352, 1.8224516, -0.0366172, 0.0351951
1: -1.5909625, -0.9891116, -1.5909631, -0.9895893, -0.1301523, 0.1296958
2: -0.1719304, 0.2524183, -0.1708013, 0.2523659, -0.2121947, 0.2083979
3: -3.9559855, -2.7515779, -3.9560568, -2.7516565, -0.5961115, 0.6008326
4: -4.2525568, -3.2459500, -4.2516203, -3.2459414, -0.3075464, 0.3074020
5: -4.5809183, -3.3535061, -4.5804238, -3.3536654, -0.6912496, 0.6940613
6: -5.3606768, -3.6063557, -5.3608217, -3.6091208, -0.7877851, 0.8013836
7: -6.2238603, -5.0625124, -6.2223716, -5.0627642, -0.1936809, 0.1931280
8: -0.8312607, -0.2973228, -0.8310285, -0.2983221, -0.2666135, 0.2726011
9: -2.3954906, -1.9249973, -2.3954973, -1.9257079, -0.1485561, 0.1494607

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 489

## Relational analysis of NS_A2_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281950, upper bound: 0.0282527
time: 11.41 seconds

## Relational analysis of NS_A2_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282154, upper bound: 0.0282519
time: 75.00 seconds

## BFS NS instance: NS_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: 1.6281275, 1.8224506, 1.6286293, 1.8224847, -0.0366518, 0.0351990
1: -1.5909625, -0.9889838, -1.5912116, -0.9894341, -0.1302131, 0.1300890
2: -0.1719521, 0.2527390, -0.1715727, 0.2527407, -0.2122214, 0.2093820
3: -3.9559970, -2.7510262, -3.9574184, -2.7509489, -0.5963569, 0.6027430
4: -4.2530656, -3.2459493, -4.2523890, -3.2446737, -0.3093479, 0.3079789
5: -4.5809264, -3.3525734, -4.5827365, -3.3524077, -0.6917524, 0.6973062
6: -5.3606873, -3.6063557, -5.3608494, -3.6090527, -0.7878952, 0.8014022
7: -6.2238603, -5.0609879, -6.2260985, -5.0609894, -0.1938513, 0.1985189
8: -0.8326482, -0.2973225, -0.8328779, -0.2949307, -0.2714111, 0.2734220
9: -2.3954916, -1.9249601, -2.3956549, -1.9256434, -0.1485882, 0.1496798

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 489

## Relational analysis of NS_A2_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282288, upper bound: 0.0282531
time: 25.39 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0282537
time: 6.51 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 38.02 seconds
NS_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0279891
NS_A1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0280088
NS_A1_A1_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282243, upper bound: 0.0280482
NS_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0280470
NS_A1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282287, upper bound: 0.0280472
NS_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0280483
NS_A1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282288, upper bound: 0.0281466
NS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281450
NS_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0280927
NS_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281122
NS_A1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282288, upper bound: 0.0281505
NS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281512
NS_A2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0280926
NS_A2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0281127
NS_A2_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0281257
NS_A2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0281506
NS_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281264
NS_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281508
NS_A2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0281953
NS_A2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0282166
NS_A2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282243, upper bound: 0.0282534
NS_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0282527
NS_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0281950, upper bound: 0.0282527
NS_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282154, upper bound: 0.0282519
NS_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282288, upper bound: 0.0282531
NS_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0282537

## BFS NS instance: NS_A1_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: 1.6287680, 1.8215555, 1.6286483, 1.8216908, -0.0352686, 0.0352541
1: -1.5906076, -0.9908775, -1.5907339, -0.9906002, -0.1287065, 0.1287357
2: -0.1696133, 0.2488055, -0.1705400, 0.2494592, -0.2073722, 0.2074238
3: -3.9546998, -2.7529998, -3.9554687, -2.7524776, -0.5962431, 0.5944734
4: -4.2505498, -3.2483010, -4.2509274, -3.2474740, -0.3054177, 0.3042890
5: -4.5783768, -3.3567019, -4.5798402, -3.3558369, -0.6886299, 0.6873420
6: -5.3504958, -3.6104527, -5.3523250, -3.6091208, -0.7879198, 0.7880285
7: -6.2194695, -5.0700521, -6.2220683, -5.0685763, -0.1868725, 0.1853609
8: -0.8229131, -0.3010274, -0.8246057, -0.2983254, -0.2641484, 0.2630066
9: -2.3947902, -1.9260532, -2.3948495, -1.9259626, -0.1479283, 0.1480559

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3427

## Relational analysis of NS_A1_A1_A1_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0279495
time: 34.68 seconds

## Relational analysis of NS_A1_A1_A1_A1_A1_A2

### Relational analysis result of NS_A1_A1_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0279859
time: 6.88 seconds

## BFS NS instance: NS_A1_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: 1.6287680, 1.8215498, 1.6286503, 1.8216913, -0.0352900, 0.0352319
1: -1.5905893, -0.9908153, -1.5907339, -0.9905480, -0.1287638, 0.1287719
2: -0.1699908, 0.2492923, -0.1705650, 0.2498850, -0.2079682, 0.2073421
3: -3.9553220, -2.7521706, -3.9554842, -2.7517684, -0.5975642, 0.5941678
4: -4.2513542, -3.2477422, -4.2515616, -3.2474718, -0.3059717, 0.3054104
5: -4.5794396, -3.3552403, -4.5798531, -3.3546281, -0.6909282, 0.6869999
6: -5.3505793, -3.6104059, -5.3523936, -3.6091208, -0.7879119, 0.7881641
7: -6.2212362, -5.0677099, -6.2220702, -5.0665140, -0.1906762, 0.1837452
8: -0.8250927, -0.2994404, -0.8264177, -0.2983247, -0.2642279, 0.2664189
9: -2.3948019, -1.9260234, -2.3948507, -1.9259435, -0.1480112, 0.1480398

Time for backsubstitution: 6.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A1_A1_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282080, upper bound: 0.0280059
time: 4.25 seconds

## Relational analysis of NS_A1_A1_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0280048
time: 28.72 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: 1.6287553, 1.8215830, 1.6286440, 1.8216912, -0.0352704, 0.0352875
1: -1.5908376, -0.9906430, -1.5907339, -0.9904184, -0.1290722, 0.1288791
2: -0.1707619, 0.2496682, -0.1705869, 0.2502060, -0.2083644, 0.2078621
3: -3.9566817, -2.7514610, -3.9554958, -2.7512183, -0.5983236, 0.5955651
4: -4.2521205, -3.2464731, -4.2520695, -3.2474713, -0.3066988, 0.3070601
5: -4.5817518, -3.3539782, -4.5798635, -3.3536994, -0.6923494, 0.6893275
6: -5.3506079, -3.6103377, -5.3524046, -3.6091208, -0.7880309, 0.7881732
7: -6.2249641, -5.0659294, -6.2220712, -5.0649929, -0.1920652, 0.1878380
8: -0.8269441, -0.2960488, -0.8278058, -0.2983245, -0.2670684, 0.2691959
9: -2.3949597, -1.9259562, -2.3948514, -1.9259088, -0.1481591, 0.1481425

Time for backsubstitution: 6.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3427

## Relational analysis of NS_A1_A1_A1_A2_B2_A1

### Relational analysis result of NS_A1_A1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0280076
time: 3.52 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_A2

### Relational analysis result of NS_A1_A1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0280431
time: 9.64 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 1.6287150, 1.8215895, 1.6286409, 1.8217590, -0.0353802, 0.0352186
1: -1.5905089, -0.9905825, -1.5909824, -0.9903704, -0.1290077, 0.1293027
2: -0.1702504, 0.2496800, -0.1714974, 0.2502050, -0.2077221, 0.2092693
3: -3.9558208, -2.7512922, -3.9572442, -2.7511559, -0.5948796, 0.6021090
4: -4.2520061, -3.2476954, -4.2523088, -3.2461934, -0.3074529, 0.3069484
5: -4.5799236, -3.3539155, -4.5825567, -3.3535323, -0.6878304, 0.6968560
6: -5.3515749, -3.6097040, -5.3532934, -3.6090527, -0.7879312, 0.7892498
7: -6.2215705, -5.0657215, -6.2260885, -5.0649977, -0.1838638, 0.1986568
8: -0.8267497, -0.2994076, -0.8280647, -0.2949324, -0.2712623, 0.2650316
9: -2.3948553, -1.9259589, -2.3950496, -1.9258907, -0.1479808, 0.1484075

Time for backsubstitution: 6.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3427

## Relational analysis of NS_A1_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0280083
time: 19.77 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0280437
time: 3.77 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: 1.6286426, 1.8219841, 1.6286826, 1.8220667, -0.0357456, 0.0351816
1: -1.5907009, -0.9895425, -1.5910628, -0.9895886, -0.1298411, 0.1291309
2: -0.1706217, 0.2519177, -0.1712824, 0.2520104, -0.2096121, 0.2088581
3: -3.9556329, -2.7510412, -3.9567113, -2.7511945, -0.5947957, 0.5998204
4: -4.2521410, -3.2467928, -4.2522736, -3.2454627, -0.3079900, 0.3070876
5: -4.5800114, -3.3530550, -4.5820465, -3.3530540, -0.6883328, 0.6943903
6: -5.3573275, -3.6091208, -5.3575621, -3.6097560, -0.7875948, 0.7955815
7: -6.2220726, -5.0624752, -6.2255955, -5.0625176, -0.1871139, 0.1955279
8: -0.8316836, -0.2983243, -0.8320223, -0.2949644, -0.2713362, 0.2700936
9: -2.3947940, -1.9256895, -2.3950202, -1.9256836, -0.1484345, 0.1483001

Time for backsubstitution: 6.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3427

## Relational analysis of NS_A1_A2_B1_B2_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281053
time: 3.63 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281411
time: 8.19 seconds

## BFS NS instance: NS_A1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: 1.6286521, 1.8220249, 1.6286470, 1.8220730, -0.0356719, 0.0352552
1: -1.5907192, -0.9897299, -1.5907339, -0.9897100, -0.1298680, 0.1288922
2: -0.1703845, 0.2510542, -0.1707243, 0.2512758, -0.2098713, 0.2076216
3: -3.9553990, -2.7525170, -3.9558229, -2.7522840, -0.5992587, 0.5952847
4: -4.2508073, -3.2473428, -4.2510128, -3.2466865, -0.3065927, 0.3050696
5: -4.5793295, -3.3556118, -4.5801954, -3.3551288, -0.6921425, 0.6878861
6: -5.3581028, -3.6091681, -5.3584490, -3.6091208, -0.7885594, 0.7953370
7: -6.2205954, -5.0666032, -6.2221999, -5.0658932, -0.1926511, 0.1847608
8: -0.8279130, -0.2999107, -0.8286284, -0.2983235, -0.2642521, 0.2680981
9: -2.3948226, -1.9257681, -2.3949137, -1.9257405, -0.1484676, 0.1480340

Time for backsubstitution: 6.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3019

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3427

## Relational analysis of NS_A1_A2_B2_A1_A1_A1

### Relational analysis result of NS_A1_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0280537
time: 9.14 seconds

## Relational analysis of NS_A1_A2_B2_A1_A1_A2

### Relational analysis result of NS_A1_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282477, upper bound: 0.0280882
time: 3.69 seconds

## BFS NS instance: NS_A1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: 1.6286520, 1.8220193, 1.6286486, 1.8220731, -0.0356934, 0.0352331
1: -1.5907009, -0.9896675, -1.5907340, -0.9896572, -0.1299256, 0.1289284
2: -0.1707615, 0.2515413, -0.1707494, 0.2517013, -0.2104672, 0.2075405
3: -3.9560204, -2.7516880, -3.9558382, -2.7515752, -0.6005802, 0.5949798
4: -4.2516131, -3.2467837, -4.2516470, -3.2466860, -0.3071491, 0.3061912
5: -4.5803919, -3.3541501, -4.5802093, -3.3539193, -0.6944404, 0.6875434
6: -5.3581858, -3.6091208, -5.3585186, -3.6091208, -0.7885518, 0.7954726
7: -6.2223630, -5.0642600, -6.2222009, -5.0638309, -0.1964549, 0.1831450
8: -0.8300918, -0.2983235, -0.8304411, -0.2983232, -0.2643316, 0.2715102
9: -2.3948345, -1.9257385, -2.3949149, -1.9257219, -0.1485504, 0.1480178

Time for backsubstitution: 6.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A1_A2_B2_A1_A2_B1

### Relational analysis result of NS_A1_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282125, upper bound: 0.0281084
time: 10.73 seconds

## Relational analysis of NS_A1_A2_B2_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281074
time: 22.81 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 1.6286392, 1.8220521, 1.6286423, 1.8220731, -0.0356738, 0.0352886
1: -1.5909492, -0.9894953, -1.5907340, -0.9895279, -0.1302338, 0.1290357
2: -0.1715325, 0.2519173, -0.1707712, 0.2520222, -0.2108639, 0.2080600
3: -3.9573812, -2.7509780, -3.9558506, -2.7510247, -0.6013389, 0.5963776
4: -4.2523828, -3.2455158, -4.2521544, -3.2466855, -0.3078815, 0.3078413
5: -4.5827045, -3.3528893, -4.5802193, -3.3529885, -0.6958618, 0.6898732
6: -5.3582149, -3.6090527, -5.3585277, -3.6091208, -0.7886705, 0.7954817
7: -6.2260900, -5.0624795, -6.2222023, -5.0623097, -0.1978440, 0.1872379
8: -0.8319427, -0.2949324, -0.8318290, -0.2983230, -0.2671721, 0.2742869
9: -2.3949921, -1.9256713, -2.3949161, -1.9256867, -0.1486986, 0.1481206

Time for backsubstitution: 6.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 489
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3019

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3427

## Relational analysis of NS_A1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281107
time: 15.41 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281456
time: 5.66 seconds

## BFS NS instance: NS_A2_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: 1.6282586, 1.8219875, 1.6286445, 1.8220689, -0.0361208, 0.0352842
1: -1.5908705, -0.9903753, -1.5909631, -0.9905835, -0.1287265, 0.1296137
2: -0.1708000, 0.2496233, -0.1705740, 0.2501747, -0.2092814, 0.2074936
3: -3.9548540, -2.7529981, -3.9554980, -2.7524762, -0.5963879, 0.5945351
4: -4.2514563, -3.2474675, -4.2509289, -3.2467301, -0.3067816, 0.3043492
5: -4.5790901, -3.3562388, -4.5798645, -3.3554423, -0.6899960, 0.6874398
6: -5.3530393, -3.6076875, -5.3545556, -3.6091208, -0.7881372, 0.7928379
7: -6.2211266, -5.0685883, -6.2220759, -5.0672750, -0.1900070, 0.1854643
8: -0.8238478, -0.3000271, -0.8254222, -0.2983239, -0.2642196, 0.2642816
9: -2.3954475, -1.9253446, -2.3954306, -1.9259626, -0.1479765, 0.1493685

Time for backsubstitution: 6.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198
type: B, layer: 1, pos: 3019

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3427

## Relational analysis of NS_A2_A1_A1_A1_A1_A1

### Relational analysis result of NS_A2_A1_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0280534
time: 6.66 seconds

## Relational analysis of NS_A2_A1_A1_A1_A1_A2

### Relational analysis result of NS_A2_A1_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0280871
time: 8.85 seconds

## BFS NS instance: NS_A2_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: 1.6282581, 1.8219819, 1.6286459, 1.8220690, -0.0361423, 0.0352621
1: -1.5908521, -0.9903134, -1.5909631, -0.9905310, -0.1287838, 0.1296497
2: -0.1711772, 0.2501104, -0.1705990, 0.2506006, -0.2098772, 0.2074120
3: -3.9554753, -2.7521687, -3.9555140, -2.7517676, -0.5977097, 0.5942297
4: -4.2522607, -3.2469079, -4.2515635, -3.2467289, -0.3073361, 0.3054705
5: -4.5801530, -3.3547766, -4.5798788, -3.3542347, -0.6922956, 0.6870971
6: -5.3531232, -3.6076407, -5.3546247, -3.6091208, -0.7881293, 0.7929738
7: -6.2228956, -5.0662465, -6.2220774, -5.0652137, -0.1938108, 0.1838486
8: -0.8260278, -0.2984397, -0.8272349, -0.2983235, -0.2642990, 0.2676939
9: -2.3954592, -1.9253150, -2.3954320, -1.9259433, -0.1480595, 0.1493523

Time for backsubstitution: 6.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3427

## Relational analysis of NS_A2_A1_A1_A1_A2_A1

### Relational analysis result of NS_A2_A1_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0280739
time: 3.34 seconds

## Relational analysis of NS_A2_A1_A1_A1_A2_A2

### Relational analysis result of NS_A2_A1_A1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0281085
time: 3.55 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A1

### Backsubstitution after applying NS history:
0: 1.6282583, 1.8220086, 1.6286473, 1.8220690, -0.0361248, 0.0353163
1: -1.5910844, -0.9902416, -1.5909631, -0.9904750, -0.1290732, 0.1296646
2: -0.1715703, 0.2499980, -0.1705957, 0.2504955, -0.2102292, 0.2075434
3: -3.9562154, -2.7522893, -3.9555094, -2.7519248, -0.5982993, 0.5947812
4: -4.2522230, -3.2461989, -4.2514377, -3.2467289, -0.3073578, 0.3061503
5: -4.5814033, -3.3549790, -4.5798745, -3.3545117, -0.6932414, 0.6879454
6: -5.3530674, -3.6076212, -5.3545675, -3.6091208, -0.7881539, 0.7929466
7: -6.2248497, -5.0668163, -6.2220774, -5.0657511, -0.1953458, 0.1856378
8: -0.8257009, -0.2966355, -0.8268102, -0.2983239, -0.2650400, 0.2690789
9: -2.3955960, -1.9252902, -2.3954318, -1.9259322, -0.1481954, 0.1494008

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3427

## Relational analysis of NS_A2_A1_A1_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0280865
time: 3.29 seconds

## Relational analysis of NS_A2_A1_A1_A2_A1_A2

### Relational analysis result of NS_A2_A1_A1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0281206
time: 14.04 seconds

## BFS NS instance: NS_A2_A1_A1_A2_A2

### Backsubstitution after applying NS history:
0: 1.6282464, 1.8220152, 1.6286391, 1.8220693, -0.0361466, 0.0352966
1: -1.5911003, -0.9901468, -1.5909631, -0.9903960, -0.1291771, 0.1297031
2: -0.1719477, 0.2504859, -0.1706209, 0.2509219, -0.2108623, 0.2074987
3: -3.9568365, -2.7514603, -3.9555252, -2.7512155, -0.5996206, 0.5944753
4: -4.2530260, -3.2456393, -4.2520723, -3.2467275, -0.3079125, 0.3072721
5: -4.5824656, -3.3535185, -4.5798879, -3.3533015, -0.6955407, 0.6876011
6: -5.3531499, -3.6075730, -5.3546367, -3.6091208, -0.7881474, 0.7930838
7: -6.2266221, -5.0644684, -6.2220788, -5.0636888, -0.1992026, 0.1840267
8: -0.8278784, -0.2950488, -0.8286226, -0.2983234, -0.2651192, 0.2724911
9: -2.3956170, -1.9252505, -2.3954332, -1.9259058, -0.1482783, 0.1493855

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A2_A1_A1_A2_A2_B1

### Relational analysis result of NS_A2_A1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282081, upper bound: 0.0281453
time: 38.69 seconds

## Relational analysis of NS_A2_A1_A1_A2_A2_B2

### Relational analysis result of NS_A2_A1_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0281450
time: 3.94 seconds

## BFS NS instance: NS_A2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 1.6282178, 1.8220270, 1.6286447, 1.8221371, -0.0362350, 0.0352470
1: -1.5907903, -0.9901812, -1.5912116, -0.9904272, -0.1290169, 0.1300732
2: -0.1710602, 0.2500099, -0.1715066, 0.2504949, -0.2095407, 0.2089021
3: -3.9553533, -2.7521198, -3.9572577, -2.7518616, -0.5948560, 0.6013234
4: -4.2521062, -3.2474203, -4.2516785, -3.2454512, -0.3081110, 0.3060389
5: -4.5795760, -3.3549166, -4.5825682, -3.3543446, -0.6887212, 0.6954721
6: -5.3540325, -3.6069851, -5.3554573, -3.6090527, -0.7880547, 0.7940247
7: -6.2214599, -5.0666075, -6.2260938, -5.0657573, -0.1871249, 0.1964188
8: -0.8255050, -0.2999942, -0.8270706, -0.2949317, -0.2692331, 0.2649156
9: -2.3955007, -1.9252932, -2.3956294, -1.9259139, -0.1480162, 0.1496654

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3427

## Relational analysis of NS_A2_A1_A2_B2_A1_A1

### Relational analysis result of NS_A2_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0280864
time: 20.35 seconds

## Relational analysis of NS_A2_A1_A2_B2_A1_A2

### Relational analysis result of NS_A2_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281225
time: 3.76 seconds

## BFS NS instance: NS_A2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 1.6282063, 1.8220215, 1.6286360, 1.8221370, -0.0362536, 0.0352248
1: -1.5907719, -0.9900858, -1.5912116, -0.9903485, -0.1290818, 0.1300957
2: -0.1714367, 0.2504978, -0.1715318, 0.2509213, -0.2100644, 0.2087505
3: -3.9559746, -2.7512908, -3.9572730, -2.7511530, -0.5961771, 0.6010176
4: -4.2529120, -3.2468610, -4.2523117, -3.2454500, -0.3086660, 0.3071601
5: -4.5806379, -3.3534555, -4.5825820, -3.3531358, -0.6910193, 0.6951288
6: -5.3541174, -3.6069388, -5.3555260, -3.6090527, -0.7880483, 0.7941599
7: -6.2232299, -5.0642614, -6.2260957, -5.0636950, -0.1909128, 0.1947569
8: -0.8276846, -0.2984072, -0.8288823, -0.2949313, -0.2693130, 0.2683278
9: -2.3955126, -1.9252532, -2.3956308, -1.9258877, -0.1480990, 0.1496491

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A2_A1_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282125, upper bound: 0.0281444
time: 14.96 seconds

## Relational analysis of NS_A2_A1_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281455
time: 3.38 seconds

## BFS NS instance: NS_A2_A2_A1_A1_A1

### Backsubstitution after applying NS history:
0: 1.6281857, 1.8224171, 1.6286430, 1.8224163, -0.0365270, 0.0352454
1: -1.5910614, -0.9893194, -1.5909631, -0.9897101, -0.1298336, 0.1294978
2: -0.1713154, 0.2518612, -0.1706147, 0.2519914, -0.2113826, 0.2074853
3: -3.9548767, -2.7527456, -3.9556417, -2.7522840, -0.5965552, 0.5950961
4: -4.2516050, -3.2465558, -4.2509999, -3.2459531, -0.3073015, 0.3051592
5: -4.5793781, -3.3553782, -4.5800185, -3.3547344, -0.6907125, 0.6877755
6: -5.3595991, -3.6071062, -5.3598700, -3.6091208, -0.7878747, 0.8000731
7: -6.2217584, -5.0653419, -6.2220788, -5.0645924, -0.1929219, 0.1852015
8: -0.8288105, -0.2989424, -0.8294179, -0.2983239, -0.2642940, 0.2693725
9: -2.3954260, -1.9250755, -2.3954546, -1.9257404, -0.1484580, 0.1493204

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3427

## Relational analysis of NS_A2_A2_A1_A1_A1_A1

### Relational analysis result of NS_A2_A2_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0281565
time: 3.44 seconds

## Relational analysis of NS_A2_A2_A1_A1_A1_A2

### Relational analysis result of NS_A2_A2_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0281899
time: 13.22 seconds

## BFS NS instance: NS_A2_A2_A1_A1_A2

### Backsubstitution after applying NS history:
0: 1.6281854, 1.8224112, 1.6286447, 1.8224164, -0.0365485, 0.0352233
1: -1.5910429, -0.9892574, -1.5909631, -0.9896571, -0.1298909, 0.1295339
2: -0.1716921, 0.2523482, -0.1706396, 0.2524170, -0.2119784, 0.2074041
3: -3.9554968, -2.7519164, -3.9556580, -2.7515757, -0.5978768, 0.5947909
4: -4.2524104, -3.2459970, -4.2516336, -3.2459521, -0.3078580, 0.3062813
5: -4.5804405, -3.3539166, -4.5800328, -3.3535254, -0.6930113, 0.6874332
6: -5.3596821, -3.6070585, -5.3599391, -3.6091208, -0.7878675, 0.8002076
7: -6.2235270, -5.0629988, -6.2220798, -5.0625300, -0.1967256, 0.1835858
8: -0.8309900, -0.2973557, -0.8312299, -0.2983234, -0.2643732, 0.2727847
9: -2.3954380, -1.9250458, -2.3954556, -1.9257210, -0.1485410, 0.1493043

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 489
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A2_A2_A1_A1_A2_B1

### Relational analysis result of NS_A2_A2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282081, upper bound: 0.0282102
time: 20.99 seconds

## Relational analysis of NS_A2_A2_A1_A1_A2_B2

### Relational analysis result of NS_A2_A2_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0282093
time: 30.43 seconds

## BFS NS instance: NS_A2_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: 1.6281819, 1.8224443, 1.6286502, 1.8224220, -0.0365510, 0.0352601
1: -1.5912912, -0.9891641, -1.5909815, -0.9896237, -0.1301768, 0.1295763
2: -0.1724381, 0.2522977, -0.1702847, 0.2522500, -0.2125264, 0.2073998
3: -3.9568429, -2.7519164, -3.9550481, -2.7518544, -0.5989411, 0.5948677
4: -4.2525463, -3.2447295, -4.2513351, -3.2465096, -0.3074704, 0.3073762
5: -4.5827403, -3.3538661, -4.5789800, -3.3540571, -0.6947765, 0.6874641
6: -5.3596430, -3.6069906, -5.3598671, -3.6091681, -0.7878513, 0.8002238
7: -6.2272515, -5.0632815, -6.2203135, -5.0633545, -0.1997764, 0.1838905
8: -0.8310283, -0.2939645, -0.8304384, -0.2999104, -0.2638017, 0.2754818
9: -2.3955944, -1.9250050, -2.3954451, -1.9257261, -0.1487052, 0.1493244

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A2_A2_A1_A2_B1_B1

### Relational analysis result of NS_A2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281841, upper bound: 0.0282487
time: 3.64 seconds

## Relational analysis of NS_A2_A2_A1_A2_B1_B2

### Relational analysis result of NS_A2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282192, upper bound: 0.0282480
time: 37.66 seconds

## BFS NS instance: NS_A2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: 1.6281729, 1.8224442, 1.6286387, 1.8224167, -0.0365288, 0.0352789
1: -1.5912912, -0.9890848, -1.5909631, -0.9895283, -0.1301993, 0.1296412
2: -0.1724632, 0.2527241, -0.1706614, 0.2527378, -0.2123750, 0.2079239
3: -3.9568586, -2.7512074, -3.9556694, -2.7510242, -0.5986362, 0.5961891
4: -4.2531796, -3.2447281, -4.2521424, -3.2459505, -0.3085910, 0.3079311
5: -4.5827532, -3.3526564, -4.5800424, -3.3525944, -0.6944332, 0.6897622
6: -5.3597116, -3.6069906, -5.3599496, -3.6091208, -0.7879863, 0.8002174
7: -6.2272539, -5.0612183, -6.2220812, -5.0610089, -0.1981146, 0.1876788
8: -0.8328401, -0.2939638, -0.8326174, -0.2983229, -0.2672138, 0.2755617
9: -2.3955960, -1.9249789, -2.3954568, -1.9256865, -0.1486890, 0.1494072

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 3413
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3198
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3427

## Relational analysis of NS_A2_A2_A1_A2_B2_A1

### Relational analysis result of NS_A2_A2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0282140
time: 16.28 seconds

## Relational analysis of NS_A2_A2_A1_A2_B2_A2

### Relational analysis result of NS_A2_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0282482
time: 3.57 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: 1.6281379, 1.8224505, 1.6286477, 1.8224573, -0.0366264, 0.0351841
1: -1.5909625, -0.9892119, -1.5909815, -0.9897091, -0.1300456, 0.1296141
2: -0.1719051, 0.2519891, -0.1704241, 0.2518746, -0.2117267, 0.2077502
3: -3.9559696, -2.7522981, -3.9554353, -2.7524991, -0.5952535, 0.5994993
4: -4.2519188, -3.2459512, -4.2508097, -3.2465010, -0.3064165, 0.3066866
5: -4.5809031, -3.3547339, -4.5793610, -3.3551483, -0.6897523, 0.6917462
6: -5.3605986, -3.6063557, -5.3607283, -3.6091681, -0.7876413, 0.8012831
7: -6.2238584, -5.0645976, -6.2206035, -5.0651345, -0.1913508, 0.1893015
8: -0.8294463, -0.2973236, -0.8288473, -0.2999092, -0.2631981, 0.2704976
9: -2.3954895, -1.9250381, -2.3954854, -1.9257623, -0.1484939, 0.1493702

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A2_A2_A2_B1_B1_B1

### Relational analysis result of NS_A2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281549, upper bound: 0.0282492
time: 3.73 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_B2

### Relational analysis result of NS_A2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281900, upper bound: 0.0282476
time: 30.02 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 1.6281396, 1.8224506, 1.6286473, 1.8224516, -0.0366042, 0.0352056
1: -1.5909625, -0.9891593, -1.5909631, -0.9896469, -0.1300816, 0.1296715
2: -0.1719303, 0.2524146, -0.1708012, 0.2523617, -0.2116455, 0.2083461
3: -3.9559858, -2.7515895, -3.9560566, -2.7516699, -0.5949473, 0.6008201
4: -4.2525530, -3.2459500, -4.2516155, -3.2459426, -0.3075372, 0.3072412
5: -4.5809169, -3.3535247, -4.5804234, -3.3536868, -0.6894088, 0.6940444
6: -5.3606682, -3.6063557, -5.3608108, -3.6091208, -0.7877765, 0.8012748
7: -6.2238603, -5.0625353, -6.2223716, -5.0627923, -0.1897351, 0.1931052
8: -0.8312589, -0.2973228, -0.8310261, -0.2983221, -0.2666101, 0.2705773
9: -2.3954904, -1.9250189, -2.3954971, -1.9257326, -0.1484777, 0.1494532

Time for backsubstitution: 5.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A2_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281754, upper bound: 0.0282478
time: 17.48 seconds

## Relational analysis of NS_A2_A2_A2_B1_B2_B2

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282103, upper bound: 0.0282488
time: 11.28 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: 1.6281413, 1.8224506, 1.6286477, 1.8224782, -0.0366584, 0.0351880
1: -1.5909625, -0.9891031, -1.5911953, -0.9895759, -0.1300963, 0.1299608
2: -0.1719269, 0.2523097, -0.1711945, 0.2522495, -0.2117764, 0.2086981
3: -3.9559808, -2.7517467, -3.9567971, -2.7517910, -0.5954990, 0.6014103
4: -4.2524271, -3.2459507, -4.2515802, -3.2452331, -0.3082172, 0.3072630
5: -4.5809121, -3.3538017, -4.5816736, -3.3538890, -0.6902566, 0.6949904
6: -5.3606091, -3.6063557, -5.3607554, -3.6091003, -0.7877493, 0.8012998
7: -6.2238593, -5.0630722, -6.2243261, -5.0633616, -0.1915242, 0.1946404
8: -0.8308344, -0.2973228, -0.8306987, -0.2965180, -0.2679955, 0.2713186
9: -2.3954904, -1.9250078, -2.3956337, -1.9257078, -0.1485260, 0.1495890

Time for backsubstitution: 5.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A2_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281886, upper bound: 0.0282486
time: 21.10 seconds

## Relational analysis of NS_A2_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282237, upper bound: 0.0282484
time: 19.63 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: 1.6281326, 1.8224506, 1.6286354, 1.8224847, -0.0366387, 0.0352100
1: -1.5909625, -0.9890245, -1.5912116, -0.9894806, -0.1301348, 0.1300648
2: -0.1719518, 0.2527358, -0.1715721, 0.2527374, -0.2117321, 0.2093311
3: -3.9559965, -2.7510374, -3.9574180, -2.7509625, -0.5951931, 0.6027315
4: -4.2530613, -3.2459488, -4.2523842, -3.2446737, -0.3093385, 0.3078176
5: -4.5809264, -3.3525922, -4.5827360, -3.3524289, -0.6899123, 0.6972891
6: -5.3606791, -3.6063557, -5.3608384, -3.6090527, -0.7878866, 0.8012936
7: -6.2238603, -5.0610104, -6.2260985, -5.0610142, -0.1899135, 0.1984976
8: -0.8326472, -0.2973225, -0.8328767, -0.2949307, -0.2714076, 0.2713982
9: -2.3954916, -1.9249818, -2.3956547, -1.9256679, -0.1485106, 0.1496720

Time for backsubstitution: 5.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3427
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3456
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3427

## Relational analysis of NS_A2_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0282134
time: 3.84 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0282477
time: 9.23 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 18.66 seconds
NS_A1_A1_A1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0279495
NS_A1_A1_A1_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0279859
NS_A1_A1_A1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282080, upper bound: 0.0280059
NS_A1_A1_A1_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0280048
NS_A1_A1_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0280076
NS_A1_A1_A1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0280431
NS_A1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0280083
NS_A1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0280437
NS_A1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281053
NS_A1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281411
NS_A1_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0280537
NS_A1_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282477, upper bound: 0.0280882
NS_A1_A2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282125, upper bound: 0.0281084
NS_A1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281074
NS_A1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281107
NS_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281456
NS_A2_A1_A1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0280534
NS_A2_A1_A1_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0280871
NS_A2_A1_A1_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0280739
NS_A2_A1_A1_A1_A2_A2, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0281085
NS_A2_A1_A1_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0280865
NS_A2_A1_A1_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0281206
NS_A2_A1_A1_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282081, upper bound: 0.0281453
NS_A2_A1_A1_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0281450
NS_A2_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0280864
NS_A2_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281225
NS_A2_A1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282125, upper bound: 0.0281444
NS_A2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281455
NS_A2_A2_A1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0281565
NS_A2_A2_A1_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0281899
NS_A2_A2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282081, upper bound: 0.0282102
NS_A2_A2_A1_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0282093
NS_A2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0281841, upper bound: 0.0282487
NS_A2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282192, upper bound: 0.0282480
NS_A2_A2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0282140
NS_A2_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0282482
NS_A2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0281549, upper bound: 0.0282492
NS_A2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0281900, upper bound: 0.0282476
NS_A2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0281754, upper bound: 0.0282478
NS_A2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282103, upper bound: 0.0282488
NS_A2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0281886, upper bound: 0.0282486
NS_A2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282237, upper bound: 0.0282484
NS_A2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0282134
NS_A2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 18.66
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0282477

## BFS NS instance: NS_A1_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: 1.6287357, 1.8216063, 1.6286589, 1.8217589, -0.0353517, 0.0352123
1: -1.5904636, -0.9909671, -1.5909822, -0.9906905, -0.1286401, 0.1289135
2: -0.1700963, 0.2486353, -0.1714504, 0.2493502, -0.2067581, 0.2081938
3: -3.9553945, -2.7530572, -3.9572139, -2.7525887, -0.5930338, 0.6003137
4: -4.2503533, -3.2480919, -4.2510047, -3.2461970, -0.3058059, 0.3052214
5: -4.5792055, -3.3569529, -4.5825291, -3.3559735, -0.6846550, 0.6937840
6: -5.3513794, -3.6097183, -5.3531318, -3.6090527, -0.7877271, 0.7890648
7: -6.2211094, -5.0701981, -6.2260852, -5.0686574, -0.1786618, 0.1936172
8: -0.8223319, -0.3004570, -0.8245131, -0.2949330, -0.2668525, 0.2603791
9: -2.3948421, -1.9261171, -2.3950467, -1.9260213, -0.1477824, 0.1482238

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2540
type: B, layer: 1, pos: 2540
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2297
type: A, layer: 1, pos: 2297
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3277
type: B, layer: 1, pos: 3277
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3231
type: A, layer: 1, pos: 3231
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2050
type: A, layer: 1, pos: 2050
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 3413
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: B, layer: 1, pos: 3199
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3200
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2552
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2552
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 3198
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2540

## Relational analysis of NS_A1_A1_A2_B2_B2_A1_A1

### Relational analysis result of NS_A1_A1_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282283, upper bound: 0.0280002
time: 113.33 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2_A1_A2

### Relational analysis result of NS_A1_A1_A2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282341, upper bound: 0.0279949
time: 16.10 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 36.91 + 1860.06 = 1896.97 seconds
