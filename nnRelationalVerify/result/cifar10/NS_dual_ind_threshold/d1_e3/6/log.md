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
execution time: IAR + RelationalAnalysis = 7.81 + 28.53 = 36.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0282724, upper bound: 0.0282727

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 546

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282713, upper bound: 0.0281695
time: 11.60 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282713, upper bound: 0.0282723
time: 25.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 37.22 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 37.22
Output dim: 2, lower bound: -0.0282713, upper bound: 0.0281695
NS_A2, status: Status.UNKNOWN, split count: 1, time: 37.22
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

Time for backsubstitution: 5.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3456

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281679, upper bound: 0.0281690
time: 29.19 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282707, upper bound: 0.0281693
time: 10.62 seconds

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

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3456
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3456

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281678, upper bound: 0.0281684
time: 37.85 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282707, upper bound: 0.0282716
time: 3.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 47.47 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 47.47
Output dim: 2, lower bound: -0.0281679, upper bound: 0.0281690
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 47.47
Output dim: 2, lower bound: -0.0282707, upper bound: 0.0281693
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 47.47
Output dim: 2, lower bound: -0.0281678, upper bound: 0.0281684
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 47.47
Output dim: 2, lower bound: -0.0282707, upper bound: 0.0282716

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 1.6286297, 1.8220241, 1.6286292, 1.8220776, -0.0353184, 0.0357318
1: -1.5907009, -0.9892251, -1.5907340, -0.9892237, -0.1292345, 0.1304403
2: -0.1710131, 0.2519267, -0.1710174, 0.2520299, -0.2089257, 0.2109924
3: -3.9573815, -2.7510030, -3.9573846, -2.7510195, -0.6014791, 0.6013050
4: -4.2525039, -3.2467763, -4.2525039, -3.2466903, -0.3080671, 0.3078740
5: -4.5817490, -3.3530240, -4.5817533, -3.3529553, -0.6950974, 0.6955163
6: -5.3587999, -3.6091208, -5.3591065, -3.6091208, -0.7968242, 0.7895168
7: -6.2234945, -5.0624070, -6.2234950, -5.0622416, -0.1948635, 0.1979116
8: -0.8317330, -0.2983228, -0.8318497, -0.2983227, -0.2729038, 0.2681091
9: -2.3948426, -1.9256477, -2.3949239, -1.9256541, -0.1481493, 0.1486828

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3019

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282515, upper bound: 0.0281539
time: 7.21 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0281546
time: 3.60 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 1.6281202, 1.8224562, 1.6286249, 1.8224556, -0.0361700, 0.0357618
1: -1.5909625, -0.9887237, -1.5909631, -0.9892071, -0.1292544, 0.1313177
2: -0.1722000, 0.2527446, -0.1710518, 0.2527456, -0.2108327, 0.2110623
3: -3.9575348, -2.7510014, -3.9574146, -2.7510190, -0.6016240, 0.6013668
4: -4.2534103, -3.2459421, -4.2525058, -3.2459464, -0.3094281, 0.3079344
5: -4.5824633, -3.3525605, -4.5817771, -3.3525615, -0.6964638, 0.6956129
6: -5.3613420, -3.6063557, -5.3613386, -3.6091208, -0.7970417, 0.7943265
7: -6.2251530, -5.0609431, -6.2235026, -5.0609412, -0.1979977, 0.1980144
8: -0.8326679, -0.2973220, -0.8326669, -0.2983214, -0.2729750, 0.2693840
9: -2.3954995, -1.9249399, -2.3955052, -1.9256536, -0.1481978, 0.1499930

Time for backsubstitution: 5.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3019

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282515, upper bound: 0.0282555
time: 21.03 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0282550
time: 16.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 43.68 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 43.68
Output dim: 2, lower bound: -0.0282515, upper bound: 0.0281539
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 43.68
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0281546
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 43.68
Output dim: 2, lower bound: -0.0282515, upper bound: 0.0282555
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 43.68
Output dim: 2, lower bound: -0.0282560, upper bound: 0.0282550

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 1.6286732, 1.8219796, 1.6286299, 1.8220383, -0.0352352, 0.0356856
1: -1.5907811, -0.9895539, -1.5907340, -0.9894980, -0.1288648, 0.1300576
2: -0.1705326, 0.2519152, -0.1706273, 0.2520296, -0.2083753, 0.2104504
3: -3.9553618, -2.7512345, -3.9556425, -2.7510209, -0.5984962, 0.5963862
4: -4.2520933, -3.2468228, -4.2521458, -3.2467008, -0.3073648, 0.3066413
5: -4.5797434, -3.3532538, -4.5800195, -3.3529558, -0.6921599, 0.6907494
6: -5.3572440, -3.6098237, -5.3577199, -3.6091208, -0.7954257, 0.7878451
7: -6.2218690, -5.0626092, -6.2220736, -5.0622420, -0.1919523, 0.1925337
8: -0.8316799, -0.2983553, -0.8318033, -0.2983242, -0.2728548, 0.2680390
9: -2.3947823, -1.9256637, -2.3948755, -1.9256538, -0.1480870, 0.1486102

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3413

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282134, upper bound: 0.0281529
time: 3.43 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0281527
time: 31.75 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 1.6286303, 1.8220189, 1.6286299, 1.8220735, -0.0353122, 0.0356467
1: -1.5907009, -0.9894817, -1.5907340, -0.9894617, -0.1291256, 0.1301251
2: -0.1707665, 0.2519267, -0.1707926, 0.2520301, -0.2085890, 0.2108457
3: -3.9558482, -2.7510033, -3.9560428, -2.7510200, -0.5968101, 0.6011719
4: -4.2521596, -3.2467768, -4.2522035, -3.2466915, -0.3068053, 0.3078409
5: -4.5802169, -3.3530231, -4.5804124, -3.3529558, -0.6905382, 0.6953802
6: -5.3582220, -3.6091208, -5.3585992, -3.6091208, -0.7952271, 0.7890203
7: -6.2222013, -5.0624070, -6.2223649, -5.0622416, -0.1891480, 0.1978667
8: -0.8317146, -0.2983232, -0.8318338, -0.2983230, -0.2728339, 0.2680892
9: -2.3948350, -1.9256479, -2.3949170, -1.9256537, -0.1481044, 0.1486784

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3413

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282179, upper bound: 0.0281521
time: 21.16 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0281531
time: 21.71 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 1.6281636, 1.8224117, 1.6286259, 1.8224165, -0.0360869, 0.0357156
1: -1.5910429, -0.9890528, -1.5909631, -0.9894817, -0.1288847, 0.1309352
2: -0.1717189, 0.2527333, -0.1706614, 0.2527454, -0.2102823, 0.2105204
3: -3.9555161, -2.7512331, -3.9556713, -2.7510200, -0.5986414, 0.5964473
4: -4.2530003, -3.2459893, -4.2521472, -3.2459574, -0.3087262, 0.3067017
5: -4.5804572, -3.3527908, -4.5800447, -3.3525617, -0.6935258, 0.6908464
6: -5.3597884, -3.6070585, -5.3599501, -3.6091208, -0.7956440, 0.7926552
7: -6.2235274, -5.0611453, -6.2220812, -5.0609412, -0.1950869, 0.1926366
8: -0.8326147, -0.2973551, -0.8326207, -0.2983229, -0.2729257, 0.2693140
9: -2.3954396, -1.9249556, -2.3954568, -1.9256538, -0.1481354, 0.1499203

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3413

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282134, upper bound: 0.0282548
time: 17.82 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0282560
time: 5.38 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 1.6281207, 1.8224509, 1.6286256, 1.8224512, -0.0361638, 0.0356767
1: -1.5909625, -0.9889804, -1.5909631, -0.9894446, -0.1291455, 0.1310027
2: -0.1719533, 0.2527445, -0.1708267, 0.2527457, -0.2104961, 0.2109156
3: -3.9560015, -2.7510018, -3.9560719, -2.7510185, -0.5969548, 0.6012343
4: -4.2530665, -3.2459431, -4.2522049, -3.2459474, -0.3081663, 0.3079010
5: -4.5809312, -3.3525603, -4.5804367, -3.3525617, -0.6919038, 0.6954770
6: -5.3607635, -3.6063557, -5.3608322, -3.6091208, -0.7954450, 0.7938303
7: -6.2238598, -5.0609431, -6.2223730, -5.0609407, -0.1922826, 0.1979696
8: -0.8326499, -0.2973225, -0.8326514, -0.2983220, -0.2729050, 0.2693640
9: -2.3954916, -1.9249402, -2.3954983, -1.9256539, -0.1481529, 0.1499885

Time for backsubstitution: 5.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3413

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282179, upper bound: 0.0282561
time: 28.72 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0282547
time: 11.13 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 45.82 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 45.82
Output dim: 2, lower bound: -0.0282134, upper bound: 0.0281529
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 45.82
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0281527
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 45.82
Output dim: 2, lower bound: -0.0282179, upper bound: 0.0281521
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 45.82
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0281531
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 45.82
Output dim: 2, lower bound: -0.0282134, upper bound: 0.0282548
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 45.82
Output dim: 2, lower bound: -0.0282503, upper bound: 0.0282560
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 45.82
Output dim: 2, lower bound: -0.0282179, upper bound: 0.0282561
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 45.82
Output dim: 2, lower bound: -0.0282548, upper bound: 0.0282547

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 1.6286798, 1.8219796, 1.6286337, 1.8220711, -0.0352557, 0.0356731
1: -1.5907811, -0.9895555, -1.5909824, -0.9894882, -0.1287611, 0.1303101
2: -0.1705324, 0.2519102, -0.1713726, 0.2520241, -0.2079621, 0.2110440
3: -3.9553595, -2.7512426, -3.9569864, -2.7509680, -0.5980735, 0.5977197
4: -4.2520928, -3.2468226, -4.2523293, -3.2454333, -0.3086423, 0.3066109
5: -4.5797415, -3.3532658, -4.5823183, -3.3528025, -0.6915476, 0.6930342
6: -5.3572116, -3.6098237, -5.3576946, -3.6090527, -0.7954865, 0.7878067
7: -6.2218685, -5.0626526, -6.2257996, -5.0622916, -0.1902666, 0.1963317
8: -0.8316785, -0.2983553, -0.8320305, -0.2949334, -0.2762434, 0.2672132
9: -2.3947823, -1.9256774, -2.3950322, -1.9256494, -0.1480499, 0.1487698

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 489

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0281263
time: 3.38 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0281494
time: 3.24 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 1.6286368, 1.8220189, 1.6286337, 1.8221060, -0.0353328, 0.0356342
1: -1.5907009, -0.9894833, -1.5909824, -0.9894516, -0.1290219, 0.1303776
2: -0.1707663, 0.2519217, -0.1715373, 0.2520244, -0.2081758, 0.2114390
3: -3.9558449, -2.7510109, -3.9573877, -2.7509665, -0.5963874, 0.6025052
4: -4.2521596, -3.2467768, -4.2523870, -3.2454238, -0.3080823, 0.3078108
5: -4.5802145, -3.3530354, -4.5827098, -3.3528016, -0.6899257, 0.6976645
6: -5.3581877, -3.6091208, -5.3585749, -3.6090527, -0.7952878, 0.7889813
7: -6.2222013, -5.0624509, -6.2260904, -5.0622911, -0.1874624, 0.2016648
8: -0.8317136, -0.2983232, -0.8320613, -0.2949319, -0.2762228, 0.2672637
9: -2.3948350, -1.9256619, -2.3950734, -1.9256495, -0.1480672, 0.1488380

Time for backsubstitution: 5.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 489

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281260
time: 69.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281508
time: 22.18 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 1.6281718, 1.8224118, 1.6286358, 1.8224162, -0.0360727, 0.0356991
1: -1.5910429, -0.9891821, -1.5909631, -0.9896273, -0.1287202, 0.1307944
2: -0.1716966, 0.2524076, -0.1706353, 0.2523652, -0.2098427, 0.2101296
3: -3.9555018, -2.7517924, -3.9556549, -2.7516747, -0.5979722, 0.5958700
4: -4.2524910, -3.2459893, -4.2515621, -3.2459583, -0.3082018, 0.3060905
5: -4.5804448, -3.3537359, -4.5800304, -3.3536654, -0.6924083, 0.6898860
6: -5.3597436, -3.6070585, -5.3598976, -3.6091208, -0.7955945, 0.7925984
7: -6.2235265, -5.0627136, -6.2220798, -5.0627656, -0.1932308, 0.1910440
8: -0.8312259, -0.2973551, -0.8309975, -0.2983232, -0.2715173, 0.2676684
9: -2.3954384, -1.9250071, -2.3954554, -1.9257138, -0.1480663, 0.1498610

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 489

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282109, upper bound: 0.0282297
time: 3.09 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282109, upper bound: 0.0282544
time: 3.27 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 1.6281703, 1.8224117, 1.6286300, 1.8224490, -0.0361074, 0.0357031
1: -1.5910429, -0.9890544, -1.5912116, -0.9894718, -0.1287810, 0.1311877
2: -0.1717186, 0.2527283, -0.1714067, 0.2527398, -0.2098693, 0.2111139
3: -3.9555132, -2.7512407, -3.9570160, -2.7509677, -0.5982180, 0.5977808
4: -4.2530003, -3.2459893, -4.2523317, -3.2446899, -0.3100035, 0.3066713
5: -4.5804548, -3.3528030, -4.5823431, -3.3524084, -0.6929133, 0.6931307
6: -5.3597541, -3.6070585, -5.3599253, -3.6090527, -0.7957048, 0.7926171
7: -6.2235274, -5.0611887, -6.2258072, -5.0609913, -0.1934012, 0.1964348
8: -0.8326137, -0.2973551, -0.8328477, -0.2949325, -0.2763143, 0.2684889
9: -2.3954396, -1.9249694, -2.3956134, -1.9256493, -0.1480982, 0.1500798

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 489

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0282283
time: 75.13 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0282541
time: 3.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 1.6281290, 1.8224509, 1.6286352, 1.8224514, -0.0361497, 0.0356602
1: -1.5909625, -0.9891104, -1.5909631, -0.9895903, -0.1289810, 0.1308619
2: -0.1719308, 0.2524191, -0.1708009, 0.2523652, -0.2100562, 0.2105245
3: -3.9559875, -2.7515619, -3.9560559, -2.7516737, -0.5962861, 0.6006558
4: -4.2525568, -3.2459435, -4.2516203, -3.2459483, -0.3076419, 0.3072898
5: -4.5809193, -3.3535061, -4.5804229, -3.3536656, -0.6907864, 0.6945164
6: -5.3607178, -3.6063557, -5.3607793, -3.6091208, -0.7953954, 0.7937732
7: -6.2238603, -5.0625114, -6.2223716, -5.0627656, -0.1904265, 0.1963768
8: -0.8312609, -0.2973230, -0.8310282, -0.2983221, -0.2714965, 0.2677186
9: -2.3954906, -1.9249917, -2.3954973, -1.9257137, -0.1480835, 0.1499292

Time for backsubstitution: 5.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 489

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282153, upper bound: 0.0282300
time: 18.56 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282154, upper bound: 0.0282515
time: 102.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 1.6281275, 1.8224509, 1.6286296, 1.8224841, -0.0361843, 0.0356642
1: -1.5909625, -0.9889821, -1.5912116, -0.9894352, -0.1290418, 0.1312551
2: -0.1719529, 0.2527396, -0.1715719, 0.2527400, -0.2100829, 0.2115090
3: -3.9559982, -2.7510095, -3.9574165, -2.7509656, -0.5965321, 0.6025674
4: -4.2530656, -3.2459435, -4.2523890, -3.2446799, -0.3094438, 0.3078710
5: -4.5809278, -3.3525734, -4.5827351, -3.3524084, -0.6912916, 0.6977612
6: -5.3607297, -3.6063557, -5.3608055, -3.6090527, -0.7955062, 0.7937917
7: -6.2238603, -5.0609870, -6.2260985, -5.0609903, -0.1905969, 0.2017679
8: -0.8326483, -0.2973225, -0.8328776, -0.2949309, -0.2762939, 0.2685392
9: -2.3954916, -1.9249543, -2.3956552, -1.9256493, -0.1481155, 0.1501482

Time for backsubstitution: 5.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 489

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0282294
time: 5.99 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282526, upper bound: 0.0282532
time: 3.53 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 15.49 seconds
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.49
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0281263
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.49
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0281494
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.49
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281260
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.49
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0281508
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.49
Output dim: 2, lower bound: -0.0282109, upper bound: 0.0282297
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.49
Output dim: 2, lower bound: -0.0282109, upper bound: 0.0282544
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.49
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0282283
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.49
Output dim: 2, lower bound: -0.0282482, upper bound: 0.0282541
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.49
Output dim: 2, lower bound: -0.0282153, upper bound: 0.0282300
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.49
Output dim: 2, lower bound: -0.0282154, upper bound: 0.0282515
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.49
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0282294
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.49
Output dim: 2, lower bound: -0.0282526, upper bound: 0.0282532

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 1.6286972, 1.8219855, 1.6286483, 1.8220710, -0.0352449, 0.0356824
1: -1.5907993, -0.9896973, -1.5909824, -0.9896078, -0.1286720, 0.1301783
2: -0.1701549, 0.2514190, -0.1713474, 0.2515947, -0.2073820, 0.2105557
3: -3.9547379, -2.7520852, -3.9569702, -2.7516885, -0.5967400, 0.5968608
4: -4.2512817, -3.2473819, -4.2516913, -3.2454352, -0.3079270, 0.3054804
5: -4.5786781, -3.3547487, -4.5823040, -3.3540299, -0.6892323, 0.6915367
6: -5.3571177, -3.6098709, -5.3576150, -3.6090527, -0.7953844, 0.7876630
7: -6.2200994, -5.0650244, -6.2257986, -5.0643768, -0.1864551, 0.1939691
8: -0.8294977, -0.2999427, -0.8302171, -0.2949340, -0.2741404, 0.2637978
9: -2.3947706, -1.9257420, -2.3950305, -1.9256974, -0.1479594, 0.1487077

Time for backsubstitution: 5.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282080, upper bound: 0.0281224
time: 3.43 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0281215
time: 10.02 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 1.6286858, 1.8219796, 1.6286392, 1.8220711, -0.0352636, 0.0356602
1: -1.5907811, -0.9896024, -1.5909824, -0.9895288, -0.1287368, 0.1302010
2: -0.1705322, 0.2519068, -0.1713723, 0.2520210, -0.2079058, 0.2104044
3: -3.9553583, -2.7512558, -3.9569869, -2.7509794, -0.5980616, 0.5965556
4: -4.2520881, -3.2468224, -4.2523251, -3.2454333, -0.3084816, 0.3066013
5: -4.5797405, -3.3532877, -4.5823169, -3.3528209, -0.6915307, 0.6911933
6: -5.3572021, -3.6098237, -5.3576856, -3.6090527, -0.7953777, 0.7877986
7: -6.2218685, -5.0626788, -6.2257996, -5.0623140, -0.1902431, 0.1923072
8: -0.8316767, -0.2983553, -0.8320290, -0.2949334, -0.2742198, 0.2672100
9: -2.3947825, -1.9257021, -2.3950319, -1.9256712, -0.1480423, 0.1486914

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282080, upper bound: 0.0281446
time: 3.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0281456
time: 48.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 1.6286544, 1.8220248, 1.6286480, 1.8221061, -0.0353219, 0.0356434
1: -1.5907192, -0.9896256, -1.5909824, -0.9895715, -0.1289327, 0.1302459
2: -0.1703893, 0.2514303, -0.1715123, 0.2515948, -0.2075957, 0.2109510
3: -3.9552236, -2.7518544, -3.9573712, -2.7516868, -0.5950544, 0.6016469
4: -4.2513480, -3.2473364, -4.2517490, -3.2454243, -0.3073674, 0.3066802
5: -4.5791507, -3.3545187, -4.5826964, -3.3540301, -0.6876106, 0.6961673
6: -5.3580933, -3.6091681, -5.3584976, -3.6090527, -0.7951853, 0.7888386
7: -6.2204328, -5.0648222, -6.2260895, -5.0643768, -0.1836509, 0.1993020
8: -0.8295325, -0.2999103, -0.8302478, -0.2949320, -0.2741196, 0.2638481
9: -2.3948228, -1.9257265, -2.3950720, -1.9256974, -0.1479765, 0.1487759

Time for backsubstitution: 5.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282125, upper bound: 0.0281219
time: 3.20 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281224
time: 3.06 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 1.6286430, 1.8220189, 1.6286390, 1.8221060, -0.0353407, 0.0356212
1: -1.5907009, -0.9895303, -1.5909824, -0.9894928, -0.1289977, 0.1302684
2: -0.1707663, 0.2519182, -0.1715372, 0.2520213, -0.2081195, 0.2107999
3: -3.9558446, -2.7510242, -3.9573870, -2.7509775, -0.5963755, 0.6013415
4: -4.2521548, -3.2467775, -4.2523832, -3.2454233, -0.3079218, 0.3078014
5: -4.5802140, -3.3530574, -4.5827098, -3.3528209, -0.6899087, 0.6958237
6: -5.3581781, -3.6091208, -5.3585653, -3.6090527, -0.7951784, 0.7889737
7: -6.2222013, -5.0624771, -6.2260914, -5.0623136, -0.1874390, 0.1976402
8: -0.8317119, -0.2983232, -0.8320594, -0.2949319, -0.2741987, 0.2672603
9: -2.3948348, -1.9256864, -2.3950732, -1.9256713, -0.1480595, 0.1487595

Time for backsubstitution: 5.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282125, upper bound: 0.0281462
time: 3.10 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282477, upper bound: 0.0281447
time: 3.39 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 1.6281840, 1.8224117, 1.6286461, 1.8224162, -0.0360833, 0.0356862
1: -1.5910429, -0.9892359, -1.5909631, -0.9896791, -0.1286958, 0.1307211
2: -0.1716965, 0.2524035, -0.1706352, 0.2523612, -0.2097894, 0.2095631
3: -3.9555013, -2.7518058, -3.9556534, -2.7516863, -0.5979602, 0.5947057
4: -4.2524867, -3.2459898, -4.2515578, -3.2459590, -0.3080410, 0.3060808
5: -4.5804439, -3.3537579, -4.5800295, -3.3536844, -0.6923919, 0.6880453
6: -5.3597326, -3.6070585, -5.3598881, -3.6091208, -0.7954855, 0.7925901
7: -6.2235265, -5.0627389, -6.2220802, -5.0627899, -0.1932074, 0.1870847
8: -0.8312238, -0.2973551, -0.8309957, -0.2983234, -0.2694932, 0.2676649
9: -2.3954384, -1.9250312, -2.3954556, -1.9257356, -0.1480585, 0.1497826

Time for backsubstitution: 5.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281708, upper bound: 0.0282490
time: 7.82 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282058, upper bound: 0.0281457
time: 37.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 1.6281880, 1.8224173, 1.6286441, 1.8224491, -0.0360966, 0.0357123
1: -1.5910614, -0.9891961, -1.5912116, -0.9895914, -0.1286920, 0.1310559
2: -0.1713415, 0.2522370, -0.1713819, 0.2523106, -0.2092891, 0.2106257
3: -3.9548907, -2.7520831, -3.9569995, -2.7516873, -0.5968847, 0.5969228
4: -4.2521877, -3.2465484, -4.2516937, -3.2446907, -0.3092878, 0.3055407
5: -4.5793910, -3.3542857, -4.5823288, -3.3536358, -0.6905973, 0.6916330
6: -5.3596606, -3.6071062, -5.3598471, -3.6090527, -0.7956023, 0.7924733
7: -6.2217593, -5.0635605, -6.2258062, -5.0630760, -0.1895897, 0.1940720
8: -0.8304324, -0.2989424, -0.8310335, -0.2949322, -0.2742112, 0.2650734
9: -2.3954275, -1.9250339, -2.3956118, -1.9256973, -0.1480077, 0.1500178

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282080, upper bound: 0.0282242
time: 3.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0282247
time: 3.95 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 1.6281765, 1.8224117, 1.6286352, 1.8224490, -0.0361153, 0.0356901
1: -1.5910429, -0.9891014, -1.5912116, -0.9895122, -0.1287568, 0.1310784
2: -0.1717184, 0.2527249, -0.1714067, 0.2527368, -0.2098129, 0.2104744
3: -3.9555125, -2.7512536, -3.9570146, -2.7509785, -0.5982063, 0.5966173
4: -4.2529941, -3.2459893, -4.2523274, -3.2446904, -0.3098429, 0.3066617
5: -4.5804539, -3.3528247, -4.5823421, -3.3524270, -0.6928961, 0.6912897
6: -5.3597436, -3.6070585, -5.3599167, -3.6090527, -0.7955956, 0.7926085
7: -6.2235274, -5.0612149, -6.2258067, -5.0610123, -0.1933777, 0.1924102
8: -0.8326124, -0.2973551, -0.8328460, -0.2949325, -0.2742908, 0.2684855
9: -2.3954391, -1.9249940, -2.3956132, -1.9256712, -0.1480906, 0.1500016

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282080, upper bound: 0.0282483
time: 7.03 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0282474
time: 11.84 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 1.6281410, 1.8224506, 1.6286458, 1.8224514, -0.0361602, 0.0356472
1: -1.5909625, -0.9891641, -1.5909631, -0.9896425, -0.1289567, 0.1307888
2: -0.1719308, 0.2524148, -0.1708007, 0.2523615, -0.2100032, 0.2099586
3: -3.9559860, -2.7515745, -3.9560556, -2.7516847, -0.5962741, 0.5994918
4: -4.2525520, -3.2459438, -4.2516155, -3.2459483, -0.3074813, 0.3072802
5: -4.5809178, -3.3535273, -4.5804219, -3.3536844, -0.6907690, 0.6926761
6: -5.3607087, -3.6063557, -5.3607702, -3.6091208, -0.7952864, 0.7937649
7: -6.2238603, -5.0625372, -6.2223716, -5.0627894, -0.1904031, 0.1924177
8: -0.8312588, -0.2973230, -0.8310264, -0.2983221, -0.2694727, 0.2677149
9: -2.3954906, -1.9250160, -2.3954968, -1.9257357, -0.1480759, 0.1498510

Time for backsubstitution: 5.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281754, upper bound: 0.0282491
time: 11.27 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282103, upper bound: 0.0282475
time: 19.00 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 1.6281452, 1.8224568, 1.6286439, 1.8224841, -0.0361736, 0.0356734
1: -1.5909811, -0.9891242, -1.5912116, -0.9895548, -0.1289527, 0.1311234
2: -0.1715756, 0.2522485, -0.1715466, 0.2523107, -0.2095028, 0.2110209
3: -3.9553764, -2.7518516, -3.9574003, -2.7516863, -0.5951989, 0.6017084
4: -4.2522545, -3.2465026, -4.2517505, -3.2446809, -0.3087277, 0.3067405
5: -4.5798650, -3.3540554, -4.5827208, -3.3536358, -0.6889763, 0.6962638
6: -5.3606367, -3.6064038, -5.3607283, -3.6090527, -0.7954032, 0.7936480
7: -6.2220922, -5.0633583, -6.2260966, -5.0630765, -0.1867854, 0.1994051
8: -0.8304675, -0.2989099, -0.8310645, -0.2949315, -0.2741904, 0.2651238
9: -2.3954799, -1.9250184, -2.3956532, -1.9256971, -0.1480250, 0.1500860

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282125, upper bound: 0.0282234
time: 136.63 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282477, upper bound: 0.0282245
time: 5.86 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 1.6281333, 1.8224509, 1.6286347, 1.8224841, -0.0361923, 0.0356512
1: -1.5909625, -0.9890293, -1.5912116, -0.9894763, -0.1290177, 0.1311459
2: -0.1719525, 0.2527363, -0.1715716, 0.2527372, -0.2100266, 0.2108696
3: -3.9559975, -2.7510223, -3.9574165, -2.7509778, -0.5965207, 0.6014031
4: -4.2530618, -3.2459431, -4.2523847, -3.2446799, -0.3092830, 0.3078611
5: -4.5809278, -3.3525941, -4.5827351, -3.3524270, -0.6912744, 0.6959205
6: -5.3607197, -3.6063557, -5.3607979, -3.6090527, -0.7953973, 0.7937837
7: -6.2238603, -5.0610123, -6.2260985, -5.0610127, -0.1905735, 0.1977432
8: -0.8326472, -0.2973225, -0.8328766, -0.2949307, -0.2742701, 0.2685357
9: -2.3954916, -1.9249786, -2.3956547, -1.9256710, -0.1481080, 0.1500698

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3427

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282125, upper bound: 0.0282478
time: 11.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0282239
time: 70.28 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 87.91 seconds
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282080, upper bound: 0.0281224
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0281215
NS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282080, upper bound: 0.0281446
NS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0281456
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282125, upper bound: 0.0281219
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281224
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282125, upper bound: 0.0281462
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282477, upper bound: 0.0281447
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0281708, upper bound: 0.0282490
NS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282058, upper bound: 0.0281457
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282080, upper bound: 0.0282242
NS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0282247
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282080, upper bound: 0.0282483
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282432, upper bound: 0.0282474
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0281754, upper bound: 0.0282491
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282103, upper bound: 0.0282475
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282125, upper bound: 0.0282234
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282477, upper bound: 0.0282245
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282125, upper bound: 0.0282478
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 87.91
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0282239

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 1.6286581, 1.8220248, 1.6286517, 1.8221061, -0.0353172, 0.0356373
1: -1.5907192, -0.9896256, -1.5909824, -0.9895718, -0.1286135, 0.1302457
2: -0.1703891, 0.2514289, -0.1715126, 0.2515932, -0.2066858, 0.2109259
3: -3.9552231, -2.7518563, -3.9573710, -2.7516904, -0.5933158, 0.6016444
4: -4.2513480, -3.2473364, -4.2517490, -3.2454245, -0.3073666, 0.3064536
5: -4.5791502, -3.3545225, -4.5826950, -3.3540349, -0.6848638, 0.6961635
6: -5.3580894, -3.6091681, -5.3584900, -3.6090527, -0.7951806, 0.7886845
7: -6.2204323, -5.0649371, -6.2260890, -5.0645204, -0.1777158, 0.1992973
8: -0.8295321, -0.2999103, -0.8302472, -0.2949320, -0.2741193, 0.2608501
9: -2.3948228, -1.9257308, -2.3950722, -1.9257028, -0.1478254, 0.1487759

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281567, upper bound: 0.0281215
time: 3.87 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282475, upper bound: 0.0281208
time: 16.08 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 1.6286463, 1.8220189, 1.6286428, 1.8221060, -0.0353360, 0.0356151
1: -1.5907009, -0.9895303, -1.5909824, -0.9894929, -0.1286785, 0.1302682
2: -0.1707661, 0.2519167, -0.1715372, 0.2520196, -0.2072096, 0.2107747
3: -3.9558439, -2.7510269, -3.9573867, -2.7509806, -0.5946362, 0.6013387
4: -4.2521544, -3.2467773, -4.2523828, -3.2454236, -0.3079209, 0.3075747
5: -4.5802131, -3.3530617, -4.5827093, -3.3528256, -0.6871622, 0.6958203
6: -5.3581719, -3.6091208, -5.3585591, -3.6090527, -0.7951741, 0.7888193
7: -6.2222018, -5.0625906, -6.2260904, -5.0624566, -0.1815038, 0.1976354
8: -0.8317114, -0.2983232, -0.8320589, -0.2949319, -0.2741987, 0.2642623
9: -2.3948345, -1.9256909, -2.3950732, -1.9256768, -0.1479084, 0.1487595

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281566, upper bound: 0.0281450
time: 39.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281458
time: 3.29 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 1.6282022, 1.8224114, 1.6286664, 1.8224330, -0.0360770, 0.0356574
1: -1.5910429, -0.9895558, -1.5909179, -0.9900631, -0.1283069, 0.1303537
2: -0.1716492, 0.2515486, -0.1704810, 0.2513167, -0.2087141, 0.2085986
3: -3.9554706, -2.7532394, -3.9552269, -2.7534513, -0.5961668, 0.5928595
4: -4.2511821, -3.2459922, -4.2499051, -3.2463565, -0.3063136, 0.3044336
5: -4.5804186, -3.3561997, -4.5793114, -3.3567204, -0.6893194, 0.6848691
6: -5.3595724, -3.6070585, -5.3596935, -3.6091361, -0.7953005, 0.7923862
7: -6.2235241, -5.0663991, -6.2216187, -5.0672655, -0.1881664, 0.1818824
8: -0.8276709, -0.2973558, -0.8265783, -0.2993733, -0.2648406, 0.2632552
9: -2.3954358, -1.9251627, -2.3954427, -1.9258934, -0.1478749, 0.1495843

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0280799, upper bound: 0.0282477
time: 3.35 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281707, upper bound: 0.0282482
time: 8.35 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 1.6281950, 1.8224115, 1.6286557, 1.8224660, -0.0361090, 0.0356613
1: -1.5910429, -0.9894211, -1.5911659, -0.9898968, -0.1283680, 0.1307103
2: -0.1716712, 0.2518699, -0.1712516, 0.2516920, -0.2087377, 0.2095089
3: -3.9554825, -2.7526865, -3.9565868, -2.7527440, -0.5964134, 0.5947685
4: -4.2516894, -3.2459912, -4.2506762, -3.2450881, -0.3081151, 0.3050146
5: -4.5804272, -3.3552675, -4.5816221, -3.3554630, -0.6898260, 0.6881126
6: -5.3595829, -3.6070585, -5.3597198, -3.6090684, -0.7954106, 0.7924047
7: -6.2235255, -5.0648742, -6.2253447, -5.0654883, -0.1883368, 0.1872087
8: -0.8290582, -0.2973554, -0.8284311, -0.2959821, -0.2696382, 0.2640757
9: -2.3954363, -1.9251246, -2.3956001, -1.9258289, -0.1479069, 0.1498030

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281170, upper bound: 0.0282488
time: 16.74 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282080, upper bound: 0.0282493
time: 3.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 1.6281799, 1.8224117, 1.6286390, 1.8224490, -0.0361107, 0.0356840
1: -1.5910429, -0.9891014, -1.5912116, -0.9895123, -0.1284376, 0.1310784
2: -0.1717186, 0.2527234, -0.1714066, 0.2527352, -0.2089031, 0.2104493
3: -3.9555123, -2.7512560, -3.9570155, -2.7509818, -0.5964675, 0.5966141
4: -4.2529945, -3.2459891, -4.2523270, -3.2446904, -0.3098422, 0.3064350
5: -4.5804539, -3.3528292, -4.5823421, -3.3524318, -0.6901498, 0.6912870
6: -5.3597393, -3.6070585, -5.3599100, -3.6090527, -0.7955909, 0.7924544
7: -6.2235274, -5.0613294, -6.2258072, -5.0611563, -0.1874428, 0.1924055
8: -0.8326119, -0.2973551, -0.8328456, -0.2949325, -0.2742904, 0.2654876
9: -2.3954394, -1.9249982, -2.3956132, -1.9256765, -0.1479394, 0.1500016

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281521, upper bound: 0.0282483
time: 24.38 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0282486
time: 11.49 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 1.6281594, 1.8224505, 1.6286662, 1.8224684, -0.0361539, 0.0356185
1: -1.5909625, -0.9894836, -1.5909179, -0.9900266, -0.1285678, 0.1304213
2: -0.1718834, 0.2515600, -0.1706463, 0.2513169, -0.2089278, 0.2089943
3: -3.9559555, -2.7530079, -3.9556277, -2.7534499, -0.5944810, 0.5976455
4: -4.2512479, -3.2459462, -4.2499628, -3.2463460, -0.3057538, 0.3056329
5: -4.5808916, -3.3559699, -4.5797038, -3.3567204, -0.6876974, 0.6894995
6: -5.3605480, -3.6063557, -5.3605723, -3.6091361, -0.7951012, 0.7935612
7: -6.2238564, -5.0661960, -6.2219100, -5.0672655, -0.1853621, 0.1872154
8: -0.8277059, -0.2973233, -0.8266088, -0.2993720, -0.2648201, 0.2633050
9: -2.3954875, -1.9251471, -2.3954840, -1.9258931, -0.1478922, 0.1496525

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0280843, upper bound: 0.0282486
time: 27.87 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281752, upper bound: 0.0282490
time: 5.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 1.6281446, 1.8224506, 1.6286497, 1.8224514, -0.0361556, 0.0356411
1: -1.5909625, -0.9891641, -1.5909631, -0.9896425, -0.1286376, 0.1307886
2: -0.1719308, 0.2524136, -0.1708003, 0.2523599, -0.2090932, 0.2099336
3: -3.9559860, -2.7515769, -3.9560552, -2.7516878, -0.5945356, 0.5994895
4: -4.2525520, -3.2459435, -4.2516155, -3.2459483, -0.3074807, 0.3070536
5: -4.5809183, -3.3535309, -4.5804224, -3.3536897, -0.6880221, 0.6926725
6: -5.3607044, -3.6063557, -5.3607621, -3.6091208, -0.7952821, 0.7936114
7: -6.2238593, -5.0626516, -6.2223711, -5.0629330, -0.1844680, 0.1924129
8: -0.8312583, -0.2973230, -0.8310257, -0.2983221, -0.2694725, 0.2647171
9: -2.3954904, -1.9250202, -2.3954968, -1.9257410, -0.1479247, 0.1498508

Time for backsubstitution: 6.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281193, upper bound: 0.0282476
time: 19.29 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282102, upper bound: 0.0282481
time: 9.49 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 1.6281486, 1.8224568, 1.6286477, 1.8224841, -0.0361690, 0.0356673
1: -1.5909811, -0.9891243, -1.5912116, -0.9895551, -0.1286336, 0.1311232
2: -0.1715757, 0.2522472, -0.1715468, 0.2523089, -0.2085931, 0.2109960
3: -3.9553773, -2.7518542, -3.9573998, -2.7516890, -0.5934601, 0.6017057
4: -4.2522540, -3.2465026, -4.2517509, -3.2446811, -0.3087270, 0.3065140
5: -4.5798645, -3.3540597, -4.5827203, -3.3536413, -0.6862290, 0.6962603
6: -5.3606310, -3.6064038, -5.3607225, -3.6090527, -0.7953985, 0.7934942
7: -6.2220922, -5.0634723, -6.2260971, -5.0632195, -0.1808506, 0.1994003
8: -0.8304673, -0.2989099, -0.8310637, -0.2949315, -0.2741901, 0.2621257
9: -2.3954797, -1.9250228, -2.3956532, -1.9257028, -0.1478740, 0.1500859

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281567, upper bound: 0.0282249
time: 13.88 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282475, upper bound: 0.0282240
time: 3.11 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 1.6281520, 1.8224508, 1.6286553, 1.8225009, -0.0361859, 0.0356223
1: -1.5909625, -0.9893491, -1.5911659, -0.9898601, -0.1286287, 0.1307779
2: -0.1719050, 0.2518812, -0.1714170, 0.2516921, -0.2089514, 0.2099043
3: -3.9559684, -2.7524567, -3.9569881, -2.7527432, -0.5947270, 0.5995547
4: -4.2517557, -3.2459455, -4.2507348, -3.2450771, -0.3075550, 0.3062143
5: -4.5809007, -3.3550375, -4.5820136, -3.3554630, -0.6882041, 0.6927429
6: -5.3605590, -3.6063557, -5.3605990, -3.6090684, -0.7952120, 0.7935796
7: -6.2238574, -5.0646720, -6.2256360, -5.0654888, -0.1855326, 0.1925418
8: -0.8290932, -0.2973235, -0.8284616, -0.2959812, -0.2696174, 0.2641262
9: -2.3954885, -1.9251097, -2.3956416, -1.9258289, -0.1479245, 0.1498712

Time for backsubstitution: 6.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281215, upper bound: 0.0282476
time: 11.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282124, upper bound: 0.0282490
time: 13.45 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 1.6281370, 1.8224509, 1.6286389, 1.8224841, -0.0361876, 0.0356451
1: -1.5909625, -0.9890294, -1.5912116, -0.9894762, -0.1286984, 0.1311458
2: -0.1719527, 0.2527350, -0.1715716, 0.2527354, -0.2091168, 0.2108445
3: -3.9559972, -2.7510245, -3.9574161, -2.7509804, -0.5947814, 0.6013998
4: -4.2530608, -3.2459433, -4.2523842, -3.2446799, -0.3092822, 0.3076346
5: -4.5809269, -3.3525982, -4.5827346, -3.3524318, -0.6885271, 0.6959169
6: -5.3607149, -3.6063557, -5.3607922, -3.6090527, -0.7953916, 0.7936299
7: -6.2238603, -5.0611267, -6.2260985, -5.0611558, -0.1846385, 0.1977385
8: -0.8326467, -0.2973225, -0.8328760, -0.2949309, -0.2742696, 0.2655377
9: -2.3954918, -1.9249829, -2.3956544, -1.9256761, -0.1479568, 0.1500697

Time for backsubstitution: 6.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3456
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0281566, upper bound: 0.0282473
time: 12.15 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0282487
time: 21.41 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 39.91 seconds
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0281567, upper bound: 0.0281215
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0282475, upper bound: 0.0281208
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0281566, upper bound: 0.0281450
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0281458
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0280799, upper bound: 0.0282477
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0281707, upper bound: 0.0282482
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0281170, upper bound: 0.0282488
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0282080, upper bound: 0.0282493
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0281521, upper bound: 0.0282483
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0282431, upper bound: 0.0282486
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0280843, upper bound: 0.0282486
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0281752, upper bound: 0.0282490
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0281193, upper bound: 0.0282476
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0282102, upper bound: 0.0282481
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0281567, upper bound: 0.0282249
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0282475, upper bound: 0.0282240
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0281215, upper bound: 0.0282476
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0282124, upper bound: 0.0282490
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0281566, upper bound: 0.0282473
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 39.91
Output dim: 2, lower bound: -0.0282476, upper bound: 0.0282487

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 1.6286578, 1.8220197, 1.6286519, 1.8221028, -0.0353167, 0.0355892
1: -1.5907192, -0.9896297, -1.5909824, -0.9895742, -0.1286120, 0.1288796
2: -0.1703884, 0.2514289, -0.1715118, 0.2515932, -0.2059639, 0.2109246
3: -3.9552188, -2.7518563, -3.9573677, -2.7516901, -0.5892024, 0.6016392
4: -4.2513466, -3.2473361, -4.2517481, -3.2454245, -0.3069782, 0.3064384
5: -4.5791478, -3.3545225, -4.5826941, -3.3540349, -0.6808007, 0.6961606
6: -5.3580723, -3.6091681, -5.3584785, -3.6090527, -0.7896199, 0.7886726
7: -6.2204318, -5.0649371, -6.2260885, -5.0645204, -0.1739096, 0.1992954
8: -0.8295209, -0.2999103, -0.8302391, -0.2949320, -0.2741261, 0.2608420
9: -2.3948090, -1.9257308, -2.3950624, -1.9257027, -0.1475924, 0.1487749

Time for backsubstitution: 6.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2540

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282406, upper bound: 0.0281024
time: 2.94 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282340, upper bound: 0.0281081
time: 12.54 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 1.6286466, 1.8220137, 1.6286429, 1.8221027, -0.0353354, 0.0355670
1: -1.5907009, -0.9895344, -1.5909824, -0.9894953, -0.1286770, 0.1289021
2: -0.1707652, 0.2519167, -0.1715368, 0.2520196, -0.2064875, 0.2107733
3: -3.9558392, -2.7510266, -3.9573836, -2.7509804, -0.5905240, 0.6013334
4: -4.2521524, -3.2467775, -4.2523808, -3.2454238, -0.3075312, 0.3075591
5: -4.5802107, -3.3530617, -4.5827079, -3.3528256, -0.6830986, 0.6958168
6: -5.3581543, -3.6091208, -5.3585482, -3.6090527, -0.7896132, 0.7888074
7: -6.2222004, -5.0625906, -6.2260900, -5.0624566, -0.1776975, 0.1976334
8: -0.8317000, -0.2983232, -0.8320513, -0.2949322, -0.2742053, 0.2642542
9: -2.3948209, -1.9256909, -2.3950639, -1.9256766, -0.1476752, 0.1487588

Time for backsubstitution: 6.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2540

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282405, upper bound: 0.0281266
time: 3.07 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282340, upper bound: 0.0281320
time: 29.46 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 1.6281917, 1.8223795, 1.6286690, 1.8224090, -0.0360202, 0.0356056
1: -1.5912913, -0.9911115, -1.5909179, -0.9912822, -0.1268492, 0.1285653
2: -0.1701390, 0.2514230, -0.1693633, 0.2513162, -0.2071001, 0.2072725
3: -3.9498668, -2.7539101, -3.9510403, -2.7534642, -0.5905662, 0.5879872
4: -4.2487431, -3.2462521, -4.2480769, -3.2464466, -0.3044122, 0.3038673
5: -4.5747070, -3.3569031, -4.5750523, -3.3567204, -0.6836197, 0.6799133
6: -5.3522682, -3.6079030, -5.3542290, -3.6091361, -0.7879701, 0.7860532
7: -6.2204781, -5.0663085, -6.2192531, -5.0672655, -0.1846782, 0.1786695
8: -0.8275996, -0.2972867, -0.8265253, -0.2993760, -0.2647676, 0.2632737
9: -2.3949838, -1.9250962, -2.3951097, -1.9258943, -0.1472716, 0.1491093

Time for backsubstitution: 6.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2540

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0280728, upper bound: 0.0282299
time: 2.95 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0280663, upper bound: 0.0282342
time: 23.19 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 1.6282022, 1.8224063, 1.6286666, 1.8224297, -0.0360764, 0.0356094
1: -1.5910429, -0.9895595, -1.5909179, -0.9900655, -0.1283054, 0.1289877
2: -0.1716485, 0.2515486, -0.1704806, 0.2513167, -0.2079923, 0.2085972
3: -3.9554663, -2.7532401, -3.9552236, -2.7534509, -0.5920529, 0.5928549
4: -4.2511797, -3.2459919, -4.2499046, -3.2463565, -0.3059299, 0.3044182
5: -4.5804148, -3.3561997, -4.5793099, -3.3567204, -0.6852520, 0.6848657
6: -5.3595572, -3.6070585, -5.3596811, -3.6091361, -0.7897406, 0.7923746
7: -6.2235222, -5.0663991, -6.2216187, -5.0672655, -0.1843851, 0.1818805
8: -0.8276595, -0.2973558, -0.8265706, -0.2993729, -0.2648475, 0.2632469
9: -2.3954215, -1.9251627, -2.3954329, -1.9258934, -0.1476417, 0.1495836

Time for backsubstitution: 6.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2540

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281637, upper bound: 0.0282294
time: 7.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281571, upper bound: 0.0282333
time: 15.66 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 1.6281846, 1.8223794, 1.6286581, 1.8224418, -0.0360522, 0.0356095
1: -1.5912913, -0.9909766, -1.5911661, -0.9911158, -0.1269101, 0.1289219
2: -0.1701610, 0.2517443, -0.1701345, 0.2516911, -0.2071238, 0.2081829
3: -3.9498775, -2.7533576, -3.9523997, -2.7527571, -0.5908124, 0.5898966
4: -4.2492509, -3.2462509, -4.2488489, -3.2451789, -0.3062146, 0.3044443
5: -4.5747166, -3.3559713, -4.5773630, -3.3554623, -0.6841266, 0.6831571
6: -5.3522797, -3.6079030, -5.3542562, -3.6090684, -0.7880797, 0.7860719
7: -6.2204790, -5.0647836, -6.2229795, -5.0654888, -0.1848485, 0.1839958
8: -0.8289872, -0.2972864, -0.8283780, -0.2959846, -0.2695647, 0.2640944
9: -2.3949850, -1.9250591, -2.3952670, -1.9258298, -0.1473038, 0.1493280

Time for backsubstitution: 6.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2540

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281100, upper bound: 0.0282288
time: 19.05 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281034, upper bound: 0.0282341
time: 7.39 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 1.6281946, 1.8224065, 1.6286557, 1.8224627, -0.0361084, 0.0356132
1: -1.5910429, -0.9894240, -1.5911659, -0.9898993, -0.1283664, 0.1293443
2: -0.1716703, 0.2518699, -0.1712514, 0.2516920, -0.2080157, 0.2095074
3: -3.9554772, -2.7526865, -3.9565837, -2.7527437, -0.5922997, 0.5947635
4: -4.2516875, -3.2459912, -4.2506752, -3.2450879, -0.3077319, 0.3049991
5: -4.5804248, -3.3552675, -4.5816193, -3.3554628, -0.6857581, 0.6881094
6: -5.3595657, -3.6070585, -5.3597093, -3.6090684, -0.7898507, 0.7923933
7: -6.2235241, -5.0648742, -6.2253442, -5.0654888, -0.1845554, 0.1872068
8: -0.8290467, -0.2973554, -0.8284234, -0.2959821, -0.2696450, 0.2640675
9: -2.3954225, -1.9251248, -2.3955905, -1.9258288, -0.1476738, 0.1498024

Time for backsubstitution: 6.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2540

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282009, upper bound: 0.0282295
time: 10.01 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281943, upper bound: 0.0282340
time: 9.08 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 1.6281695, 1.8223795, 1.6286414, 1.8224250, -0.0360539, 0.0356322
1: -1.5912913, -0.9906571, -1.5912116, -0.9907314, -0.1269799, 0.1292899
2: -0.1702083, 0.2525982, -0.1702895, 0.2527345, -0.2072889, 0.2091240
3: -3.9499078, -2.7519269, -3.9528282, -2.7509944, -0.5908668, 0.5917428
4: -4.2505560, -3.2462482, -4.2504997, -3.2447808, -0.3079424, 0.3058643
5: -4.5747428, -3.3535323, -4.5780840, -3.3524323, -0.6844504, 0.6863315
6: -5.3524351, -3.6079030, -5.3544474, -3.6090527, -0.7882595, 0.7861214
7: -6.2204809, -5.0612388, -6.2234416, -5.0611563, -0.1839545, 0.1891925
8: -0.8325400, -0.2972858, -0.8327917, -0.2949352, -0.2742171, 0.2655063
9: -2.3949881, -1.9249319, -2.3952806, -1.9256778, -0.1473362, 0.1495266

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2540

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281451, upper bound: 0.0282296
time: 3.38 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0281385, upper bound: 0.0282351
time: 15.93 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 1.6281799, 1.8224065, 1.6286390, 1.8224455, -0.0361101, 0.0356359
1: -1.5910429, -0.9891050, -1.5912116, -0.9895148, -0.1284361, 0.1297123
2: -0.1717177, 0.2527234, -0.1714063, 0.2527352, -0.2081810, 0.2104480
3: -3.9555078, -2.7512565, -3.9570122, -2.7509813, -0.5923533, 0.5966089
4: -4.2529926, -3.2459888, -4.2523260, -3.2446904, -0.3094600, 0.3064196
5: -4.5804510, -3.3528292, -4.5823402, -3.3524318, -0.6860819, 0.6912831
6: -5.3597221, -3.6070585, -5.3598990, -3.6090527, -0.7900300, 0.7924426
7: -6.2235270, -5.0613289, -6.2258067, -5.0611563, -0.1836615, 0.1924036
8: -0.8326002, -0.2973551, -0.8328375, -0.2949325, -0.2742968, 0.2654794
9: -2.3954258, -1.9249980, -2.3956039, -1.9256765, -0.1477064, 0.1500008

Time for backsubstitution: 6.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 3198

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2540

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282361, upper bound: 0.0282279
time: 54.10 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282295, upper bound: 0.0282346
time: 46.22 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 36.34 + 1815.17 = 1851.51 seconds
