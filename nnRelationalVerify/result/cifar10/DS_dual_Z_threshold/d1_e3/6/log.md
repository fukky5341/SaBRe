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
execution time: IAR + RelationalAnalysis = 7.81 + 28.88 = 36.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0282724, upper bound: 0.0282727

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3437

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282529, upper bound: 0.0282728
time: 50.13 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282723, upper bound: 0.0282525
time: 42.19 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 92.38 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 92.38
Output dim: 2, lower bound: -0.0282529, upper bound: 0.0282728
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 92.38
Output dim: 2, lower bound: -0.0282723, upper bound: 0.0282525

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0361605, 0.0361604
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1306732, 0.1306799
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116634, 0.2116644
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6011009, 0.6011019
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086479, 0.3086452
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6950319, 0.6950040
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7995768, 0.7995820
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1969496, 0.1968732
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718222, 0.2718837
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1487373, 0.1487556

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2552

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0282718
time: 18.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282525, upper bound: 0.0282724
time: 6.73 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0361604, 0.0361605
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1306800, 0.1306732
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116645, 0.2116634
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6011019, 0.6011009
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086452, 0.3086479
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6950040, 0.6950319
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7995820, 0.7995768
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1968732, 0.1969496
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718837, 0.2718222
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1487555, 0.1487373

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2552

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282721, upper bound: 0.0282535
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282720, upper bound: 0.0282531
time: 3.39 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 12.66 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.66
Output dim: 2, lower bound: -0.0282527, upper bound: 0.0282718
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.66
Output dim: 2, lower bound: -0.0282525, upper bound: 0.0282724
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.66
Output dim: 2, lower bound: -0.0282721, upper bound: 0.0282535
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.66
Output dim: 2, lower bound: -0.0282720, upper bound: 0.0282531

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0361594, 0.0361592
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1306137, 0.1306292
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116619, 0.2116628
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6010766, 0.6010873
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086326, 0.3086296
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6950159, 0.6949902
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7995646, 0.7995741
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1969311, 0.1968563
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718214, 0.2718801
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486967, 0.1487120

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2602

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282486, upper bound: 0.0282676
time: 38.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282479, upper bound: 0.0282684
time: 70.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0361593, 0.0361593
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1306224, 0.1306206
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116618, 0.2116629
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6010866, 0.6010787
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086323, 0.3086300
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6950178, 0.6949894
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7995694, 0.7995701
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1969328, 0.1968555
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718186, 0.2718830
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486939, 0.1487168

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2602

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282484, upper bound: 0.0282680
time: 18.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282478, upper bound: 0.0282693
time: 3.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0361593, 0.0361593
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1306206, 0.1306224
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116629, 0.2116618
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6010787, 0.6010866
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086300, 0.3086323
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6949894, 0.6950181
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7995703, 0.7995694
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1968555, 0.1969327
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718830, 0.2718186
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1487168, 0.1486939

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2602

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282680, upper bound: 0.0282475
time: 23.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282674, upper bound: 0.0282494
time: 21.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0361592, 0.0361594
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1306292, 0.1306137
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116628, 0.2116619
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6010876, 0.6010766
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3086296, 0.3086326
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6949902, 0.6950159
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7995741, 0.7995651
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1968562, 0.1969311
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718801, 0.2718214
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1487121, 0.1486967

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2602

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282679, upper bound: 0.0282491
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282672, upper bound: 0.0282478
time: 16.97 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 26.68 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 2, lower bound: -0.0282486, upper bound: 0.0282676
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 2, lower bound: -0.0282479, upper bound: 0.0282684
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 2, lower bound: -0.0282484, upper bound: 0.0282680
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 2, lower bound: -0.0282478, upper bound: 0.0282693
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 2, lower bound: -0.0282680, upper bound: 0.0282475
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 2, lower bound: -0.0282674, upper bound: 0.0282494
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 2, lower bound: -0.0282679, upper bound: 0.0282491
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 2, lower bound: -0.0282672, upper bound: 0.0282478

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360976, 0.0360972
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304505, 0.1304695
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116815, 0.2116824
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004043, 0.6004063
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058779, 0.3058512
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943331, 0.6942906
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992485, 0.7992535
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1934479, 0.1933547
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718199, 0.2718787
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486948, 0.1487107

Time for backsubstitution: 6.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2973

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282415, upper bound: 0.0282608
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282419, upper bound: 0.0282599
time: 3.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360974, 0.0360975
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304541, 0.1304659
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116815, 0.2116824
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6003952, 0.6004153
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058543, 0.3058748
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943166, 0.6943071
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992442, 0.7992578
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1934295, 0.1933731
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718199, 0.2718787
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486953, 0.1487101

Time for backsubstitution: 6.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2973

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282409, upper bound: 0.0282611
time: 4.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282413, upper bound: 0.0282593
time: 51.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360975, 0.0360973
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304592, 0.1304609
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116813, 0.2116825
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004143, 0.6003977
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058774, 0.3058517
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943350, 0.6942902
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992527, 0.7992492
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1934495, 0.1933540
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718171, 0.2718815
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486920, 0.1487154

Time for backsubstitution: 6.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2973

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282414, upper bound: 0.0282613
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282418, upper bound: 0.0282592
time: 4.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360973, 0.0360976
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304628, 0.1304573
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116813, 0.2116825
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004052, 0.6004068
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058538, 0.3058753
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943188, 0.6943066
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992485, 0.7992537
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1934311, 0.1933723
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718171, 0.2718815
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486925, 0.1487149

Time for backsubstitution: 6.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2973

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282407, upper bound: 0.0282613
time: 33.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282411, upper bound: 0.0282599
time: 41.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360976, 0.0360973
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304573, 0.1304628
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116825, 0.2116814
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004066, 0.6004053
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058753, 0.3058538
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943066, 0.6943185
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992537, 0.7992485
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933723, 0.1934312
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718815, 0.2718171
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1487149, 0.1486925

Time for backsubstitution: 6.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2973

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282595, upper bound: 0.0282418
time: 40.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282606, upper bound: 0.0282419
time: 3.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360973, 0.0360975
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304609, 0.1304592
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116825, 0.2116814
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6003976, 0.6004144
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058517, 0.3058774
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942902, 0.6943350
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992489, 0.7992527
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933540, 0.1934495
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718815, 0.2718171
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1487154, 0.1486919

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2973

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282589, upper bound: 0.0282420
time: 69.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282600, upper bound: 0.0282426
time: 3.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360975, 0.0360974
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304659, 0.1304541
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116824, 0.2116815
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004152, 0.6003953
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058748, 0.3058543
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943073, 0.6943166
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992575, 0.7992442
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933730, 0.1934295
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718786, 0.2718199
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1487101, 0.1486953

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2973

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282594, upper bound: 0.0282429
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282605, upper bound: 0.0282404
time: 39.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360972, 0.0360976
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304695, 0.1304505
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116824, 0.2116815
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004062, 0.6004044
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058512, 0.3058779
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942909, 0.6943331
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992537, 0.7992485
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933547, 0.1934479
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718786, 0.2718199
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1487107, 0.1486948

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2973

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282587, upper bound: 0.0282432
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282599, upper bound: 0.0282422
time: 24.77 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 34.60 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282415, upper bound: 0.0282608
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282419, upper bound: 0.0282599
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282409, upper bound: 0.0282611
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282413, upper bound: 0.0282593
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282414, upper bound: 0.0282613
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282418, upper bound: 0.0282592
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282407, upper bound: 0.0282613
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282411, upper bound: 0.0282599
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282595, upper bound: 0.0282418
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282606, upper bound: 0.0282419
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282589, upper bound: 0.0282420
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282600, upper bound: 0.0282426
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282594, upper bound: 0.0282429
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282605, upper bound: 0.0282404
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282587, upper bound: 0.0282432
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -0.0282599, upper bound: 0.0282422

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360908, 0.0360867
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303264, 0.1303958
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116722, 0.2116714
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004028, 0.6004046
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058550, 0.3058136
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943116, 0.6942677
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992094, 0.7992218
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933995, 0.1933087
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718148, 0.2718736
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486453, 0.1486787

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282414, upper bound: 0.0282592
time: 9.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282403, upper bound: 0.0282599
time: 20.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360871, 0.0360904
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303768, 0.1303454
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116705, 0.2116707
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004024, 0.6004046
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058401, 0.3058282
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943099, 0.6942649
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992165, 0.7992144
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1934018, 0.1933001
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718149, 0.2718735
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486621, 0.1486613

Time for backsubstitution: 5.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282418, upper bound: 0.0282586
time: 17.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282406, upper bound: 0.0282584
time: 39.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360905, 0.0360869
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303300, 0.1303922
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116722, 0.2116714
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6003938, 0.6004137
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058314, 0.3058372
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942954, 0.6942840
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992051, 0.7992260
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933812, 0.1933270
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718148, 0.2718736
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486459, 0.1486781

Time for backsubstitution: 5.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282408, upper bound: 0.0282604
time: 13.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282397, upper bound: 0.0282595
time: 77.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360868, 0.0360906
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303804, 0.1303418
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116705, 0.2116707
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6003933, 0.6004137
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058165, 0.3058518
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942935, 0.6942811
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992127, 0.7992187
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933835, 0.1933185
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718149, 0.2718735
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486626, 0.1486607

Time for backsubstitution: 5.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282412, upper bound: 0.0282592
time: 62.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282400, upper bound: 0.0282607
time: 41.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360907, 0.0360867
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303351, 0.1303873
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116721, 0.2116716
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004128, 0.6003958
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058546, 0.3058139
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943138, 0.6942670
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992136, 0.7992175
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1934012, 0.1933079
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718120, 0.2718765
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486425, 0.1486834

Time for backsubstitution: 5.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282413, upper bound: 0.0282599
time: 5.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282402, upper bound: 0.0282605
time: 7.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360870, 0.0360904
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303855, 0.1303368
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116704, 0.2116708
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004124, 0.6003958
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058398, 0.3058287
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943119, 0.6942642
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992213, 0.7992101
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1934034, 0.1932994
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718121, 0.2718763
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486593, 0.1486660

Time for backsubstitution: 5.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282417, upper bound: 0.0282593
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282405, upper bound: 0.0282600
time: 4.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360904, 0.0360870
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303387, 0.1303836
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116721, 0.2116717
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004038, 0.6004051
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058310, 0.3058375
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942973, 0.6942835
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992094, 0.7992218
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933828, 0.1933263
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718120, 0.2718765
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486431, 0.1486829

Time for backsubstitution: 5.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282407, upper bound: 0.0282597
time: 5.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282396, upper bound: 0.0282617
time: 8.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360867, 0.0360907
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303891, 0.1303332
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116704, 0.2116708
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004033, 0.6004051
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058162, 0.3058523
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942954, 0.6942806
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992165, 0.7992146
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933851, 0.1933177
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718121, 0.2718763
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486598, 0.1486655

Time for backsubstitution: 5.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282411, upper bound: 0.0282598
time: 91.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282399, upper bound: 0.0282601
time: 19.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360907, 0.0360867
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303332, 0.1303891
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116709, 0.2116705
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004052, 0.6004032
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058523, 0.3058162
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942806, 0.6942954
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992146, 0.7992167
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933178, 0.1933851
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718763, 0.2718121
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486655, 0.1486598

Time for backsubstitution: 5.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282595, upper bound: 0.0282395
time: 9.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282587, upper bound: 0.0282417
time: 17.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360870, 0.0360904
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303836, 0.1303387
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116716, 0.2116720
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004052, 0.6004037
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058375, 0.3058310
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942835, 0.6942973
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992218, 0.7992094
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933263, 0.1933828
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718765, 0.2718120
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486829, 0.1486431

Time for backsubstitution: 5.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282606, upper bound: 0.0282408
time: 4.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282594, upper bound: 0.0282413
time: 21.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360904, 0.0360870
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303368, 0.1303855
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116709, 0.2116705
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6003957, 0.6004122
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058287, 0.3058398
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942642, 0.6943119
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992098, 0.7992210
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1932994, 0.1934034
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718763, 0.2718121
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486660, 0.1486593

Time for backsubstitution: 5.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282588, upper bound: 0.0282403
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282580, upper bound: 0.0282414
time: 11.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360867, 0.0360907
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303872, 0.1303350
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116716, 0.2116720
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6003957, 0.6004127
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058139, 0.3058546
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942670, 0.6943138
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992175, 0.7992136
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933079, 0.1934012
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718765, 0.2718120
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486834, 0.1486425

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282599, upper bound: 0.0282408
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282587, upper bound: 0.0282411
time: 16.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360906, 0.0360868
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303418, 0.1303804
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116706, 0.2116706
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004138, 0.6003932
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058518, 0.3058165
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942813, 0.6942935
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992184, 0.7992125
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933185, 0.1933835
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718735, 0.2718149
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486607, 0.1486626

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282593, upper bound: 0.0282414
time: 5.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282586, upper bound: 0.0282420
time: 13.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360869, 0.0360905
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303922, 0.1303300
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116715, 0.2116722
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004138, 0.6003937
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058372, 0.3058314
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942840, 0.6942952
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992260, 0.7992051
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933270, 0.1933812
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718736, 0.2718147
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486782, 0.1486459

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282605, upper bound: 0.0282402
time: 83.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282593, upper bound: 0.0282413
time: 42.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360904, 0.0360871
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303454, 0.1303768
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116706, 0.2116706
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004047, 0.6004022
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058282, 0.3058401
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942649, 0.6943099
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992146, 0.7992167
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933001, 0.1934018
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718735, 0.2718149
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486613, 0.1486621

Time for backsubstitution: 6.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282587, upper bound: 0.0282411
time: 15.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282579, upper bound: 0.0282423
time: 13.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360867, 0.0360908
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303958, 0.1303263
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116715, 0.2116722
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004047, 0.6004027
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058136, 0.3058550
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6942677, 0.6943116
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992218, 0.7992094
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933087, 0.1933995
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718736, 0.2718147
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486787, 0.1486453

Time for backsubstitution: 6.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3198
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282598, upper bound: 0.0282414
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282586, upper bound: 0.0282430
time: 13.36 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 24.16 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282414, upper bound: 0.0282592
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282403, upper bound: 0.0282599
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282418, upper bound: 0.0282586
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282406, upper bound: 0.0282584
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282408, upper bound: 0.0282604
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282397, upper bound: 0.0282595
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282412, upper bound: 0.0282592
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282400, upper bound: 0.0282607
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282413, upper bound: 0.0282599
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282402, upper bound: 0.0282605
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282417, upper bound: 0.0282593
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282405, upper bound: 0.0282600
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282407, upper bound: 0.0282597
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282396, upper bound: 0.0282617
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282411, upper bound: 0.0282598
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282399, upper bound: 0.0282601
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282595, upper bound: 0.0282395
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282587, upper bound: 0.0282417
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282606, upper bound: 0.0282408
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282594, upper bound: 0.0282413
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282588, upper bound: 0.0282403
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282580, upper bound: 0.0282414
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282599, upper bound: 0.0282408
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282587, upper bound: 0.0282411
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282593, upper bound: 0.0282414
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282586, upper bound: 0.0282420
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282605, upper bound: 0.0282402
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282593, upper bound: 0.0282413
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282587, upper bound: 0.0282411
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282579, upper bound: 0.0282423
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282598, upper bound: 0.0282414
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 24.16
Output dim: 2, lower bound: -0.0282586, upper bound: 0.0282430

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360933, 0.0360893
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303979, 0.1304628
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116680, 0.2116673
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004009, 0.6004028
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058558, 0.3058144
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943238, 0.6942801
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7991953, 0.7992069
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933981, 0.1933072
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718214, 0.2718768
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486726, 0.1487005

Time for backsubstitution: 6.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2531

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282395, upper bound: 0.0282586
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282401, upper bound: 0.0282560
time: 15.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360934, 0.0360892
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303934, 0.1304659
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116680, 0.2116672
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004009, 0.6004028
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058559, 0.3058141
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943240, 0.6942799
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7991943, 0.7992078
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933981, 0.1933072
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718179, 0.2718804
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486671, 0.1487051

Time for backsubstitution: 6.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2531

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282384, upper bound: 0.0282594
time: 7.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282389, upper bound: 0.0282587
time: 3.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360896, 0.0360930
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304469, 0.1304124
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116663, 0.2116665
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004004, 0.6004028
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058407, 0.3058290
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943221, 0.6942770
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992029, 0.7991995
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1934004, 0.1932987
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718216, 0.2718767
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486885, 0.1486830

Time for backsubstitution: 6.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2531

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282399, upper bound: 0.0282569
time: 3.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282404, upper bound: 0.0282572
time: 3.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360897, 0.0360929
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304438, 0.1304170
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116664, 0.2116665
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004004, 0.6004028
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058410, 0.3058289
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943223, 0.6942770
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7992020, 0.7992004
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1934004, 0.1932987
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718182, 0.2718801
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486839, 0.1486886

Time for backsubstitution: 6.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2531

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282387, upper bound: 0.0282585
time: 14.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282393, upper bound: 0.0282582
time: 5.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360930, 0.0360896
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304016, 0.1304592
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116680, 0.2116673
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6003919, 0.6004119
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058321, 0.3058380
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943073, 0.6942966
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7991910, 0.7992111
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933798, 0.1933255
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718214, 0.2718768
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486731, 0.1486999

Time for backsubstitution: 6.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2531

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282388, upper bound: 0.0282583
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282394, upper bound: 0.0282584
time: 5.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360931, 0.0360894
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1303970, 0.1304623
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116680, 0.2116672
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6003919, 0.6004119
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058323, 0.3058378
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943076, 0.6942961
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7991900, 0.7992121
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933798, 0.1933255
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718180, 0.2718804
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486677, 0.1487046

Time for backsubstitution: 6.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2531

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282377, upper bound: 0.0282606
time: 3.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282383, upper bound: 0.0282594
time: 8.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360893, 0.0360932
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304505, 0.1304088
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116663, 0.2116665
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6003914, 0.6004119
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058171, 0.3058527
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943057, 0.6942935
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7991986, 0.7992040
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933820, 0.1933170
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718216, 0.2718767
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486891, 0.1486825

Time for backsubstitution: 6.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2531

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282392, upper bound: 0.0282578
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282398, upper bound: 0.0282578
time: 3.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360895, 0.0360931
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304474, 0.1304133
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116664, 0.2116665
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6003914, 0.6004119
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058174, 0.3058525
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943059, 0.6942933
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7991977, 0.7992047
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933820, 0.1933170
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718182, 0.2718801
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486844, 0.1486880

Time for backsubstitution: 6.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2531

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282380, upper bound: 0.0282575
time: 11.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282386, upper bound: 0.0282588
time: 8.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360932, 0.0360894
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304066, 0.1304543
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116678, 0.2116674
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004109, 0.6003940
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058553, 0.3058147
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943259, 0.6942794
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7991996, 0.7992026
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933997, 0.1933065
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718186, 0.2718797
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486697, 0.1487052

Time for backsubstitution: 6.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2531

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282394, upper bound: 0.0282584
time: 23.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282399, upper bound: 0.0282578
time: 15.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.6286242, 1.8224576, 1.6286242, 1.8224576, -0.0360933, 0.0360892
1: -1.5909975, -0.9892011, -1.5909975, -0.9892011, -0.1304021, 0.1304574
2: -0.1710566, 0.2527606, -0.1710566, 0.2527606, -0.2116679, 0.2116673
3: -3.9574311, -2.7509117, -3.9574311, -2.7509117, -0.6004109, 0.6003942
4: -4.2525058, -3.2459002, -4.2525058, -3.2459002, -0.3058555, 0.3058145
5: -4.5817947, -3.3524761, -4.5817947, -3.3524761, -0.6943259, 0.6942792
6: -5.3619261, -3.6091208, -5.3619261, -3.6091208, -0.7991986, 0.7992035
7: -6.2235036, -5.0609093, -6.2235036, -5.0609093, -0.1933997, 0.1933065
8: -0.8326688, -0.2983215, -0.8326688, -0.2983215, -0.2718152, 0.2718832
9: -2.3955522, -1.9256266, -2.3955522, -1.9256266, -0.1486643, 0.1487099

Time for backsubstitution: 6.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3456

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2531

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0282382, upper bound: 0.0282394
time: 42.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0282388, upper bound: 0.0282593
time: 3.85 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 53.12 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282395, upper bound: 0.0282586
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282401, upper bound: 0.0282560
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282384, upper bound: 0.0282594
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282389, upper bound: 0.0282587
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282399, upper bound: 0.0282569
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282404, upper bound: 0.0282572
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282387, upper bound: 0.0282585
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282393, upper bound: 0.0282582
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282388, upper bound: 0.0282583
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282394, upper bound: 0.0282584
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282377, upper bound: 0.0282606
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282383, upper bound: 0.0282594
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282392, upper bound: 0.0282578
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282398, upper bound: 0.0282578
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282380, upper bound: 0.0282575
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282386, upper bound: 0.0282588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282394, upper bound: 0.0282584
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282399, upper bound: 0.0282578
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282382, upper bound: 0.0282394
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 53.12
Output dim: 2, lower bound: -0.0282388, upper bound: 0.0282593
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282417, upper bound: 0.0282593
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282405, upper bound: 0.0282600
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282407, upper bound: 0.0282597
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282396, upper bound: 0.0282617
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282411, upper bound: 0.0282598
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282399, upper bound: 0.0282601
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282595, upper bound: 0.0282395
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282587, upper bound: 0.0282417
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282606, upper bound: 0.0282408
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282594, upper bound: 0.0282413
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282588, upper bound: 0.0282403
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282580, upper bound: 0.0282414
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282599, upper bound: 0.0282408
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282587, upper bound: 0.0282411
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282593, upper bound: 0.0282414
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282586, upper bound: 0.0282420
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282605, upper bound: 0.0282402
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282593, upper bound: 0.0282413
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282587, upper bound: 0.0282411
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282579, upper bound: 0.0282423
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282598, upper bound: 0.0282414
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 53.12
Output dim: 2, lower bound: -0.0282586, upper bound: 0.0282430

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 36.69 + 1798.95 = 1835.64 seconds
