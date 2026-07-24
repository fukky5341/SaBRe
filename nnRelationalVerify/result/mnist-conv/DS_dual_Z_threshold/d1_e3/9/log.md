## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.117360782


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1806938, 0.1806939)
1: (-14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3769088, 0.3769090)
2: (-4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3111380, 0.3111379)
3: (-11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.2081676, 0.2081676)
4: (-10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2295593, 0.2295593)
5: (-5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1522999, 0.1522999)
6: (-3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1604722, 0.1604723)
7: (-10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3422725, 0.3422724)
8: (-3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2667599, 0.2667599)
9: (-2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2661802, 0.2661802)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.85 + 34.35 = 56.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1197558, upper bound: 0.1197559

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 4617
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 5859

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188656, upper bound: 0.1197551
time: 4.47 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197549, upper bound: 0.1188658
time: 5.01 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.70 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.70
Output dim: 0, lower bound: -0.1188656, upper bound: 0.1197551
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.70
Output dim: 0, lower bound: -0.1197549, upper bound: 0.1188658

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1787297, 0.1797402
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3740401, 0.3755159
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3102337, 0.3106985
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.2079657, 0.2077515
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2264953, 0.2232481
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1521422, 0.1522225
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1582217, 0.1593801
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3375294, 0.3324983
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2644153, 0.2656236
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2637726, 0.2612188

Time for backsubstitution: 20.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4617
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 4617

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188630, upper bound: 0.1188279
time: 4.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179386, upper bound: 0.1197525
time: 4.63 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1797401, 0.1787298
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3755159, 0.3740404
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3106986, 0.3102336
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.2077515, 0.2079657
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2232480, 0.2264953
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1522225, 0.1521422
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1593801, 0.1582216
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3324983, 0.3375297
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2656236, 0.2644153
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2612187, 0.2637726

Time for backsubstitution: 21.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4617
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 4617

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197524, upper bound: 0.1179385
time: 5.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188279, upper bound: 0.1188630
time: 4.87 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.49 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.49
Output dim: 0, lower bound: -0.1188630, upper bound: 0.1188279
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.49
Output dim: 0, lower bound: -0.1179386, upper bound: 0.1197525
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.49
Output dim: 0, lower bound: -0.1197524, upper bound: 0.1179385
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.49
Output dim: 0, lower bound: -0.1188279, upper bound: 0.1188630

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1741309, 0.1742238
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3689196, 0.3712451
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3098540, 0.3102012
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1989501, 0.2002354
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2233497, 0.2194334
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1465048, 0.1475318
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1580582, 0.1592508
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3374465, 0.3323513
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2591558, 0.2613065
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2637564, 0.2611645

Time for backsubstitution: 20.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188394, upper bound: 0.1181236
time: 4.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1181562, upper bound: 0.1188044
time: 4.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1732135, 0.1751413
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3697696, 0.3703954
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3097363, 0.3103189
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.2004496, 0.1987358
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2226807, 0.2201024
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1474516, 0.1465850
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1580923, 0.1592167
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3373826, 0.3324153
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2600983, 0.2603638
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2637185, 0.2612026

Time for backsubstitution: 21.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179149, upper bound: 0.1190484
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1172317, upper bound: 0.1197288
time: 4.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1751413, 0.1732135
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3703954, 0.3697696
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3103189, 0.3097363
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1987358, 0.2004496
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2201024, 0.2226806
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1465850, 0.1474516
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1592167, 0.1580923
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3324153, 0.3373826
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2603641, 0.2600982
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2612025, 0.2637185

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197287, upper bound: 0.1172319
time: 4.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1190482, upper bound: 0.1179151
time: 4.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1742239, 0.1741309
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3712451, 0.3689196
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3102012, 0.3098540
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.2002354, 0.1989502
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2194334, 0.2233496
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1475318, 0.1465048
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1592508, 0.1580583
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3323512, 0.3374467
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2613066, 0.2591558
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2611645, 0.2637564

Time for backsubstitution: 21.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188042, upper bound: 0.1181563
time: 4.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1181238, upper bound: 0.1188392
time: 5.39 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.70 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.1188394, upper bound: 0.1181236
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.1181562, upper bound: 0.1188044
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.1179149, upper bound: 0.1190484
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.1172317, upper bound: 0.1197288
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.1197287, upper bound: 0.1172319
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.1190482, upper bound: 0.1179151
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.1188042, upper bound: 0.1181563
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.70
Output dim: 0, lower bound: -0.1181238, upper bound: 0.1188392

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1729286, 0.1728271
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3685374, 0.3707180
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3091224, 0.3095770
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1987557, 0.2000033
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2179083, 0.2147389
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1459998, 0.1469431
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1564445, 0.1573827
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3311057, 0.3268821
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2580622, 0.2603943
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2610598, 0.2588718

Time for backsubstitution: 21.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2453

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 3105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1168142, upper bound: 0.1161595
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1168751, upper bound: 0.1160988
time: 3.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1727349, 0.1730215
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3683920, 0.3708630
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3092295, 0.3094695
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1987180, 0.2000411
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2186552, 0.2139937
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1459154, 0.1470268
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1561911, 0.1576371
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3319774, 0.3260143
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2582442, 0.2602131
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2614636, 0.2584702

Time for backsubstitution: 21.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2453

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 3, pos: 3105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1161310, upper bound: 0.1168402
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1161919, upper bound: 0.1167793
time: 3.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1720111, 0.1737446
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3693874, 0.3698680
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3090044, 0.3096950
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.2002553, 0.1985039
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2172393, 0.2154080
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1469465, 0.1459963
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1564786, 0.1573486
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3310416, 0.3269461
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2590050, 0.2594519
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2610221, 0.2589098

Time for backsubstitution: 21.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2453

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 3105

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1158898, upper bound: 0.1170842
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1159507, upper bound: 0.1170233
time: 2.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1718173, 0.1739390
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3692420, 0.3700130
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3091115, 0.3095872
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.2002175, 0.1985415
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2179861, 0.2146627
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1468622, 0.1460800
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1562252, 0.1576030
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3319132, 0.3260784
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2591869, 0.2592704
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2614255, 0.2585081

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2453

Time for candidate selection: 0.39 seconds

### Candidate
type: DSZ, layer: 3, pos: 3105

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1152066, upper bound: 0.1177647
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1152675, upper bound: 0.1177038
time: 3.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1739390, 0.1718173
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3700130, 0.3692420
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3095874, 0.3091116
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1985415, 0.2002175
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2146627, 0.2179861
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1460800, 0.1468622
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1576030, 0.1562252
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3260784, 0.3319132
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2592705, 0.2591867
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2585082, 0.2614256

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2453

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 3, pos: 3105

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1177035, upper bound: 0.1152678
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1177644, upper bound: 0.1152069
time: 2.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1737447, 0.1720111
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3698680, 0.3693874
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3096949, 0.3090045
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1985039, 0.2002553
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2154080, 0.2172393
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1459963, 0.1469466
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1573486, 0.1564786
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3269460, 0.3310416
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2594517, 0.2590048
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2589097, 0.2610220

Time for backsubstitution: 21.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2453

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 3105

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1170231, upper bound: 0.1159509
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1170840, upper bound: 0.1158900
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1730216, 0.1727349
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3708630, 0.3683920
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3094693, 0.3092294
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.2000411, 0.1987180
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2139937, 0.2186552
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1470268, 0.1459154
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1576371, 0.1561911
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3260143, 0.3319774
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2602130, 0.2582443
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2584701, 0.2614636

Time for backsubstitution: 21.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2453

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 3, pos: 3105

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1167791, upper bound: 0.1161922
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1168400, upper bound: 0.1161313
time: 2.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1728271, 0.1729286
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3707180, 0.3685374
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3095771, 0.3091223
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.2000033, 0.1987559
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2147390, 0.2179083
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1469431, 0.1459998
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1573827, 0.1564445
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3268821, 0.3311057
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2603945, 0.2580624
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2588718, 0.2610599

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2453

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 3105

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1160986, upper bound: 0.1168754
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1161595, upper bound: 0.1168145
time: 2.81 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 27.66 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1168142, upper bound: 0.1161595
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1168751, upper bound: 0.1160988
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1161310, upper bound: 0.1168402
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1161919, upper bound: 0.1167793
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1158898, upper bound: 0.1170842
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1159507, upper bound: 0.1170233
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1152066, upper bound: 0.1177647
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1152675, upper bound: 0.1177038
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1177035, upper bound: 0.1152678
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1177644, upper bound: 0.1152069
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1170231, upper bound: 0.1159509
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1170840, upper bound: 0.1158900
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1167791, upper bound: 0.1161922
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1168400, upper bound: 0.1161313
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1160986, upper bound: 0.1168754
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.1161595, upper bound: 0.1168145

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1579124, 0.1606533
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3514564, 0.3533132
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3028994, 0.3022392
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1693690, 0.1689030
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2079121, 0.2045984
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1422049, 0.1420411
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1501236, 0.1506151
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.2979567, 0.2942882
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2455119, 0.2446531
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2527212, 0.2497077

Time for backsubstitution: 21.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2453

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 724

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1124190, upper bound: 0.1137733
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1116194, upper bound: 0.1148495
time: 3.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1585316, 0.1600341
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3525422, 0.3522274
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3017635, 0.3033752
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1705789, 0.1676931
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2079219, 0.2045886
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1428233, 0.1414227
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1492373, 0.1515014
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3001230, 0.2921215
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2445694, 0.2455957
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2526253, 0.2498038

Time for backsubstitution: 21.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2453

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 724

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1124798, upper bound: 0.1137126
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1116801, upper bound: 0.1147887
time: 3.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1600341, 0.1585316
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3522274, 0.3525422
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3033752, 0.3017635
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1676931, 0.1705789
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2045886, 0.2079218
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1414227, 0.1428233
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1515014, 0.1492373
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.2921216, 0.3001231
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2455955, 0.2445694
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2498039, 0.2526252

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2453

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 724

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1147884, upper bound: 0.1116804
time: 3.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137124, upper bound: 0.1124801
time: 5.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1606533, 0.1579124
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3533132, 0.3514564
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3022392, 0.3028994
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1689030, 0.1693690
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2045984, 0.2079121
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1420411, 0.1422049
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1506151, 0.1501235
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.2942882, 0.2979566
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2446531, 0.2455119
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2497076, 0.2527213

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2453

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 724

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1148493, upper bound: 0.1116197
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1124193
time: 3.07 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.28 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.28
Output dim: 0, lower bound: -0.1124190, upper bound: 0.1137733
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.28
Output dim: 0, lower bound: -0.1116194, upper bound: 0.1148495
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.28
Output dim: 0, lower bound: -0.1124798, upper bound: 0.1137126
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.28
Output dim: 0, lower bound: -0.1116801, upper bound: 0.1147887
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.28
Output dim: 0, lower bound: -0.1147884, upper bound: 0.1116804
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.28
Output dim: 0, lower bound: -0.1137124, upper bound: 0.1124801
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.28
Output dim: 0, lower bound: -0.1148493, upper bound: 0.1116197
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.28
Output dim: 0, lower bound: -0.1137731, upper bound: 0.1124193

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 56.20 + 537.54 = 593.74 seconds
