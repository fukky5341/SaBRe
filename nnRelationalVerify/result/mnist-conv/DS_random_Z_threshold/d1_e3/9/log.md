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
execution time: IAR + RelationalAnalysis = 23.68 + 33.41 = 57.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.1197558, upper bound: 0.1197559

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5859
type: DSZ, layer: 1, pos: 4617
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5859

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188656, upper bound: 0.1197551
time: 4.07 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197549, upper bound: 0.1188658
time: 4.85 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.93 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.93
Output dim: 0, lower bound: -0.1188656, upper bound: 0.1197551
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.93
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

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 4617

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188419, upper bound: 0.1190509
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1181587, upper bound: 0.1197311
time: 6.07 seconds

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

Time for backsubstitution: 22.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4617
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4617

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197524, upper bound: 0.1179385
time: 4.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188279, upper bound: 0.1188630
time: 4.68 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.55 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.55
Output dim: 0, lower bound: -0.1188419, upper bound: 0.1190509
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.55
Output dim: 0, lower bound: -0.1181587, upper bound: 0.1197311
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.55
Output dim: 0, lower bound: -0.1197524, upper bound: 0.1179385
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.55
Output dim: 0, lower bound: -0.1188279, upper bound: 0.1188630

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1775272, 0.1783432
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3736582, 0.3749888
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3095021, 0.3100746
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.2077712, 0.2075192
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2210541, 0.2185538
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1516373, 0.1516338
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1566080, 0.1575121
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3311892, 0.3270296
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2633220, 0.2647114
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2610762, 0.2589260

Time for backsubstitution: 22.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4617

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4617

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188394, upper bound: 0.1181236
time: 5.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1179149, upper bound: 0.1190484
time: 4.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1773335, 0.1785376
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3735127, 0.3751338
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3096092, 0.3099670
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.2077333, 0.2075570
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2218010, 0.2178085
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1515529, 0.1517175
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1563546, 0.1577665
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3320608, 0.3261619
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2635037, 0.2645302
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2614799, 0.2585243

Time for backsubstitution: 22.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4617

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4617

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1181562, upper bound: 0.1188044
time: 4.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1172317, upper bound: 0.1197288
time: 4.37 seconds

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

Time for backsubstitution: 22.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1197287, upper bound: 0.1172319
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1190482, upper bound: 0.1179151
time: 4.23 seconds

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

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188042, upper bound: 0.1181563
time: 4.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1181238, upper bound: 0.1188392
time: 5.49 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 32.81 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 0, lower bound: -0.1188394, upper bound: 0.1181236
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 0, lower bound: -0.1179149, upper bound: 0.1190484
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 0, lower bound: -0.1181562, upper bound: 0.1188044
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 0, lower bound: -0.1172317, upper bound: 0.1197288
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 0, lower bound: -0.1197287, upper bound: 0.1172319
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 0, lower bound: -0.1190482, upper bound: 0.1179151
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 0, lower bound: -0.1188042, upper bound: 0.1181563
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.81
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

Time for backsubstitution: 23.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2453
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 724

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 396

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1188390, upper bound: 0.1174708
time: 4.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1181863, upper bound: 0.1181236
time: 3.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 22.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 2453
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 396

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 724

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1151250, upper bound: 0.1150640
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1143234, upper bound: 0.1161360
time: 2.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 22.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 2453
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 724

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1153688, upper bound: 0.1148140
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1145698, upper bound: 0.1158895
time: 2.80 seconds

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

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 2453
type: DSZ, layer: 3, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 207

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1171238, upper bound: 0.1188582
time: 4.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1163610, upper bound: 0.1196209
time: 4.76 seconds

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

Time for backsubstitution: 21.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2453
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 1792

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 75

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1186560, upper bound: 0.1155979
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1180776, upper bound: 0.1161513
time: 2.93 seconds

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

Time for backsubstitution: 21.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 2453
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3105

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1170231, upper bound: 0.1159509
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1170840, upper bound: 0.1158900
time: 2.70 seconds

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

Time for backsubstitution: 21.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2453
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 396

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1725

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1176281, upper bound: 0.1175887
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1182365, upper bound: 0.1169802
time: 3.96 seconds

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

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 2453
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 3127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1725

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1169476, upper bound: 0.1182719
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1175561, upper bound: 0.1176634
time: 3.27 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 27.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1188390, upper bound: 0.1174708
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1181863, upper bound: 0.1181236
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1151250, upper bound: 0.1150640
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1143234, upper bound: 0.1161360
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1153688, upper bound: 0.1148140
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1145698, upper bound: 0.1158895
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1171238, upper bound: 0.1188582
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1163610, upper bound: 0.1196209
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1186560, upper bound: 0.1155979
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1180776, upper bound: 0.1161513
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1170231, upper bound: 0.1159509
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1170840, upper bound: 0.1158900
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1176281, upper bound: 0.1175887
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1182365, upper bound: 0.1169802
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1169476, upper bound: 0.1182719
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.73
Output dim: 0, lower bound: -0.1175561, upper bound: 0.1176634

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1729342, 0.1728314
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3685639, 0.3707466
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3091182, 0.3095717
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1986598, 0.1999117
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2179112, 0.2147433
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1460003, 0.1469437
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1564613, 0.1574049
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3310993, 0.3268780
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2580466, 0.2603807
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2610419, 0.2588525

Time for backsubstitution: 22.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 207
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2453
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2580

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1173807, upper bound: 0.1158251
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1171932, upper bound: 0.1160126
time: 2.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1729330, 0.1728327
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3685658, 0.3707445
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3091172, 0.3095729
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1986642, 0.1999072
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2179127, 0.2147419
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1460003, 0.1469436
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1564667, 0.1573994
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3311017, 0.3268756
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2580490, 0.2603786
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2610404, 0.2588538

Time for backsubstitution: 22.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 2453
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1412

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1178743, upper bound: 0.1179957
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1180584, upper bound: 0.1178115
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1680791, 0.1696016
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3604796, 0.3616710
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.2994971, 0.3020329
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1944573, 0.1934812
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2177246, 0.2144594
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1416118, 0.1399858
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1560289, 0.1574571
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3212562, 0.3149251
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2569009, 0.2563257
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2534902, 0.2494528

Time for backsubstitution: 23.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2453
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 228

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1153765, upper bound: 0.1173548
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1156205, upper bound: 0.1171108
time: 2.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.2021646, 8.8049355, 8.2021646, 8.8049355, -0.1674799, 0.1702008
1: -14.5419455, -13.6810780, -14.5419455, -13.6810780, -0.3608997, 0.3612506
2: -4.4596033, -3.7353108, -4.4596033, -3.7353108, -0.3015573, 0.2999728
3: -11.2466173, -10.5855494, -11.2466173, -10.5855494, -0.1951571, 0.1927813
4: -10.9946346, -10.2116566, -10.9946346, -10.2116566, -0.2177827, 0.2144012
5: -5.0402365, -4.4762449, -5.0402365, -4.4762449, -0.1407679, 0.1408296
6: -3.7309656, -3.1356418, -3.7309656, -3.1356418, -0.1560792, 0.1574068
7: -10.1710739, -9.2903175, -10.1710739, -9.2903175, -0.3207598, 0.3154212
8: -3.1486435, -2.5063982, -3.1486435, -2.5063982, -0.2562419, 0.2569847
9: -2.4534123, -1.7026299, -2.4534123, -1.7026299, -0.2523704, 0.2505727

Time for backsubstitution: 23.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 396
type: DSZ, layer: 3, pos: 3127
type: DSZ, layer: 3, pos: 1792
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 1156
type: DSZ, layer: 3, pos: 1412
type: DSZ, layer: 3, pos: 2580
type: DSZ, layer: 3, pos: 2139
type: DSZ, layer: 3, pos: 2453
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 724
type: DSZ, layer: 3, pos: 416

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 228

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1146137, upper bound: 0.1181176
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1148577, upper bound: 0.1178736
time: 3.33 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.23 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.23
Output dim: 0, lower bound: -0.1173807, upper bound: 0.1158251
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.23
Output dim: 0, lower bound: -0.1171932, upper bound: 0.1160126
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.23
Output dim: 0, lower bound: -0.1178743, upper bound: 0.1179957
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.23
Output dim: 0, lower bound: -0.1180584, upper bound: 0.1178115
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.23
Output dim: 0, lower bound: -0.1153765, upper bound: 0.1173548
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.23
Output dim: 0, lower bound: -0.1156205, upper bound: 0.1171108
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.23
Output dim: 0, lower bound: -0.1146137, upper bound: 0.1181176
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.23
Output dim: 0, lower bound: -0.1148577, upper bound: 0.1178736
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.23
Output dim: 0, lower bound: -0.1186560, upper bound: 0.1155979
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.23
Output dim: 0, lower bound: -0.1180776, upper bound: 0.1161513
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.23
Output dim: 0, lower bound: -0.1176281, upper bound: 0.1175887
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.23
Output dim: 0, lower bound: -0.1182365, upper bound: 0.1169802
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.23
Output dim: 0, lower bound: -0.1169476, upper bound: 0.1182719
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.23
Output dim: 0, lower bound: -0.1175561, upper bound: 0.1176634

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.09 + 547.50 = 604.59 seconds
