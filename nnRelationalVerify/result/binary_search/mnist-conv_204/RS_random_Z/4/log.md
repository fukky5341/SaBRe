## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 2.00905310091
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175)
1: (-15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554)
2: (-9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5983925, 4.5983925)
3: (-7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804)
4: (-12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999)
5: (-6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669)
6: (-13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.5662365, 4.5662365)
7: (-10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956)
8: (7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923)
9: (-7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9519506, 3.9519506)

## BASE Result
execution time: IAR + LP analysis = 13.21 + 34.91 = 48.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.6198275, upper bound: 2.6198270


# Binary Search by BASE starts (time budget: 3551.88 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.14231538772583
rel_dist={8: [-2.00908813677729, 2.009087543266727]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.8899760246276855
rel_dist={8: [-1.5721964040429413, 1.5721963328311173]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=2.9740891456604004
rel_dist={8: [-1.725812598105506, 1.7258119688107083]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=3.0582022666931152
rel_dist={8: [-1.8721584799452113, 1.8721582345714651]}

## Binary Search Result
Binary search time: 207.00 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_random_Z) starts
Time budget: 3344.88 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 930

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3471305, upper bound: 2.3457529
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3457529, upper bound: 2.3471308
time: 7.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.28 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.28
Output dim: 8, lower bound: -2.3471305, upper bound: 2.3457529
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.28
Output dim: 8, lower bound: -2.3457529, upper bound: 2.3471308

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5514193, 4.5514188
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2637634, 4.2637653
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9325333, 3.9325337

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 511

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 958

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3471063, upper bound: 2.3315521
time: 5.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3329422, upper bound: 2.3457294
time: 8.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5514183, 4.5514193
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2637653, 4.2637639
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9325342, 3.9325333

Time for backsubstitution: 12.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5761

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3450107, upper bound: 2.3471283
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3457510, upper bound: 2.3463900
time: 4.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.78
Output dim: 8, lower bound: -2.3471063, upper bound: 2.3315521
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.78
Output dim: 8, lower bound: -2.3329422, upper bound: 2.3457294
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.78
Output dim: 8, lower bound: -2.3450107, upper bound: 2.3471283
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.78
Output dim: 8, lower bound: -2.3457510, upper bound: 2.3463900

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5543661, 4.5554776
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2667999, 4.2656736
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9311762, 3.9316869

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 511

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 846

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3470819, upper bound: 2.3315135
time: 14.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3470819, upper bound: 2.3315330
time: 8.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5554781, 4.5543661
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2656717, 4.2668018
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9316864, 3.9311767

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 511

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 581

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3329392, upper bound: 2.3429121
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3301495, upper bound: 2.3457261
time: 8.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5440121, 4.5515375
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2594900, 4.2638264
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9325237, 3.9322796

Time for backsubstitution: 12.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3450106, upper bound: 2.3457902
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3437127, upper bound: 2.3471288
time: 5.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5514183, 4.5440125
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2637653, 4.2594891
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9322796, 3.9325333

Time for backsubstitution: 12.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 581

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3446726, upper bound: 2.3463769
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3457371, upper bound: 2.3453631
time: 4.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.89 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.89
Output dim: 8, lower bound: -2.3470819, upper bound: 2.3315135
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.89
Output dim: 8, lower bound: -2.3470819, upper bound: 2.3315330
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.89
Output dim: 8, lower bound: -2.3329392, upper bound: 2.3429121
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.89
Output dim: 8, lower bound: -2.3301495, upper bound: 2.3457261
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.89
Output dim: 8, lower bound: -2.3450106, upper bound: 2.3457902
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.89
Output dim: 8, lower bound: -2.3437127, upper bound: 2.3471288
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.89
Output dim: 8, lower bound: -2.3446726, upper bound: 2.3463769
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.89
Output dim: 8, lower bound: -2.3457371, upper bound: 2.3453631

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5377092, 4.5325894
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2601824, 4.2615361
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9296484, 3.9299183

Time for backsubstitution: 12.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 4555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 581

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3470787, upper bound: 2.3287206
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3442561, upper bound: 2.3315120
time: 5.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5314779, 4.5388207
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2626619, 4.2590566
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9294081, 3.9301596

Time for backsubstitution: 12.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 930

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5761

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3463418, upper bound: 2.3315320
time: 7.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3470800, upper bound: 2.3307873
time: 6.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5537624, 4.5559053
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2527266, 4.2587085
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9319019, 3.9313064

Time for backsubstitution: 12.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 511

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3329352, upper bound: 2.3405376
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3306165, upper bound: 2.3429073
time: 10.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5570173, 4.5526505
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2575808, 4.2538567
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9318161, 3.9313912

Time for backsubstitution: 12.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 846

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3301272, upper bound: 2.3457023
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3301207, upper bound: 2.3457022
time: 6.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5440092, 4.5515375
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2594910, 4.2638259
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9325247, 3.9322805

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 511

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3450065, upper bound: 2.3447437
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3426503, upper bound: 2.3447374
time: 13.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5440121, 4.5515351
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2594900, 4.2638264
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9325256, 3.9322801

Time for backsubstitution: 12.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 4555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3436913, upper bound: 2.3238654
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3204960, upper bound: 2.3471131
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5463057, 4.5402937
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2669067, 4.2614293
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9251428, 3.9280763

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3337306, upper bound: 2.3463552
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3446515, upper bound: 2.3353638
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5476999, 4.5388994
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2657070, 4.2626295
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9278226, 3.9253964

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 847

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3374761, upper bound: 2.3433063
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3434373, upper bound: 2.3433027
time: 5.16 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3470787, upper bound: 2.3287206
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3442561, upper bound: 2.3315120
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3463418, upper bound: 2.3315320
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3470800, upper bound: 2.3307873
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3329352, upper bound: 2.3405376
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3306165, upper bound: 2.3429073
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3301272, upper bound: 2.3457023
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3301207, upper bound: 2.3457022
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3450065, upper bound: 2.3447437
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3426503, upper bound: 2.3447374
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3436913, upper bound: 2.3238654
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3204960, upper bound: 2.3471131
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3337306, upper bound: 2.3463552
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3446515, upper bound: 2.3353638
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3374761, upper bound: 2.3433063
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 8, lower bound: -2.3434373, upper bound: 2.3433027

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5359936, 4.5341291
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2472372, 4.2534437
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9298630, 3.9300470

Time for backsubstitution: 13.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3470759, upper bound: 2.3275897
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3445755, upper bound: 2.3277091
time: 6.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5392494, 4.5308743
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2520895, 4.2485909
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9297771, 3.9301319

Time for backsubstitution: 12.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 6195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3434098, upper bound: 2.3315101
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3442544, upper bound: 2.3306687
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5240717, 4.5389395
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2583857, 4.2591181
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9293966, 3.9299049

Time for backsubstitution: 13.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 847

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3442850, upper bound: 2.3292325
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3442894, upper bound: 2.3226739
time: 5.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5314779, 4.5314140
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2626619, 4.2547808
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9291525, 3.9301596

Time for backsubstitution: 12.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 581

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3470769, upper bound: 2.3279954
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3442542, upper bound: 2.3307851
time: 4.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5501842, 4.5536690
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2572603, 4.2613392
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9123468, 3.9000297

Time for backsubstitution: 12.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3329351, upper bound: 2.3405288
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3306027, upper bound: 2.3405372
time: 16.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5515261, 4.5523272
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2553577, 4.2632418
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9006252, 3.9117522

Time for backsubstitution: 12.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 846

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3305920, upper bound: 2.3428828
time: 10.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3305920, upper bound: 2.3428836
time: 4.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5403614, 4.5297623
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2509623, 4.2497187
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9302883, 3.9296217

Time for backsubstitution: 12.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3293018, upper bound: 2.3457006
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3301255, upper bound: 2.3448535
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5341301, 4.5359936
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2534418, 4.2472391
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9300470, 3.9298630

Time for backsubstitution: 12.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3301147, upper bound: 2.3445799
time: 7.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3289977, upper bound: 2.3456963
time: 5.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5404301, 4.5493002
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2640266, 4.2664585
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9129710, 3.9010048

Time for backsubstitution: 12.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 581

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3441697, upper bound: 2.3424096
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3426726, upper bound: 2.3439223
time: 4.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5417719, 4.5479584
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2621231, 4.2683616
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9012485, 3.9127259

Time for backsubstitution: 12.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 581

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3426471, upper bound: 2.3418962
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3398222, upper bound: 2.3447348
time: 5.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4739418, 4.5077214
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2348795, 4.2244673
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9364991, 3.9231205

Time for backsubstitution: 12.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 847

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3416276, upper bound: 2.3215435
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3416320, upper bound: 2.3155208
time: 5.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5001984, 4.4814644
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2201309, 4.2392159
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9233651, 3.9362559

Time for backsubstitution: 13.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 846

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3196841, upper bound: 2.3447794
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3181616, upper bound: 2.3462745
time: 5.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5384302, 4.5276890
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2642899, 4.2597766
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9311838, 3.9374967

Time for backsubstitution: 12.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 581

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3337274, upper bound: 2.3435303
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3309041, upper bound: 2.3463518
time: 5.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5337019, 4.5324163
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2652550, 4.2588110
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9345636, 3.9341164

Time for backsubstitution: 12.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 846

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3446284, upper bound: 2.3353384
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3446274, upper bound: 2.3353388
time: 7.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5464725, 4.5404706
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2637186, 4.2651625
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9274049, 3.9259310

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 581

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3374697, upper bound: 2.3421873
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3363486, upper bound: 2.3433002
time: 6.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5476999, 4.5376711
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2657070, 4.2606416
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9278226, 3.9249792

Time for backsubstitution: 12.94 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=3.3050923347473145
rel_dist={8: [-2.3480243628211177, 2.348024799945545]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 5761

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 511

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1376661, upper bound: 2.1365722
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1365717, upper bound: 2.1376666
time: 4.89 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.26 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.26
Output dim: 8, lower bound: -2.1376661, upper bound: 2.1365722
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.26
Output dim: 8, lower bound: -2.1365717, upper bound: 2.1376666

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6019478, 4.5974832
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3485785, 4.3496218
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7059917, 3.7105579
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.0323238, 4.0308442
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2267628, 3.2270088
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7356358, 3.7265191

Time for backsubstitution: 12.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 930

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6195

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1376520, upper bound: 2.1175106
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186549, upper bound: 2.1365562
time: 8.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5974827, 4.6019473
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3496218, 4.3485785
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7105579, 3.7059917
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.0308437, 4.0323243
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2270088, 3.2267632
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7265186, 3.7356367

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 846

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 847

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1349693, upper bound: 2.1364542
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1353620, upper bound: 2.1360551
time: 9.43 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.54
Output dim: 8, lower bound: -2.1376520, upper bound: 2.1175106
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.54
Output dim: 8, lower bound: -2.1186549, upper bound: 2.1365562
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.54
Output dim: 8, lower bound: -2.1349693, upper bound: 2.1364542
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.54
Output dim: 8, lower bound: -2.1353620, upper bound: 2.1360551

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6069183
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2785091, 4.2999744
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0242510, 4.0419154
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6990252, 3.7007270
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.0044394, 3.9914870
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2066193, 3.1985760
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7367029, 3.7173634

Time for backsubstitution: 12.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1372866, upper bound: 2.1175094
time: 9.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1376510, upper bound: 2.1170934
time: 5.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2989311, 4.2795515
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0393629, 4.0268030
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6961613, 3.7035904
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9929667, 4.0029583
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1983299, 3.2068615
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7264805, 3.7275796

Time for backsubstitution: 12.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 4555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186548, upper bound: 2.1365497
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1175191, upper bound: 2.1365566
time: 5.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5968399, 4.6024475
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3483934, 4.3495278
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7111816, 3.7051830
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.0288553, 4.0338516
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2272797, 3.2264109
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7261009, 3.7359586

Time for backsubstitution: 12.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 6195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1345146, upper bound: 2.1347660
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1332830, upper bound: 2.1359959
time: 16.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5974827, 4.6013041
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3496218, 4.3473506
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7097492, 3.7059917
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.0308437, 4.0303354
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2266560, 3.2267632
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7265186, 3.7352180

Time for backsubstitution: 12.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1344024, upper bound: 2.1360424
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1353488, upper bound: 2.1350950
time: 5.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 8, lower bound: -2.1372866, upper bound: 2.1175094
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 8, lower bound: -2.1376510, upper bound: 2.1170934
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 8, lower bound: -2.1186548, upper bound: 2.1365497
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 8, lower bound: -2.1175191, upper bound: 2.1365566
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 8, lower bound: -2.1345146, upper bound: 2.1347660
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 8, lower bound: -2.1332830, upper bound: 2.1359959
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 8, lower bound: -2.1344024, upper bound: 2.1360424
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 8, lower bound: -2.1353488, upper bound: 2.1350950

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5802383, 4.5812411
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2624311, 4.2885699
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0172663, 4.0369616
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6999550, 3.7020617
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9629221, 3.9328551
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2069249, 3.1982317
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7388549, 3.7204604

Time for backsubstitution: 12.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 4555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 847

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1357752, upper bound: 2.1162883
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1361623, upper bound: 2.1158970
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5908060, 4.5706739
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2671041, 4.2838979
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0192976, 4.0349302
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7003603, 3.7016563
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9458075, 3.9499702
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2062745, 3.1988821
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7398000, 3.7195153

Time for backsubstitution: 12.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 4555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1366737, upper bound: 2.1150082
time: 7.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1356285, upper bound: 2.1161019
time: 7.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2989283, 4.2795506
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0393553, 4.0267982
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6961622, 3.7035904
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9929657, 4.0029569
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1983309, 3.2068624
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7264805, 3.7275791

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 958

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186316, upper bound: 2.1254434
time: 8.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1075359, upper bound: 2.1365279
time: 7.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2989302, 4.2795486
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0393581, 4.0267954
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6961613, 3.7035890
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9929657, 4.0029573
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1983309, 3.2068620
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7264805, 3.7275796

Time for backsubstitution: 12.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 846

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1175163, upper bound: 2.1343714
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1152663, upper bound: 2.1365536
time: 6.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5968418, 4.6024475
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3483934, 4.3495269
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7111826, 3.7051830
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.0288534, 4.0338507
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2272778, 3.2264099
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7261009, 3.7359571

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 581

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1335139, upper bound: 2.1327162
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1324620, upper bound: 2.1337710
time: 5.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5968399, 4.6024494
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3483934, 4.3495274
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7111816, 3.7051835
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.0288534, 4.0338497
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2272787, 3.2264090
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7260990, 3.7359581

Time for backsubstitution: 13.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 581

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1332674, upper bound: 2.1169616
time: 14.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1142097, upper bound: 2.1359798
time: 6.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5926476, 4.5944748
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3448191, 4.3436313
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7107363, 3.7072716
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.0339870, 4.0325446
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2274971, 3.2277803
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7193823, 3.7301660

Time for backsubstitution: 13.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5761

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1337858, upper bound: 2.1315133
time: 8.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1344003, upper bound: 2.1358070
time: 5.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5906563, 4.5964656
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3459044, 4.3425465
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7110300, 3.7069778
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.0330534, 4.0334783
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2276726, 3.2276039
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7214670, 3.7280827

Time for backsubstitution: 12.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1353486, upper bound: 2.1340008
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1353455, upper bound: 2.1350937
time: 8.98 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 27.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1357752, upper bound: 2.1162883
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1361623, upper bound: 2.1158970
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1366737, upper bound: 2.1150082
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1356285, upper bound: 2.1161019
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1186316, upper bound: 2.1254434
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1075359, upper bound: 2.1365279
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1175163, upper bound: 2.1343714
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1152663, upper bound: 2.1365536
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1335139, upper bound: 2.1327162
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1324620, upper bound: 2.1337710
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1332674, upper bound: 2.1169616
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1142097, upper bound: 2.1359798
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1337858, upper bound: 2.1315133
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1344003, upper bound: 2.1358070
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1353486, upper bound: 2.1340008
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 8, lower bound: -2.1353455, upper bound: 2.1350937

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5795946, 4.5817409
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2612038, 4.2895193
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0171404, 4.0370626
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7005816, 3.7012558
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9609327, 3.9343824
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2071962, 3.1978788
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7384372, 3.7207818

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1357725, upper bound: 2.1140421
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1335112, upper bound: 2.1162846
time: 8.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5802383, 4.5805974
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2624311, 4.2873421
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0172663, 4.0368366
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6991491, 3.7020617
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9629221, 3.9308662
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2065725, 3.1982317
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7388549, 3.7200418

Time for backsubstitution: 12.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 846

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1361572, upper bound: 2.1148393
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1350976, upper bound: 2.1158911
time: 7.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5908070, 4.5706754
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2671041, 4.2838974
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0192957, 4.0349274
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7003603, 3.7016559
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9458036, 3.9499674
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2062745, 3.1988835
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7398000, 3.7195158

Time for backsubstitution: 13.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 930

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 958

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1366510, upper bound: 2.1039268
time: 13.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1255304, upper bound: 2.1149846
time: 5.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5908070, 4.5706749
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2671041, 4.2838974
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0192938, 4.0349293
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7003593, 3.7016563
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9458046, 3.9499664
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2062764, 3.1988821
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7398000, 3.7195153

Time for backsubstitution: 12.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 581

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1345743, upper bound: 2.1138101
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1344745, upper bound: 2.1160989
time: 8.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3018751, 4.2833629
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0423875, 4.0286808
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7022133, 3.7082672
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9960032, 4.0051169
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1960969, 3.2037077
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7251225, 3.7266183

Time for backsubstitution: 12.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1181959, upper bound: 2.1237700
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1169516, upper bound: 2.1249983
time: 7.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3027391, 4.2824984
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0412374, 4.0298309
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7008390, 3.7096419
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9951267, 4.0059938
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1951756, 3.2046280
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7255192, 3.7262216

Time for backsubstitution: 12.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1075332, upper bound: 2.1343413
time: 9.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1053062, upper bound: 2.1365236
time: 5.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6075969
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2998056, 4.2824211
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0084362, 3.9829721
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6980162, 3.7062640
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9569826, 3.9521689
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2039280, 3.2039175
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7274904, 3.7304401

Time for backsubstitution: 12.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 581

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1165353, upper bound: 2.1333647
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1143621, upper bound: 2.1333570
time: 6.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6069622, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3018026, 4.2804241
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9955349, 3.9958730
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6988363, 3.7054439
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9421778, 3.9669747
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1953869, 3.2124591
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7293406, 3.7285900

Time for backsubstitution: 13.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 581

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 958

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1152426, upper bound: 2.1254468
time: 8.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1041743, upper bound: 2.1365319
time: 7.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5968428, 4.6024489
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3483934, 4.3495259
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7111816, 3.7051821
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.0288506, 4.0338492
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2272787, 3.2264113
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7261009, 3.7359571

Time for backsubstitution: 12.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 6195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1255187, upper bound: 2.1327011
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1334986, upper bound: 2.1247326
time: 5.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5968437, 4.6024485
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3483925, 4.3495264
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7111816, 3.7051830
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.0288515, 4.0338478
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2272787, 3.2264104
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7261009, 3.7359571

Time for backsubstitution: 12.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1321475, upper bound: 2.1337698
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1324610, upper bound: 2.1334873
time: 25.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2783241, 4.2998805
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0266762, 4.0394607
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7042160, 3.6953545
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.0009661, 3.9944906
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2071309, 3.1979752
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7271595, 3.7268019

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5761

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1330315, upper bound: 2.1169597
time: 6.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1287364, upper bound: 2.1163378
time: 6.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6062727, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2987461, 4.2794580
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0417881, 4.0243487
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7013521, 3.6982193
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9894943, 4.0059624
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1988444, 3.2062640
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7169437, 3.7370243

Time for backsubstitution: 13.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 581

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 846

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1141965, upper bound: 2.1355015
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1137592, upper bound: 2.1359666
time: 5.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5917664, 4.5903096
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3374109, 4.3420763
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7099428, 3.7033701
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.0297098, 4.0316434
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2259827, 3.2274599
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7193193, 3.7299123

Time for backsubstitution: 13.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 581

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1327623, upper bound: 2.1294692
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1317019, upper bound: 2.1305310
time: 7.21 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1357725, upper bound: 2.1140421
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1335112, upper bound: 2.1162846
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1361572, upper bound: 2.1148393
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1350976, upper bound: 2.1158911
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1366510, upper bound: 2.1039268
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1255304, upper bound: 2.1149846
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1345743, upper bound: 2.1138101
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1344745, upper bound: 2.1160989
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1181959, upper bound: 2.1237700
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1169516, upper bound: 2.1249983
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1075332, upper bound: 2.1343413
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1053062, upper bound: 2.1365236
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1165353, upper bound: 2.1333647
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1143621, upper bound: 2.1333570
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1152426, upper bound: 2.1254468
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1041743, upper bound: 2.1365319
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1255187, upper bound: 2.1327011
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1334986, upper bound: 2.1247326
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1321475, upper bound: 2.1337698
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1324610, upper bound: 2.1334873
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1330315, upper bound: 2.1169597
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1287364, upper bound: 2.1163378
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1141965, upper bound: 2.1355015
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1137592, upper bound: 2.1359666
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1327623, upper bound: 2.1294692
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.67
Output dim: 8, lower bound: -2.1317019, upper bound: 2.1305310
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.67
Output dim: 8, lower bound: -2.1344003, upper bound: 2.1358070
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.67
Output dim: 8, lower bound: -2.1353486, upper bound: 2.1340008
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.67
Output dim: 8, lower bound: -2.1353455, upper bound: 2.1350937
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=3.226428508758545
rel_dist={8: [-2.137668074409355, 2.1376683881402094]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0090856, upper bound: 2.0070978
time: 7.89 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0070977, upper bound: 2.0090845
time: 9.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.39 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.39
Output dim: 8, lower bound: -2.0090856, upper bound: 2.0070978
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.39
Output dim: 8, lower bound: -2.0070977, upper bound: 2.0090845

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3652906, 4.3686094
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4727125, 4.4675436
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2534018, 4.2551126
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9506922, 3.9396343
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6354790, 3.6361818
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8723373, 3.8596468
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1466942, 3.1393728
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6714363, 3.6730213

Time for backsubstitution: 13.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0090798, upper bound: 2.0060563
time: 7.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0080438, upper bound: 2.0070925
time: 5.83 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3686094, 4.3652911
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4675436, 4.4727125
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2551126, 4.2534018
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9396343, 3.9506922
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6361818, 3.6354790
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8596468, 3.8723373
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1393719, 3.1466942
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6730213, 3.6714358

Time for backsubstitution: 13.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5761

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0070844, upper bound: 2.0090856
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0070977, upper bound: 2.0090760
time: 9.02 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 28.07 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 28.07
Output dim: 8, lower bound: -2.0090798, upper bound: 2.0060563
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 28.07
Output dim: 8, lower bound: -2.0080438, upper bound: 2.0070925
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 28.07
Output dim: 8, lower bound: -2.0070844, upper bound: 2.0090856
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 28.07
Output dim: 8, lower bound: -2.0070977, upper bound: 2.0090760

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3657913, 4.3662105
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4732380, 4.4649653
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2538767, 4.2527857
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9514637, 3.9358807
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6330662, 3.6366825
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8723078, 3.8596535
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1469884, 3.1379261
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6714711, 3.6728477

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0086303, upper bound: 2.0060548
time: 28.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0090788, upper bound: 2.0056118
time: 5.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3636084, 4.3636794
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4661942, 4.4685473
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2477045, 4.2510099
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9384117, 3.9469271
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6349230, 3.6315765
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8553705, 3.8709531
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1378593, 3.1462040
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6729312, 3.6711826

Time for backsubstitution: 12.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 846

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0062864, upper bound: 2.0090731
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0070717, upper bound: 2.0083087
time: 31.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3669977, 4.3602896
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4633789, 4.4713626
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2527218, 4.2459927
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9358692, 3.9494691
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6322794, 3.6342201
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8582621, 3.8680615
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1388817, 3.1451817
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6727691, 3.6713457

Time for backsubstitution: 12.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0070842, upper bound: 1.9923766
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9903953, upper bound: 2.0090630
time: 5.87 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.73 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 23.73
Output dim: 8, lower bound: -2.0086303, upper bound: 2.0060548
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 8, lower bound: -2.0090788, upper bound: 2.0056118
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 8, lower bound: -2.0062864, upper bound: 2.0090731
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 23.73
Output dim: 8, lower bound: -2.0070717, upper bound: 2.0083087
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 23.73
Output dim: 8, lower bound: -2.0070842, upper bound: 1.9923766
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 8, lower bound: -1.9903953, upper bound: 2.0090630

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3422976, 4.3485923
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4460506, 4.4287214
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2418051, 4.2367086
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9462204, 3.9288955
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6343441, 3.6376133
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8136778, 3.8156929
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1466432, 3.1381373
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6744318, 3.6749983

Time for backsubstitution: 12.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 958

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0090780, upper bound: 2.0053000
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0087631, upper bound: 2.0056119
time: 24.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3624544, 4.3621416
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4610720, 4.4617195
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2430573, 4.2472916
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9436874, 3.9511318
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6359119, 3.6328163
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8585138, 3.8732958
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1387243, 3.1472192
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6657972, 3.6658340

Time for backsubstitution: 13.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9997694, upper bound: 2.0090608
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0062740, upper bound: 2.0025085
time: 6.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3739414, 4.3761177
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4728107, 4.4851651
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2001572, 4.1759233
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.8993244, 3.8999715
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6224508, 3.6268473
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8189039, 3.8385353
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1104546, 3.1238585
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6636095, 3.6709433

Time for backsubstitution: 13.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 930

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9895519, upper bound: 2.0090497
time: 9.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9903829, upper bound: 2.0082781
time: 5.91 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 28.87 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.87
Output dim: 8, lower bound: -2.0090780, upper bound: 2.0053000
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 28.87
Output dim: 8, lower bound: -2.0087631, upper bound: 2.0056119
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.87
Output dim: 8, lower bound: -1.9997694, upper bound: 2.0090608
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 28.87
Output dim: 8, lower bound: -2.0062740, upper bound: 2.0025085
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.87
Output dim: 8, lower bound: -1.9895519, upper bound: 2.0090497
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 28.87
Output dim: 8, lower bound: -1.9903829, upper bound: 2.0082781

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3422956, 4.3485899
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4460516, 4.4287219
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2418032, 4.2367086
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9462185, 3.9288955
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6343451, 3.6376133
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8136778, 3.8156924
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1466413, 3.1381359
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6744318, 3.6749988

Time for backsubstitution: 13.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 4555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0086711, upper bound: 2.0037734
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0075487, upper bound: 2.0049070
time: 5.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3585968, 4.3592463
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4644489, 4.4641776
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2336025, 4.2346859
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9424219, 3.9494543
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6468449, 3.6478224
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8558960, 3.8713217
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1247616, 3.1367497
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6718340, 3.6741247

Time for backsubstitution: 13.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 511

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9997638, upper bound: 2.0080195
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9987290, upper bound: 2.0090551
time: 7.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.16 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.16
Output dim: 8, lower bound: -2.0086711, upper bound: 2.0037734
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 26.16
Output dim: 8, lower bound: -2.0075487, upper bound: 2.0049070
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.16
Output dim: 8, lower bound: -1.9997638, upper bound: 2.0080195
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.16
Output dim: 8, lower bound: -1.9987290, upper bound: 2.0090551

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3561974, 4.3592463
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4618711, 4.4641776
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2312756, 4.2346859
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9386692, 3.9494543
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6468449, 3.6454086
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8558960, 3.8712912
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1233139, 3.1367497
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6716590, 3.6741247

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 581

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 958

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9987068, upper bound: 1.9995683
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9892279, upper bound: 2.0090330
time: 6.47 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 27.23 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 27.23
Output dim: 8, lower bound: -1.9987068, upper bound: 1.9995683
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 27.23
Output dim: 8, lower bound: -1.9892279, upper bound: 2.0090330
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=3.14231538772583
rel_dist={8: [-2.00908813677729, 2.009087543266727]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 2028.77 seconds
