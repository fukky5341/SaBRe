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
execution time: IAR + LP analysis = 13.04 + 35.18 = 48.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.6198275, upper bound: 2.6198270


# Binary Search by BASE starts (time budget: 3551.78 seconds, max iter: 100)

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
Binary search time: 208.19 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_dual_Z) starts
Time budget: 3343.59 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3480212, upper bound: 2.3454544
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3454538, upper bound: 2.3480218
time: 4.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.44
Output dim: 8, lower bound: -2.3480212, upper bound: 2.3454544
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.44
Output dim: 8, lower bound: -2.3454538, upper bound: 2.3480218

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5522947, 4.5548625
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2320156, 4.2129798
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9335423, 3.9359217

Time for backsubstitution: 13.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6195

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3480057, upper bound: 2.3221761
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247530, upper bound: 2.3454387
time: 6.27 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5548620, 4.5522952
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2129803, 4.2320156
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9359217, 3.9335427

Time for backsubstitution: 13.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6195

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3454382, upper bound: 2.3247546
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221758, upper bound: 2.3480052
time: 7.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.53 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.53
Output dim: 8, lower bound: -2.3480057, upper bound: 2.3221761
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.53
Output dim: 8, lower bound: -2.3247530, upper bound: 2.3454387
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.53
Output dim: 8, lower bound: -2.3454382, upper bound: 2.3247546
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.53
Output dim: 8, lower bound: -2.3221758, upper bound: 2.3480052

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4822254, 4.5110502
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2074060, 4.1736212
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9375186, 3.9267626

Time for backsubstitution: 13.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3370395, upper bound: 2.3221551
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3479845, upper bound: 2.3113132
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5084829, 4.4847932
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1926565, 4.1883702
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9243846, 3.9398980

Time for backsubstitution: 12.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3138721, upper bound: 2.3454167
time: 6.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247320, upper bound: 2.3344705
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4847927, 4.5084829
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1883698, 4.1926570
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9398990, 3.9243841

Time for backsubstitution: 13.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3344699, upper bound: 2.3247313
time: 10.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3454170, upper bound: 2.3138716
time: 7.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5110502, 4.4822254
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1736212, 4.2074060
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9267631, 3.9375191

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113129, upper bound: 2.3479848
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221548, upper bound: 2.3370398
time: 6.13 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.62 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.62
Output dim: 8, lower bound: -2.3370395, upper bound: 2.3221551
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.62
Output dim: 8, lower bound: -2.3479845, upper bound: 2.3113132
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.62
Output dim: 8, lower bound: -2.3138721, upper bound: 2.3454167
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.62
Output dim: 8, lower bound: -2.3247320, upper bound: 2.3344705
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.62
Output dim: 8, lower bound: -2.3344699, upper bound: 2.3247313
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.62
Output dim: 8, lower bound: -2.3454170, upper bound: 2.3138716
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.62
Output dim: 8, lower bound: -2.3113129, upper bound: 2.3479848
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.62
Output dim: 8, lower bound: -2.3221548, upper bound: 2.3370398

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4743462, 4.4984426
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2047882, 4.1719694
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9435573, 3.9361811

Time for backsubstitution: 13.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3361498, upper bound: 2.3221520
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3370375, upper bound: 2.3215161
time: 6.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4696178, 4.5031705
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.2057543, 4.1710043
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9469371, 3.9328008

Time for backsubstitution: 12.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3471046, upper bound: 2.3113103
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3479825, upper bound: 2.3106829
time: 6.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5006027, 4.4721851
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1900396, 4.1867185
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9304214, 3.9493160

Time for backsubstitution: 12.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3131083, upper bound: 2.3454151
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3138700, upper bound: 2.3445644
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4958754, 4.4769130
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1910048, 4.1857529
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9338021, 3.9459362

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3239825, upper bound: 2.3344684
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247300, upper bound: 2.3336045
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4769135, 4.4958754
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1857529, 4.1910052
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9459357, 3.9338021

Time for backsubstitution: 12.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3336041, upper bound: 2.3247299
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3344680, upper bound: 2.3239827
time: 8.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4721851, 4.5006027
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1867180, 4.1900396
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9493165, 3.9304218

Time for backsubstitution: 12.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3445639, upper bound: 2.3138717
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3454151, upper bound: 2.3131076
time: 6.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.5031700, 4.4696178
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1710043, 4.2057538
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9328008, 3.9469376

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3106813, upper bound: 2.3479829
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113107, upper bound: 2.3471054
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4984426, 4.4743457
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1719694, 4.2047887
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9361815, 3.9435573

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3215163, upper bound: 2.3370381
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221527, upper bound: 2.3361507
time: 5.00 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3361498, upper bound: 2.3221520
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3370375, upper bound: 2.3215161
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3471046, upper bound: 2.3113103
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3479825, upper bound: 2.3106829
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3131083, upper bound: 2.3454151
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3138700, upper bound: 2.3445644
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3239825, upper bound: 2.3344684
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3247300, upper bound: 2.3336045
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3336041, upper bound: 2.3247299
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3344680, upper bound: 2.3239827
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3445639, upper bound: 2.3138717
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3454151, upper bound: 2.3131076
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3106813, upper bound: 2.3479829
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3113107, upper bound: 2.3471054
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3215163, upper bound: 2.3370381
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.86
Output dim: 8, lower bound: -2.3221527, upper bound: 2.3361507

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4582720, 4.4883757
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1681614, 4.1133380
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9457064, 3.9395456

Time for backsubstitution: 13.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3361432, upper bound: 2.3210384
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3350337, upper bound: 2.3221480
time: 5.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4642792, 4.4823685
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1461563, 4.1353426
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9469213, 3.9383307

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3370312, upper bound: 2.3203991
time: 6.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3359213, upper bound: 2.3215093
time: 7.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4535437, 4.4931030
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1691265, 4.1123724
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9490881, 3.9361658

Time for backsubstitution: 13.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3470981, upper bound: 2.3101960
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3459886, upper bound: 2.3113055
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4595518, 4.4870963
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1471214, 4.1343775
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9503031, 3.9349508

Time for backsubstitution: 13.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3479763, upper bound: 2.3095641
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3468664, upper bound: 2.3106740
time: 6.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4845295, 4.4621181
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1534128, 4.1280866
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9325724, 3.9519506

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3131019, upper bound: 2.3442990
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3119925, upper bound: 2.3454090
time: 4.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4905367, 4.4561110
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1314077, 4.1500916
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9337873, 3.9514661

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3138643, upper bound: 2.3434499
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3127545, upper bound: 2.3445581
time: 4.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4798012, 4.4668460
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1543789, 4.1271210
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9359522, 3.9493008

Time for backsubstitution: 13.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3239762, upper bound: 2.3333527
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3228667, upper bound: 2.3344624
time: 4.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4858084, 4.4608388
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1323738, 4.1491261
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9371672, 3.9480858

Time for backsubstitution: 13.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247242, upper bound: 2.3324886
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3236145, upper bound: 2.3335981
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4608393, 4.4858084
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1491261, 4.1323733
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9480867, 3.9371672

Time for backsubstitution: 13.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3335977, upper bound: 2.3236147
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3324882, upper bound: 2.3247245
time: 5.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4668465, 4.4798012
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1271210, 4.1543784
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9493017, 3.9359522

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3344619, upper bound: 2.3228659
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3333521, upper bound: 2.3239758
time: 7.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4561110, 4.4905357
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1500912, 4.1314082
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9514666, 3.9337869

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3445576, upper bound: 2.3127542
time: 8.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3434482, upper bound: 2.3138660
time: 9.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4621191, 4.4845285
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1280861, 4.1534128
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9519506, 3.9325719

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3454090, upper bound: 2.3119915
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3442992, upper bound: 2.3131012
time: 8.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4870968, 4.4595509
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1343775, 4.1471224
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9349508, 3.9503021

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3106750, upper bound: 2.3468668
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3095654, upper bound: 2.3479766
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4931040, 4.4535437
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1123724, 4.1691275
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9361658, 3.9490871

Time for backsubstitution: 13.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113052, upper bound: 2.3459892
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3101957, upper bound: 2.3470986
time: 5.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4823694, 4.4642787
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1353426, 4.1461568
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9383307, 3.9469218

Time for backsubstitution: 13.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3215100, upper bound: 2.3359218
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3204005, upper bound: 2.3370317
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.4883757, 4.4582715
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0690804, 4.0690804
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.8463669, 3.8463669
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -4.1133375, 4.1681619
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.3050923, 3.3050923
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.9395456, 3.9457073

Time for backsubstitution: 13.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221472, upper bound: 2.3350339
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3210377, upper bound: 2.3361439
time: 5.39 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3361432, upper bound: 2.3210384
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3350337, upper bound: 2.3221480
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3370312, upper bound: 2.3203991
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3359213, upper bound: 2.3215093
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3470981, upper bound: 2.3101960
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3459886, upper bound: 2.3113055
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3479763, upper bound: 2.3095641
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3468664, upper bound: 2.3106740
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3131019, upper bound: 2.3442990
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3119925, upper bound: 2.3454090
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3138643, upper bound: 2.3434499
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3127545, upper bound: 2.3445581
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3239762, upper bound: 2.3333527
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3228667, upper bound: 2.3344624
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3247242, upper bound: 2.3324886
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3236145, upper bound: 2.3335981
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3335977, upper bound: 2.3236147
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3324882, upper bound: 2.3247245
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3344619, upper bound: 2.3228659
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3333521, upper bound: 2.3239758
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3445576, upper bound: 2.3127542
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3434482, upper bound: 2.3138660
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3454090, upper bound: 2.3119915
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3442992, upper bound: 2.3131012
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3106750, upper bound: 2.3468668
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3095654, upper bound: 2.3479766
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3113052, upper bound: 2.3459892
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3101957, upper bound: 2.3470986
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3215100, upper bound: 2.3359218
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3204005, upper bound: 2.3370317
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3221472, upper bound: 2.3350339
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.24
Output dim: 8, lower bound: -2.3210377, upper bound: 2.3361439
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=3.3050923347473145
rel_dist={8: [-2.3480243628211177, 2.348024799945545]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1376654, upper bound: 2.1355132
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1355129, upper bound: 2.1376653
time: 8.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.09 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.09
Output dim: 8, lower bound: -2.1376654, upper bound: 2.1355132
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.09
Output dim: 8, lower bound: -2.1355129, upper bound: 2.1376653

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5921898, 4.5861592
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3530321, 4.3550296
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0511961, 4.0382948
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7235126, 3.7243328
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9922295, 3.9774246
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2320271, 3.2234859
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7588043, 3.7606549

Time for backsubstitution: 12.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6195

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1376510, upper bound: 2.1164223
time: 8.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186543, upper bound: 2.1354986
time: 6.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5861588, 4.5921898
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3550291, 4.3530326
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0382948, 4.0511961
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7243328, 3.7235126
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9774246, 3.9922299
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2234859, 3.2320271
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7606554, 3.7588048

Time for backsubstitution: 13.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6195

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1354980, upper bound: 2.1186545
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164219, upper bound: 2.1376510
time: 12.21 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 31.07 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 31.07
Output dim: 8, lower bound: -2.1376510, upper bound: 2.1164223
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 31.07
Output dim: 8, lower bound: -2.1186543, upper bound: 2.1354986
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 31.07
Output dim: 8, lower bound: -2.1354980, upper bound: 2.1186545
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 31.07
Output dim: 8, lower bound: -2.1164219, upper bound: 2.1376510

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6067209, 4.5955915
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2829628, 4.3053823
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0016975, 4.0039091
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7165480, 3.7145033
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9643431, 3.9380660
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2118893, 3.1950588
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7598624, 3.7514963

Time for backsubstitution: 12.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1296843, upper bound: 2.1164067
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1376358, upper bound: 2.1084467
time: 5.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6016226, 4.6006908
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3033857, 4.2849598
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0168095, 3.9887972
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7136831, 3.7173681
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9528713, 3.9495373
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2035999, 3.2033477
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7496467, 3.7617126

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1106856, upper bound: 2.1354825
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186392, upper bound: 2.1275290
time: 6.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6006918, 4.6016221
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2849598, 4.3033857
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9887972, 4.0168099
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7173681, 3.7136831
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9495373, 3.9528713
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2033482, 3.2036004
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7617126, 3.7496462

Time for backsubstitution: 13.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1275270, upper bound: 2.1186394
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1354828, upper bound: 2.1106855
time: 5.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5955915, 4.6067214
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.3053818, 4.2829633
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0039091, 4.0016975
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7145033, 3.7165480
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9380655, 3.9643431
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1950588, 3.2118893
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7514968, 3.7598624

Time for backsubstitution: 13.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1084462, upper bound: 2.1376358
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164067, upper bound: 2.1296847
time: 6.88 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 28.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 8, lower bound: -2.1296843, upper bound: 2.1164067
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 8, lower bound: -2.1376358, upper bound: 2.1084467
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 8, lower bound: -2.1106856, upper bound: 2.1354825
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 8, lower bound: -2.1186392, upper bound: 2.1275290
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 8, lower bound: -2.1275270, upper bound: 2.1186394
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 8, lower bound: -2.1354828, upper bound: 2.1106855
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 8, lower bound: -2.1084462, upper bound: 2.1376358
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.17
Output dim: 8, lower bound: -2.1164067, upper bound: 2.1296847

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.5980473
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2740326, 4.2927742
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0005026, 4.0022335
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7274828, 3.7301888
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9617252, 3.9361997
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1979256, 3.1851711
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7659001, 3.7601633

Time for backsubstitution: 12.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1293176, upper bound: 2.1164074
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1296833, upper bound: 2.1159524
time: 9.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6082554, 4.5991197
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2703552, 4.2964516
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0000219, 4.0027142
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7322330, 3.7254381
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9624758, 3.9354486
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2020006, 3.1810961
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7685294, 3.7575340

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1372705, upper bound: 2.1084451
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1376348, upper bound: 2.1079899
time: 9.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6051502, 4.6031466
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2944546, 4.2723522
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0156145, 3.9871216
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7246180, 3.7330532
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9502535, 3.9476709
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1896372, 3.1934595
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7556834, 3.7703795

Time for backsubstitution: 12.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1102511, upper bound: 2.1354820
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1106844, upper bound: 2.1349966
time: 14.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6040783, 4.6042185
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2907772, 4.2760296
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0151339, 3.9876022
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7293682, 3.7283030
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9510050, 3.9469204
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1937122, 3.1893849
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7583137, 3.7677507

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1182124, upper bound: 2.1275256
time: 11.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186384, upper bound: 2.1270409
time: 6.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6042175, 4.6040778
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2760296, 4.2907777
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9876022, 4.0151343
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7283030, 3.7293687
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9469194, 3.9510050
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1893845, 3.1937122
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7677503, 3.7583132

Time for backsubstitution: 13.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1270411, upper bound: 2.1186388
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1275260, upper bound: 2.1182127
time: 8.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.6031456, 4.6051497
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2723522, 4.2944551
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9871216, 4.0156150
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7330532, 3.7246180
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9476709, 3.9502544
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1934595, 3.1896372
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7703795, 3.7556839

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1349970, upper bound: 2.1106846
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1354818, upper bound: 2.1102512
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5991192, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2964516, 4.2703552
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0027142, 4.0000224
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7254381, 3.7322330
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9354486, 3.9624767
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1810961, 3.2020006
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7575345, 3.7685294

Time for backsubstitution: 13.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1079896, upper bound: 2.1376348
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1084450, upper bound: 2.1372699
time: 6.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5980473, 4.6082554
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2927742, 4.2740326
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0022335, 4.0005031
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7301884, 3.7274828
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9362001, 3.9617257
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1851711, 3.1979260
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7601638, 3.7659006

Time for backsubstitution: 13.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1159523, upper bound: 2.1296826
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164055, upper bound: 2.1293175
time: 5.22 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1293176, upper bound: 2.1164074
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1296833, upper bound: 2.1159524
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1372705, upper bound: 2.1084451
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1376348, upper bound: 2.1079899
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1102511, upper bound: 2.1354820
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1106844, upper bound: 2.1349966
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1182124, upper bound: 2.1275256
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1186384, upper bound: 2.1270409
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1270411, upper bound: 2.1186388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1275260, upper bound: 2.1182127
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1349970, upper bound: 2.1106846
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1354818, upper bound: 2.1102512
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1079896, upper bound: 2.1376348
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1084450, upper bound: 2.1372699
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1159523, upper bound: 2.1296826
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.05
Output dim: 8, lower bound: -2.1164055, upper bound: 2.1293175

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4497290
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5740051, 4.5723710
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2579584, 4.2813725
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9935198, 3.9972811
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7284117, 3.7315226
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9202089, 3.8775682
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1982317, 3.1848264
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7680502, 3.7632580

Time for backsubstitution: 13.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1293123, upper bound: 2.1153419
time: 8.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1282481, upper bound: 2.1164005
time: 6.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4550991, 4.4565840
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5845728, 4.5618038
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2626314, 4.2767000
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9955511, 3.9952497
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7288160, 3.7311172
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9030933, 3.8946829
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1975813, 3.1854768
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7689962, 3.7623129

Time for backsubstitution: 12.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1296781, upper bound: 2.1148893
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1286411, upper bound: 2.1159475
time: 5.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4486065
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5729332, 4.5734429
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2542810, 4.2850499
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9930391, 3.9977617
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7331619, 3.7267718
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9209595, 3.8768172
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2023067, 3.1807518
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7706785, 3.7606287

Time for backsubstitution: 12.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1372654, upper bound: 2.1073821
time: 10.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1362080, upper bound: 2.1084401
time: 5.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4562216, 4.4554615
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5835009, 4.5628757
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2589540, 4.2803774
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9950705, 3.9957304
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7335672, 3.7263670
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9038448, 3.8939323
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.2016563, 3.1814022
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7716246, 3.7596841

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1376297, upper bound: 2.1069266
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1365931, upper bound: 2.1079846
time: 5.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4515896, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5689068, 4.5774703
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2783813, 4.2609501
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0086317, 3.9821692
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7255468, 3.7343874
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9087372, 3.8890390
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1899433, 3.1931152
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7578344, 3.7734742

Time for backsubstitution: 13.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1102459, upper bound: 2.1344352
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1091878, upper bound: 2.1354770
time: 5.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4447346, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5794735, 4.5669031
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2830534, 4.2562780
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0106630, 3.9801378
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7259521, 3.7339821
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8916225, 3.9061542
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1892929, 3.1937656
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7587786, 3.7725296

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1106789, upper bound: 2.1339262
time: 9.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1096297, upper bound: 2.1349926
time: 6.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4527121, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5678349, 4.5785422
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2747040, 4.2646275
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0081511, 3.9826498
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7302980, 3.7296367
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9094887, 3.8882885
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1940184, 3.1890402
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7604628, 3.7708454

Time for backsubstitution: 12.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1182073, upper bound: 2.1264776
time: 22.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1171488, upper bound: 2.1275207
time: 8.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4458570, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5784016, 4.5679750
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2793760, 4.2599554
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -4.0101824, 3.9806185
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7307024, 3.7292314
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8923731, 3.9054031
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1933680, 3.1896906
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7614088, 3.7699003

Time for backsubstitution: 13.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186332, upper bound: 2.1259734
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1175896, upper bound: 2.1270357
time: 8.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4458570
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5679750, 4.5784016
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2599554, 4.2793756
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9806185, 4.0101819
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7292318, 3.7307024
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9054031, 3.8923736
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1896906, 3.1933675
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7699003, 3.7614079

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1270358, upper bound: 2.1175904
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1259715, upper bound: 2.1186337
time: 5.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4527121
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5785427, 4.5678344
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2646284, 4.2747035
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9826498, 4.0081506
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7296362, 3.7302971
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8882885, 3.9094887
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1890402, 3.1940184
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7708445, 3.7604628

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1275209, upper bound: 2.1171490
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1264779, upper bound: 2.1182067
time: 7.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4447346
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5669031, 4.5794735
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2562780, 4.2830529
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9801378, 4.0106626
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7339821, 3.7259517
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.9061546, 3.8916225
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1937656, 3.1892929
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7725286, 3.7587786

Time for backsubstitution: 13.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1349919, upper bound: 2.1096301
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1339271, upper bound: 2.1106792
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4567175, 4.4515896
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5774708, 4.5689063
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2609510, 4.2783809
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9821692, 4.0086312
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7343874, 3.7255468
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8890390, 3.9087377
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1931152, 3.1899433
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7734747, 3.7578340

Time for backsubstitution: 12.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1354768, upper bound: 2.1091875
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1344348, upper bound: 2.1102459
time: 5.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4554605, 4.4562225
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5628757, 4.5835009
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2803783, 4.2589531
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9957304, 3.9950700
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7263670, 3.7335672
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8939323, 3.9038448
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1814022, 3.2016563
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7596846, 3.7716241

Time for backsubstitution: 12.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1079844, upper bound: 2.1365933
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1069263, upper bound: 2.1376301
time: 5.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.4486055, 4.4567175
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.5734425, 4.5729337
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2850504, 4.2542810
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9977617, 3.9930387
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.7267723, 3.7331619
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8768167, 3.9209595
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1807508, 3.2023067
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.7606287, 3.7706795

Time for backsubstitution: 12.35 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=3.226428508758545
rel_dist={8: [-2.137668074409355, 2.1376683881402094]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0090856, upper bound: 2.0070978
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0070977, upper bound: 2.0090845
time: 9.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.49
Output dim: 8, lower bound: -2.0090856, upper bound: 2.0070978
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.49
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

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6195

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0090717, upper bound: 1.9903954
time: 7.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9923845, upper bound: 2.0070847
time: 5.82 seconds

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

Time for backsubstitution: 13.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6195
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6195

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0070842, upper bound: 1.9923846
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9903953, upper bound: 2.0090721
time: 5.95 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.50 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.50
Output dim: 8, lower bound: -2.0090717, upper bound: 1.9903954
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 24.50
Output dim: 8, lower bound: -1.9923845, upper bound: 2.0070847
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 24.50
Output dim: 8, lower bound: -2.0070842, upper bound: 1.9923846
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.50
Output dim: 8, lower bound: -1.9903953, upper bound: 2.0090721

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3811178, 4.3755531
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4865160, 4.4769759
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.1833324, 4.2025480
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9011936, 3.9030895
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6281052, 3.6263523
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8428106, 3.8202887
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1253719, 3.1109457
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6710334, 3.6638632

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0025073, upper bound: 1.9903829
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0090602, upper bound: 1.9838980
time: 5.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3755522, 4.3811178
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4769773, 4.4865155
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.2025480, 4.1833320
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9030895, 3.9011941
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6263523, 3.6281052
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8202887, 3.8428116
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1109457, 3.1253719
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6638637, 3.6710339

Time for backsubstitution: 12.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4555
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9838977, upper bound: 2.0090601
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9903829, upper bound: 2.0025078
time: 6.23 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 24.51
Output dim: 8, lower bound: -2.0025073, upper bound: 1.9903829
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.51
Output dim: 8, lower bound: -2.0090602, upper bound: 1.9838980
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.51
Output dim: 8, lower bound: -1.9838977, upper bound: 2.0090601
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 24.51
Output dim: 8, lower bound: -1.9903829, upper bound: 2.0025078

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3782272, 4.3717003
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4889717, 4.4803505
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.1707239, 4.1930923
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.8995190, 3.9018259
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6431122, 3.6372871
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8408375, 3.8176713
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1149015, 3.0969830
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6793256, 3.6699009

Time for backsubstitution: 13.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0086103, upper bound: 1.9838967
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0090586, upper bound: 1.9834690
time: 6.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3717003, 4.3782282
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4803505, 4.4889712
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.1930923, 4.1707239
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.9018259, 3.8995190
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6372871, 3.6431117
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.8176708, 3.8408380
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.0969830, 3.1149015
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6699004, 3.6793251

Time for backsubstitution: 13.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9834689, upper bound: 2.0090600
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9838962, upper bound: 2.0086100
time: 8.86 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 29.81 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 29.81
Output dim: 8, lower bound: -2.0086103, upper bound: 1.9838967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.81
Output dim: 8, lower bound: -2.0090586, upper bound: 1.9834690
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.81
Output dim: 8, lower bound: -1.9834689, upper bound: 2.0090600
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 29.81
Output dim: 8, lower bound: -1.9838962, upper bound: 2.0086100

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3547325, 4.3540812
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4617853, 4.4441071
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.1586552, 4.1770182
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.8942766, 3.8948421
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6443882, 3.6382160
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.7822056, 3.7737093
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1145563, 3.0971961
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6822844, 3.6720505

Time for backsubstitution: 13.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0090534, upper bound: 1.9824289
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0080169, upper bound: 1.9834637
time: 6.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3540812, 4.3547335
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4441071, 4.4617853
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.1770182, 4.1586547
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.8948421, 3.8942761
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6382160, 3.6443877
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.7737093, 3.7822061
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.0971956, 3.1145568
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6720514, 3.6822853

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9834633, upper bound: 2.0080172
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9824286, upper bound: 2.0090529
time: 5.94 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.17 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.17
Output dim: 8, lower bound: -2.0090534, upper bound: 1.9824289
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.17
Output dim: 8, lower bound: -2.0080169, upper bound: 1.9834637
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.17
Output dim: 8, lower bound: -1.9834633, upper bound: 2.0080172
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.17
Output dim: 8, lower bound: -1.9824286, upper bound: 2.0090529

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.2134104, -1.7566929, -6.2134104, -1.7566929, -4.3552332, 4.3516817
1: -15.3121986, -10.7039433, -15.3121986, -10.7039433, -4.4623117, 4.4415298
2: -9.1587076, -4.5603151, -9.1587076, -4.5603151, -4.1591291, 4.1746907
3: -7.6163192, -3.5472386, -7.6163192, -3.5472386, -3.8950472, 3.8910875
4: -12.2817459, -7.3541460, -12.2817459, -7.3541460, -4.9275999, 4.9275999
5: -6.0326176, -2.1862507, -6.0326176, -2.1862507, -3.6419764, 3.6387177
6: -13.8143063, -9.2480698, -13.8143063, -9.2480698, -3.7821770, 3.7737160
7: -10.2709742, -5.8640785, -10.2709742, -5.8640785, -4.4068956, 4.4068956
8: 7.8114066, 11.1164989, 7.8114066, 11.1164989, -3.1148500, 3.0957475
9: -7.1753616, -3.2234111, -7.1753616, -3.2234111, -3.6823211, 3.6718764

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 5761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 958

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0090300, upper bound: 1.9729407
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9995657, upper bound: 1.9824057
time: 7.78 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 26.09 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.09
Output dim: 8, lower bound: -2.0090300, upper bound: 1.9729407
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 26.09
Output dim: 8, lower bound: -1.9995657, upper bound: 1.9824057
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=3.14231538772583
rel_dist={8: [-2.00908813677729, 2.009087543266727]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 1911.08 seconds
