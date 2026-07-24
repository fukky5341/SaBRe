## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.263905785


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1343257, 1.1343260)
1: (-6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.9096284, 0.9096284)
2: (-0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7806470, 0.7806470)
3: (-2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.6144085, 0.6144085)
4: (-9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.8248234, 0.8248234)
5: (-8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5613703, 0.5613702)
6: (-10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7603209, 0.7603209)
7: (3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6888497, 0.6888497)
8: (-4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6941956, 0.6941956)
9: (-3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9527485, 0.9527488)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.94 + 35.65 = 58.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.2665715, upper bound: 0.2665707

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 6135
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 540

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665711, upper bound: 0.2665718
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665711, upper bound: 0.2665718
time: 3.45 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.18 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.18
Output dim: 7, lower bound: -0.2665711, upper bound: 0.2665718
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.18
Output dim: 7, lower bound: -0.2665711, upper bound: 0.2665718

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1331403, 1.1292434
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.9094920, 0.9090486
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7796502, 0.7763760
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.6136392, 0.6111203
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.8233778, 0.8244867
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5611781, 0.5613251
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7593246, 0.7560635
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6886754, 0.6881058
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6941642, 0.6940614
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9525354, 0.9518449

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6135
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 6135

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643269, upper bound: 0.2665690
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665684, upper bound: 0.2643275
time: 3.69 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1292436, 1.1331406
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.9090486, 0.9094920
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7763760, 0.7796502
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.6111203, 0.6136394
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.8244867, 0.8233778
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5613251, 0.5611780
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7560635, 0.7593246
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6881058, 0.6886754
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6940614, 0.6941643
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9518449, 0.9525356

Time for backsubstitution: 21.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6135
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 6135

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643268, upper bound: 0.2665691
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665683, upper bound: 0.2643276
time: 3.72 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.51 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.51
Output dim: 7, lower bound: -0.2643269, upper bound: 0.2665690
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.51
Output dim: 7, lower bound: -0.2665684, upper bound: 0.2643275
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.51
Output dim: 7, lower bound: -0.2643268, upper bound: 0.2665691
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.51
Output dim: 7, lower bound: -0.2665683, upper bound: 0.2643276

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1382320, 1.1356268
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8781495, 0.8827167
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7749317, 0.7709403
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5931010, 0.5931432
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7900040, 0.7863679
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5300522, 0.5257638
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7424030, 0.7355433
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6744831, 0.6762645
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6940156, 0.6929789
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9488487, 0.9476373

Time for backsubstitution: 22.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2642478, upper bound: 0.2646341
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2623918, upper bound: 0.2664899
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1395237, 1.1343350
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8831601, 0.8777061
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7742145, 0.7716575
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5956624, 0.5905819
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7852590, 0.7911129
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5256168, 0.5301993
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7388043, 0.7391419
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6768343, 0.6739132
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6930817, 0.6939127
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9483280, 0.9481578

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664893, upper bound: 0.2623925
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2646332, upper bound: 0.2642484
time: 3.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1343348, 1.1395237
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8777061, 0.8831601
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7716575, 0.7742145
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5905819, 0.5956622
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7911129, 0.7852590
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5301993, 0.5256168
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7391419, 0.7388043
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6739135, 0.6768343
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6939129, 0.6930817
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9481578, 0.9483280

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2642477, upper bound: 0.2646342
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2623918, upper bound: 0.2664900
time: 3.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1356266, 1.1382320
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8827167, 0.8781495
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7709403, 0.7749317
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5931432, 0.5931009
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7863679, 0.7900040
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5257638, 0.5300522
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7355433, 0.7424030
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6762648, 0.6744831
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6929787, 0.6940156
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9476371, 0.9488485

Time for backsubstitution: 22.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664892, upper bound: 0.2623926
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2646332, upper bound: 0.2642465
time: 3.59 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.28 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.28
Output dim: 7, lower bound: -0.2642478, upper bound: 0.2646341
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.28
Output dim: 7, lower bound: -0.2623918, upper bound: 0.2664899
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.28
Output dim: 7, lower bound: -0.2664893, upper bound: 0.2623925
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.28
Output dim: 7, lower bound: -0.2646332, upper bound: 0.2642484
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.28
Output dim: 7, lower bound: -0.2642477, upper bound: 0.2646342
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.28
Output dim: 7, lower bound: -0.2623918, upper bound: 0.2664900
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.28
Output dim: 7, lower bound: -0.2664892, upper bound: 0.2623926
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.28
Output dim: 7, lower bound: -0.2646332, upper bound: 0.2642465

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1382365, 1.1356325
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8781509, 0.8827186
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7749355, 0.7709434
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5931020, 0.5931443
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7900014, 0.7863660
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5300496, 0.5257614
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7424057, 0.7355464
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6744838, 0.6762652
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6940126, 0.6929765
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9488478, 0.9476364

Time for backsubstitution: 21.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2642475, upper bound: 0.2635702
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2631839, upper bound: 0.2646339
time: 3.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1382375, 1.1356316
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8781514, 0.8827183
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7749348, 0.7709441
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5931020, 0.5931442
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7900023, 0.7863653
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5300498, 0.5257611
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7424061, 0.7355459
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6744835, 0.6762655
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6940131, 0.6929761
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9488478, 0.9476364

Time for backsubstitution: 22.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2623916, upper bound: 0.2654260
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2613281, upper bound: 0.2664896
time: 3.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1395288, 1.1343410
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8831615, 0.8777080
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7742183, 0.7716606
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5956631, 0.5905830
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7852564, 0.7911112
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5256140, 0.5301968
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7388070, 0.7391450
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6768351, 0.6739140
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6930790, 0.6939104
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9483271, 0.9481571

Time for backsubstitution: 22.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664890, upper bound: 0.2613287
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2654254, upper bound: 0.2623922
time: 3.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1395297, 1.1343398
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8831620, 0.8777077
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7742176, 0.7716613
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5956635, 0.5905827
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7852571, 0.7911103
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5256144, 0.5301965
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7388074, 0.7391448
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6768348, 0.6739142
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6930795, 0.6939100
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9483271, 0.9481568

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2646332, upper bound: 0.2631845
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2635696, upper bound: 0.2642481
time: 3.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1343398, 1.1395297
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8777080, 0.8831618
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7716613, 0.7742176
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5905828, 0.5956633
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7911103, 0.7852571
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5301965, 0.5256144
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7391446, 0.7388074
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6739142, 0.6768348
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6939101, 0.6930794
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9481568, 0.9483271

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2642475, upper bound: 0.2635703
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2631838, upper bound: 0.2646340
time: 3.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1343408, 1.1395285
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8777080, 0.8831615
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7716606, 0.7742183
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5905828, 0.5956632
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7911112, 0.7852564
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5301968, 0.5256141
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7391450, 0.7388070
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6739140, 0.6768351
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6939101, 0.6930790
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9481571, 0.9483271

Time for backsubstitution: 22.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2623915, upper bound: 0.2654261
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2613281, upper bound: 0.2664897
time: 3.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1356316, 1.1382380
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8827186, 0.8781512
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7709441, 0.7749348
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5931439, 0.5931020
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7863653, 0.7900023
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5257611, 0.5300498
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7355459, 0.7424061
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6762655, 0.6744835
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6929760, 0.6940132
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9476361, 0.9488478

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664889, upper bound: 0.2613288
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2654253, upper bound: 0.2623923
time: 3.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1356325, 1.1382370
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8827186, 0.8781509
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7709434, 0.7749355
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5931444, 0.5931019
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7863660, 0.7900014
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5257614, 0.5300494
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7355459, 0.7424057
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6762652, 0.6744838
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6929765, 0.6940129
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9476364, 0.9488475

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2646333, upper bound: 0.2631845
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2635695, upper bound: 0.2642482
time: 3.66 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.36 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2642475, upper bound: 0.2635702
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2631839, upper bound: 0.2646339
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2623916, upper bound: 0.2654260
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2613281, upper bound: 0.2664896
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2664890, upper bound: 0.2613287
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2654254, upper bound: 0.2623922
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2646332, upper bound: 0.2631845
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2635696, upper bound: 0.2642481
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2642475, upper bound: 0.2635703
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2631838, upper bound: 0.2646340
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2623915, upper bound: 0.2654261
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2613281, upper bound: 0.2664897
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2664889, upper bound: 0.2613288
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2654253, upper bound: 0.2623923
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2646333, upper bound: 0.2631845
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 7, lower bound: -0.2635695, upper bound: 0.2642482

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1383524, 1.1357634
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8780026, 0.8823040
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7744808, 0.7705481
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5930383, 0.5930719
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7903810, 0.7867012
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5305465, 0.5263246
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7350209, 0.7291076
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6745818, 0.6762557
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6927060, 0.6919981
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9463711, 0.9447159

Time for backsubstitution: 22.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 1116
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 681
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2593
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 606
type: DSZ, layer: 3, pos: 583
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1681
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2360
type: DSZ, layer: 3, pos: 2641

Time for candidate selection: 0.43 seconds

### Candidate
type: DSZ, layer: 3, pos: 1684

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.2582372, upper bound: 0.2623384
time: 4.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.2630326, upper bound: 0.2575419
time: 3.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1383677, 1.1357484
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8777366, 0.8825700
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7745399, 0.7704887
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5930295, 0.5930808
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7903366, 0.7867455
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5306128, 0.5262585
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7359664, 0.7281616
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6744740, 0.6763630
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6930345, 0.6916696
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9459271, 0.9451596

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 1116
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 681
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2593
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 606
type: DSZ, layer: 3, pos: 583
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1681
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2360
type: DSZ, layer: 3, pos: 2641

Time for candidate selection: 0.44 seconds

### Candidate
type: DSZ, layer: 3, pos: 1684

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.2571672, upper bound: 0.2634097
time: 3.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.2619618, upper bound: 0.2586117
time: 3.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1383533, 1.1357625
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8780026, 0.8823037
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7744801, 0.7705486
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5930383, 0.5930718
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7903817, 0.7867002
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5305469, 0.5263244
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7350214, 0.7291071
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6745815, 0.6762557
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6927065, 0.6919978
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9463711, 0.9447159

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 1116
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 681
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2593
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 606
type: DSZ, layer: 3, pos: 583
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1681
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2360
type: DSZ, layer: 3, pos: 2641

Time for candidate selection: 0.50 seconds

### Candidate
type: DSZ, layer: 3, pos: 1684

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2563822, upper bound: 0.2641901
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.2611804, upper bound: 0.2593965
time: 3.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1383686, 1.1357472
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8777366, 0.8825698
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7745395, 0.7704892
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5930297, 0.5930805
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7903373, 0.7867446
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5306131, 0.5262581
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7359664, 0.7281611
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6744740, 0.6763632
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6930350, 0.6916693
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9459271, 0.9451593

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 1116
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 681
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2593
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 606
type: DSZ, layer: 3, pos: 583
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1681
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2360
type: DSZ, layer: 3, pos: 2641

Time for candidate selection: 0.43 seconds

### Candidate
type: DSZ, layer: 3, pos: 1684

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2553133, upper bound: 0.2652614
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.2601096, upper bound: 0.2604663
time: 3.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1396441, 1.1344719
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8830132, 0.8772933
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7737637, 0.7712653
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5955997, 0.5905106
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7856357, 0.7914462
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5261111, 0.5307600
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7314222, 0.7327063
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6769331, 0.6739042
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6917719, 0.6929320
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9458504, 0.9452366

Time for backsubstitution: 22.05 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.59 + 560.27 = 618.86 seconds
