## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.200980818


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (9.4635420, 10.3109207, 9.4635420, 10.3109207, -0.3757746, 0.3757749)
1: (-18.0093117, -16.7421856, -18.0093117, -16.7421856, -0.4540387, 0.4540386)
2: (-3.0654144, -2.1132352, -3.0654144, -2.1132352, -0.3724043, 0.3724043)
3: (-10.2060471, -9.0677299, -10.2060471, -9.0677299, -0.4265924, 0.4265924)
4: (-21.1053066, -19.9463177, -21.1053066, -19.9463177, -0.3780588, 0.3780588)
5: (-0.6191039, 0.2850968, -0.6191039, 0.2850968, -0.3018680, 0.3018680)
6: (-5.3789139, -4.5177712, -5.3789139, -4.5177712, -0.2712505, 0.2712505)
7: (-4.0503621, -3.0631983, -4.0503621, -3.0631983, -0.3652309, 0.3652310)
8: (1.1103096, 1.8106346, 1.1103096, 1.8106346, -0.3119166, 0.3119164)
9: (-7.7042732, -6.6410213, -7.7042732, -6.6410213, -0.3552496, 0.3552498)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.89 + 34.57 = 57.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.2011819, upper bound: 0.2011820

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 6158
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 5821
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 5834
type: DSZ, layer: 1, pos: 871

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2011811, upper bound: 0.1999706
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.1999706, upper bound: 0.2011811
time: 5.55 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.89 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.89
Output dim: 0, lower bound: -0.2011811, upper bound: 0.1999706
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.89
Output dim: 0, lower bound: -0.1999706, upper bound: 0.2011811

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 9.4635420, 10.3109207, 9.4635420, 10.3109207, -0.3713584, 0.3704777
1: -18.0093117, -16.7421856, -18.0093117, -16.7421856, -0.4545951, 0.4538059
2: -3.0654144, -2.1132352, -3.0654144, -2.1132352, -0.3708603, 0.3705528
3: -10.2060471, -9.0677299, -10.2060471, -9.0677299, -0.4159646, 0.4177306
4: -21.1053066, -19.9463177, -21.1053066, -19.9463177, -0.3668728, 0.3645201
5: -0.6191039, 0.2850968, -0.6191039, 0.2850968, -0.2898775, 0.2918746
6: -5.3789139, -4.5177712, -5.3789139, -4.5177712, -0.2679520, 0.2670659
7: -4.0503621, -3.0631983, -4.0503621, -3.0631983, -0.3561372, 0.3579102
8: 1.1103096, 1.8106346, 1.1103096, 1.8106346, -0.3046198, 0.3031672
9: -7.7042732, -6.6410213, -7.7042732, -6.6410213, -0.3541136, 0.3525964

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 6158
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 5834
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 5821

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2009513, upper bound: 0.1997507
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2009612, upper bound: 0.1997407
time: 6.03 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 9.4635420, 10.3109207, 9.4635420, 10.3109207, -0.3704774, 0.3713584
1: -18.0093117, -16.7421856, -18.0093117, -16.7421856, -0.4538057, 0.4545951
2: -3.0654144, -2.1132352, -3.0654144, -2.1132352, -0.3705528, 0.3708603
3: -10.2060471, -9.0677299, -10.2060471, -9.0677299, -0.4177308, 0.4159644
4: -21.1053066, -19.9463177, -21.1053066, -19.9463177, -0.3645201, 0.3668729
5: -0.6191039, 0.2850968, -0.6191039, 0.2850968, -0.2918748, 0.2898777
6: -5.3789139, -4.5177712, -5.3789139, -4.5177712, -0.2670659, 0.2679518
7: -4.0503621, -3.0631983, -4.0503621, -3.0631983, -0.3579103, 0.3561373
8: 1.1103096, 1.8106346, 1.1103096, 1.8106346, -0.3031673, 0.3046197
9: -7.7042732, -6.6410213, -7.7042732, -6.6410213, -0.3525963, 0.3541137

Time for backsubstitution: 22.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 5821
type: DSZ, layer: 1, pos: 6158
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 5834

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1997408, upper bound: 0.2009613
time: 4.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.1997507, upper bound: 0.2009513
time: 35.16 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 62.47 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 62.47
Output dim: 0, lower bound: -0.2009513, upper bound: 0.1997507
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 62.47
Output dim: 0, lower bound: -0.2009612, upper bound: 0.1997407
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 62.47
Output dim: 0, lower bound: -0.1997408, upper bound: 0.2009613
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 62.47
Output dim: 0, lower bound: -0.1997507, upper bound: 0.2009513

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 57.46 + 103.90 = 161.36 seconds
