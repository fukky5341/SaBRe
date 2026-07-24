## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.279429504


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142)
1: (-0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979)
2: (-0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389)
3: (-0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414)
4: (-0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357)
5: (-0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473)
6: (-0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181)
7: (-0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449)
8: (-0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810)
9: (0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.02 + 3.46 = 5.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.2910724, upper bound: 0.2910724

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 173

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2818634, upper bound: 0.2791797
time: 2.18 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2790444, upper bound: 0.2790444
time: 2.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.39 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.39
Output dim: 9, lower bound: -0.2818634, upper bound: 0.2791797
NS_A2, status: Status.VERIFIED, split count: 1, time: 4.39
Output dim: 9, lower bound: -0.2790444, upper bound: 0.2790444

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0113459, 0.0225548, -0.0142819, 0.0515324, -0.0628783, 0.0368367
1: -0.0583343, 0.0523046, -0.0885537, 0.0610442, -0.1193785, 0.1408582
2: -0.0197179, 0.0499577, -0.0313161, 0.0638228, -0.0835407, 0.0812738
3: -0.0200292, 0.0316662, -0.0428029, 0.0392385, -0.0592677, 0.0744691
4: -0.0332081, 0.0212415, -0.0428602, 0.0839756, -0.1171836, 0.0641017
5: -0.0292640, 0.0556123, -0.0430394, 0.0690079, -0.0982718, 0.0986518
6: -0.0129069, 0.0355330, -0.0223391, 0.0418790, -0.0547859, 0.0578721
7: -0.0429737, 0.0318591, -0.0666547, 0.0700902, -0.1130640, 0.0985138
8: -0.0574238, 0.0378170, -0.0758634, 0.0571175, -0.1145413, 0.1136804
9: 0.8726047, 1.1019974, 0.8029326, 1.1139959, -0.2413912, 0.2990648

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 173

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2790444, upper bound: 0.2790444
time: 2.03 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2790444, upper bound: 0.2790444
time: 1.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.04 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 6.04
Output dim: 9, lower bound: -0.2790444, upper bound: 0.2790444
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 6.04
Output dim: 9, lower bound: -0.2790444, upper bound: 0.2790444

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 5.48 + 10.43 = 15.91 seconds
