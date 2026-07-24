## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.00067574


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577)
1: (-0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694)
2: (0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515)
3: (-0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447)
4: (0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.91 + 0.63 = 3.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0010395, upper bound: 0.0010396

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009976, upper bound: 0.0010062
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010060, upper bound: 0.0009978
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.80 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -0.0009976, upper bound: 0.0010062
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -0.0010060, upper bound: 0.0009978

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008129, upper bound: 0.0009249
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009204, upper bound: 0.0009258
time: 0.28 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009258, upper bound: 0.0009205
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009249, upper bound: 0.0008128
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.47 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -0.0008129, upper bound: 0.0009249
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -0.0009204, upper bound: 0.0009258
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -0.0009258, upper bound: 0.0009205
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -0.0009249, upper bound: 0.0008128

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006730, upper bound: 0.0006034
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006730, upper bound: 0.0006034
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006730, upper bound: 0.0006283
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006730, upper bound: 0.0006283
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006282, upper bound: 0.0006730
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006282, upper bound: 0.0006730
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006033, upper bound: 0.0006730
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006033, upper bound: 0.0006730
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.51 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.51
Output dim: 0, lower bound: -0.0006730, upper bound: 0.0006034
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.51
Output dim: 0, lower bound: -0.0006730, upper bound: 0.0006034
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.51
Output dim: 0, lower bound: -0.0006730, upper bound: 0.0006283
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.51
Output dim: 0, lower bound: -0.0006730, upper bound: 0.0006283
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.51
Output dim: 0, lower bound: -0.0006282, upper bound: 0.0006730
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.51
Output dim: 0, lower bound: -0.0006282, upper bound: 0.0006730
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.51
Output dim: 0, lower bound: -0.0006033, upper bound: 0.0006730
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.51
Output dim: 0, lower bound: -0.0006033, upper bound: 0.0006730

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.54 + 21.72 = 25.26 seconds
