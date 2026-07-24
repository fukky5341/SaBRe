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
execution time: IAR + RelationalAnalysis = 1.02 + 0.48 = 1.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0010395, upper bound: 0.0010396

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009491, upper bound: 0.0010378
time: 0.13 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010376, upper bound: 0.0009491
time: 0.13 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.28 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.28
Output dim: 0, lower bound: -0.0009491, upper bound: 0.0010378
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.28
Output dim: 0, lower bound: -0.0010376, upper bound: 0.0009491

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008970, upper bound: 0.0009612
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009038, upper bound: 0.0009614
time: 0.13 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009985, upper bound: 0.0009255
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009843, upper bound: 0.0009263
time: 0.13 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.22 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.22
Output dim: 0, lower bound: -0.0008970, upper bound: 0.0009612
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.22
Output dim: 0, lower bound: -0.0009038, upper bound: 0.0009614
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.22
Output dim: 0, lower bound: -0.0009985, upper bound: 0.0009255
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.22
Output dim: 0, lower bound: -0.0009843, upper bound: 0.0009263

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008050, upper bound: 0.0009233
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008257, upper bound: 0.0009186
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008724, upper bound: 0.0008744
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008812, upper bound: 0.0009428
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009916, upper bound: 0.0008649
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008481, upper bound: 0.0009199
time: 0.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009774, upper bound: 0.0008129
time: 0.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008721, upper bound: 0.0009204
time: 0.13 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.22 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0008050, upper bound: 0.0009233
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0008257, upper bound: 0.0009186
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0008724, upper bound: 0.0008744
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0008812, upper bound: 0.0009428
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0009916, upper bound: 0.0008649
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0008481, upper bound: 0.0009199
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0009774, upper bound: 0.0008129
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0008721, upper bound: 0.0009204

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006633, upper bound: 0.0005638
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006633, upper bound: 0.0005749
time: 0.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008195, upper bound: 0.0007695
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007028, upper bound: 0.0008909
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008668, upper bound: 0.0007353
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007602, upper bound: 0.0008685
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008773, upper bound: 0.0007613
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008452, upper bound: 0.0009358
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009196, upper bound: 0.0007404
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009226, upper bound: 0.0007465
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007745, upper bound: 0.0008547
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007464, upper bound: 0.0008540
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009084, upper bound: 0.0007427
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008566, upper bound: 0.0007465
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007352, upper bound: 0.0008669
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008551, upper bound: 0.0008686
time: 0.14 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.47 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0006633, upper bound: 0.0005638
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0006633, upper bound: 0.0005749
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0008195, upper bound: 0.0007695
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0007028, upper bound: 0.0008909
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0008668, upper bound: 0.0007353
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0007602, upper bound: 0.0008685
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0008773, upper bound: 0.0007613
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0008452, upper bound: 0.0009358
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0009196, upper bound: 0.0007404
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0009226, upper bound: 0.0007465
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0007745, upper bound: 0.0008547
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0007464, upper bound: 0.0008540
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0009084, upper bound: 0.0007427
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0008566, upper bound: 0.0007465
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0007352, upper bound: 0.0008669
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.47
Output dim: 0, lower bound: -0.0008551, upper bound: 0.0008686

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006020, upper bound: 0.0006261
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006020, upper bound: 0.0006376
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005833, upper bound: 0.0006177
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005861, upper bound: 0.0006406
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007694, upper bound: 0.0005963
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007917, upper bound: 0.0006551
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006802, upper bound: 0.0007936
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006420, upper bound: 0.0007701
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007930, upper bound: 0.0006853
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007934, upper bound: 0.0006844
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006982, upper bound: 0.0008650
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006298, upper bound: 0.0007678
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007678, upper bound: 0.0006299
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008621, upper bound: 0.0006732
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008650, upper bound: 0.0006981
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008645, upper bound: 0.0006764
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006844, upper bound: 0.0007934
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007140, upper bound: 0.0007935
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006851, upper bound: 0.0007930
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006859, upper bound: 0.0007754
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007700, upper bound: 0.0006419
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008509, upper bound: 0.0006764
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007935, upper bound: 0.0006802
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007956, upper bound: 0.0006778
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006551, upper bound: 0.0007917
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005963, upper bound: 0.0007696
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007197, upper bound: 0.0007955
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005595, upper bound: 0.0007536
time: 0.15 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.30 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0006020, upper bound: 0.0006261
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0006020, upper bound: 0.0006376
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0005833, upper bound: 0.0006177
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0005861, upper bound: 0.0006406
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0007694, upper bound: 0.0005963
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0007917, upper bound: 0.0006551
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0006802, upper bound: 0.0007936
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0006420, upper bound: 0.0007701
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0007930, upper bound: 0.0006853
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0007934, upper bound: 0.0006844
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0006982, upper bound: 0.0008650
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0006298, upper bound: 0.0007678
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0007678, upper bound: 0.0006299
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0008621, upper bound: 0.0006732
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0008650, upper bound: 0.0006981
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0008645, upper bound: 0.0006764
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0006844, upper bound: 0.0007934
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0007140, upper bound: 0.0007935
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0006851, upper bound: 0.0007930
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0006859, upper bound: 0.0007754
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0007700, upper bound: 0.0006419
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0008509, upper bound: 0.0006764
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0007935, upper bound: 0.0006802
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0007956, upper bound: 0.0006778
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0006551, upper bound: 0.0007917
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0005963, upper bound: 0.0007696
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0007197, upper bound: 0.0007955
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.30
Output dim: 0, lower bound: -0.0005595, upper bound: 0.0007536

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007317, upper bound: 0.0005480
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006136, upper bound: 0.0005729
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005644, upper bound: 0.0005892
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007346, upper bound: 0.0005790
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006554, upper bound: 0.0005571
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006198, upper bound: 0.0007217
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006190, upper bound: 0.0005688
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006096, upper bound: 0.0007230
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006845, upper bound: 0.0006408
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007360, upper bound: 0.0005784
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006689, upper bound: 0.0006395
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007363, upper bound: 0.0005815
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006263, upper bound: 0.0008156
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006404, upper bound: 0.0006618
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006080, upper bound: 0.0006387
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006003, upper bound: 0.0007254
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007253, upper bound: 0.0006004
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006386, upper bound: 0.0006080
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006468, upper bound: 0.0006099
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008151, upper bound: 0.0005935
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007861, upper bound: 0.0006658
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006718, upper bound: 0.0006684
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006250, upper bound: 0.0006148
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008154, upper bound: 0.0005947
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005816, upper bound: 0.0007364
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006395, upper bound: 0.0006690
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006831, upper bound: 0.0005990
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006831, upper bound: 0.0007296
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005784, upper bound: 0.0007361
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006407, upper bound: 0.0006846
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005175, upper bound: 0.0007100
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006414, upper bound: 0.0006833
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007229, upper bound: 0.0006096
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005688, upper bound: 0.0006191
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006751, upper bound: 0.0006177
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008133, upper bound: 0.0005922
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006632, upper bound: 0.0006211
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007364, upper bound: 0.0005829
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006423, upper bound: 0.0006189
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007398, upper bound: 0.0005879
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006303, upper bound: 0.0006091
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005808, upper bound: 0.0007315
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005195, upper bound: 0.0007125
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005052, upper bound: 0.0005931
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006890, upper bound: 0.0006086
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006895, upper bound: 0.0007312
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004842, upper bound: 0.0006947
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004889, upper bound: 0.0006184
time: 0.16 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.62 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0007317, upper bound: 0.0005480
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006136, upper bound: 0.0005729
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0005644, upper bound: 0.0005892
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0007346, upper bound: 0.0005790
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006554, upper bound: 0.0005571
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006198, upper bound: 0.0007217
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006190, upper bound: 0.0005688
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006096, upper bound: 0.0007230
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006845, upper bound: 0.0006408
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0007360, upper bound: 0.0005784
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006689, upper bound: 0.0006395
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0007363, upper bound: 0.0005815
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006263, upper bound: 0.0008156
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006404, upper bound: 0.0006618
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006080, upper bound: 0.0006387
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006003, upper bound: 0.0007254
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0007253, upper bound: 0.0006004
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006386, upper bound: 0.0006080
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006468, upper bound: 0.0006099
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0008151, upper bound: 0.0005935
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0007861, upper bound: 0.0006658
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006718, upper bound: 0.0006684
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006250, upper bound: 0.0006148
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0008154, upper bound: 0.0005947
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0005816, upper bound: 0.0007364
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006395, upper bound: 0.0006690
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006831, upper bound: 0.0005990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006831, upper bound: 0.0007296
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0005784, upper bound: 0.0007361
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006407, upper bound: 0.0006846
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0005175, upper bound: 0.0007100
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006414, upper bound: 0.0006833
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0007229, upper bound: 0.0006096
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0005688, upper bound: 0.0006191
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006751, upper bound: 0.0006177
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0008133, upper bound: 0.0005922
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006632, upper bound: 0.0006211
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0007364, upper bound: 0.0005829
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006423, upper bound: 0.0006189
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0007398, upper bound: 0.0005879
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006303, upper bound: 0.0006091
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0005808, upper bound: 0.0007315
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0005195, upper bound: 0.0007125
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0005052, upper bound: 0.0005931
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006890, upper bound: 0.0006086
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0006895, upper bound: 0.0007312
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0004842, upper bound: 0.0006947
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.62
Output dim: 0, lower bound: -0.0004889, upper bound: 0.0006184

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005742, upper bound: 0.0004764
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006818, upper bound: 0.0004655
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006816, upper bound: 0.0004921
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005556, upper bound: 0.0005606
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005032, upper bound: 0.0006718
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005652, upper bound: 0.0006116
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004819, upper bound: 0.0006721
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005560, upper bound: 0.0006134
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006442, upper bound: 0.0006055
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005512, upper bound: 0.0006148
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006915, upper bound: 0.0004917
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005614
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006802, upper bound: 0.0004972
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005434, upper bound: 0.0005643
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005989, upper bound: 0.0006033
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005802, upper bound: 0.0007308
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004899, upper bound: 0.0006731
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005441, upper bound: 0.0006088
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006087, upper bound: 0.0005441
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006730, upper bound: 0.0004899
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007325, upper bound: 0.0005082
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006387, upper bound: 0.0005782
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006104, upper bound: 0.0006090
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007308, upper bound: 0.0005803
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007311, upper bound: 0.0005147
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006042, upper bound: 0.0005794
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005643, upper bound: 0.0005434
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004972, upper bound: 0.0006803
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005650, upper bound: 0.0005412
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006386, upper bound: 0.0004757
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005012, upper bound: 0.0006794
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006386, upper bound: 0.0006267
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005613, upper bound: 0.0006070
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004918, upper bound: 0.0006915
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006147, upper bound: 0.0005512
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006055, upper bound: 0.0006442
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005016, upper bound: 0.0005285
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004747, upper bound: 0.0006744
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006154, upper bound: 0.0004757
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006055, upper bound: 0.0006280
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006134, upper bound: 0.0005561
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006721, upper bound: 0.0004819
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007326, upper bound: 0.0005017
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006426, upper bound: 0.0005770
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006717, upper bound: 0.0005033
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004895, upper bound: 0.0005668
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006774, upper bound: 0.0005032
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005153, upper bound: 0.0005730
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004920, upper bound: 0.0006817
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005174, upper bound: 0.0005369
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005599
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004654, upper bound: 0.0006817
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005987, upper bound: 0.0005549
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006389, upper bound: 0.0004657
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005811, upper bound: 0.0006815
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006389, upper bound: 0.0006181
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004654, upper bound: 0.0005392
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004654, upper bound: 0.0006730
time: 0.16 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.43 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005742, upper bound: 0.0004764
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006818, upper bound: 0.0004655
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006816, upper bound: 0.0004921
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005556, upper bound: 0.0005606
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005032, upper bound: 0.0006718
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005652, upper bound: 0.0006116
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0004819, upper bound: 0.0006721
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005560, upper bound: 0.0006134
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006442, upper bound: 0.0006055
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005512, upper bound: 0.0006148
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006915, upper bound: 0.0004917
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006069, upper bound: 0.0005614
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006802, upper bound: 0.0004972
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005434, upper bound: 0.0005643
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005989, upper bound: 0.0006033
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005802, upper bound: 0.0007308
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0004899, upper bound: 0.0006731
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005441, upper bound: 0.0006088
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006087, upper bound: 0.0005441
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006730, upper bound: 0.0004899
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0007325, upper bound: 0.0005082
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006387, upper bound: 0.0005782
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006104, upper bound: 0.0006090
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0007308, upper bound: 0.0005803
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0007311, upper bound: 0.0005147
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006042, upper bound: 0.0005794
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005643, upper bound: 0.0005434
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0004972, upper bound: 0.0006803
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005650, upper bound: 0.0005412
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006386, upper bound: 0.0004757
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005012, upper bound: 0.0006794
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006386, upper bound: 0.0006267
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005613, upper bound: 0.0006070
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0004918, upper bound: 0.0006915
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006147, upper bound: 0.0005512
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006055, upper bound: 0.0006442
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005016, upper bound: 0.0005285
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0004747, upper bound: 0.0006744
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006154, upper bound: 0.0004757
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006055, upper bound: 0.0006280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006134, upper bound: 0.0005561
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006721, upper bound: 0.0004819
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0007326, upper bound: 0.0005017
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006426, upper bound: 0.0005770
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006717, upper bound: 0.0005033
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0004895, upper bound: 0.0005668
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006774, upper bound: 0.0005032
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005153, upper bound: 0.0005730
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0004920, upper bound: 0.0006817
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005174, upper bound: 0.0005369
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005599
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0004654, upper bound: 0.0006817
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005987, upper bound: 0.0005549
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006389, upper bound: 0.0004657
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0005811, upper bound: 0.0006815
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0006389, upper bound: 0.0006181
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0004654, upper bound: 0.0005392
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.43
Output dim: 0, lower bound: -0.0004654, upper bound: 0.0006730

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 21
type: DSZ, layer: 5, pos: 5

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006435, upper bound: 0.0004305
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006453, upper bound: 0.0004316
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 21
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 5

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 5, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006436, upper bound: 0.0004403
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006450, upper bound: 0.0004373
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 21
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 5, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006435, upper bound: 0.0004416
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006549, upper bound: 0.0004408
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 21
type: DSZ, layer: 5, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006429, upper bound: 0.0004428
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006436, upper bound: 0.0004410
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 21
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004680, upper bound: 0.0006622
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004865, upper bound: 0.0006574
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006525, upper bound: 0.0004646
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006628, upper bound: 0.0004616
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 21
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006574, upper bound: 0.0004865
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006622, upper bound: 0.0004678
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 21
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006532, upper bound: 0.0004678
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006623, upper bound: 0.0004622
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 21
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004410, upper bound: 0.0006437
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004428, upper bound: 0.0006429
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Candidate
type: DSZ, layer: 3, pos: 7

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 21
type: DSZ, layer: 5, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004430, upper bound: 0.0006428
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004448, upper bound: 0.0006424
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 21
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004407, upper bound: 0.0006548
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004415, upper bound: 0.0006434
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 21
type: DSZ, layer: 5, pos: 5

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006512, upper bound: 0.0004614
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006639, upper bound: 0.0004553
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Candidate
type: DSZ, layer: 3, pos: 12

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006405, upper bound: 0.0004579
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006405, upper bound: 0.0004510
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 21
type: DSZ, layer: 5, pos: 5

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004373, upper bound: 0.0006450
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004402, upper bound: 0.0006436
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004314, upper bound: 0.0006454
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004304, upper bound: 0.0006434
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577
1: -0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694
2: 0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515
3: -0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447
4: 0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Candidate
type: DSZ, layer: 3, pos: 7

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 21
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004505, upper bound: 0.0006450
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004628, upper bound: 0.0006437
time: 0.17 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.76 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006435, upper bound: 0.0004305
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006453, upper bound: 0.0004316
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006436, upper bound: 0.0004403
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006450, upper bound: 0.0004373
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006435, upper bound: 0.0004416
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006549, upper bound: 0.0004408
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006429, upper bound: 0.0004428
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006436, upper bound: 0.0004410
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004680, upper bound: 0.0006622
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004865, upper bound: 0.0006574
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006525, upper bound: 0.0004646
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006628, upper bound: 0.0004616
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006574, upper bound: 0.0004865
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006622, upper bound: 0.0004678
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006532, upper bound: 0.0004678
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006623, upper bound: 0.0004622
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004410, upper bound: 0.0006437
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004428, upper bound: 0.0006429
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004430, upper bound: 0.0006428
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004448, upper bound: 0.0006424
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004407, upper bound: 0.0006548
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004415, upper bound: 0.0006434
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006512, upper bound: 0.0004614
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006639, upper bound: 0.0004553
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006405, upper bound: 0.0004579
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0006405, upper bound: 0.0004510
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004373, upper bound: 0.0006450
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004402, upper bound: 0.0006436
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004314, upper bound: 0.0006454
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004304, upper bound: 0.0006434
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004505, upper bound: 0.0006450
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0004628, upper bound: 0.0006437

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.51 + 142.43 = 143.94 seconds
