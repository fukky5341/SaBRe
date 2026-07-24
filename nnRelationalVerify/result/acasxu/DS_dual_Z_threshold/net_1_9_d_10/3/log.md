## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.00038128


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356)
1: (-0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730)
2: (-0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119)
3: (-0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747)
4: (-0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.17 + 0.55 = 1.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0004766, upper bound: 0.0004766

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004747, upper bound: 0.0004748
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004748, upper bound: 0.0004747
time: 0.14 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.48 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.48
Output dim: 0, lower bound: -0.0004747, upper bound: 0.0004748
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.48
Output dim: 0, lower bound: -0.0004748, upper bound: 0.0004747

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003303, upper bound: 0.0004743
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004742, upper bound: 0.0003094
time: 0.14 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003094, upper bound: 0.0004742
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004743, upper bound: 0.0003303
time: 0.15 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.26 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.26
Output dim: 0, lower bound: -0.0003303, upper bound: 0.0004743
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.26
Output dim: 0, lower bound: -0.0004742, upper bound: 0.0003094
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.26
Output dim: 0, lower bound: -0.0003094, upper bound: 0.0004742
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.26
Output dim: 0, lower bound: -0.0004743, upper bound: 0.0003303

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003268, upper bound: 0.0003975
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002885, upper bound: 0.0004645
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004696, upper bound: 0.0002823
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003607, upper bound: 0.0002931
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002931, upper bound: 0.0003607
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002823, upper bound: 0.0004696
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004645, upper bound: 0.0002885
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003975, upper bound: 0.0003268
time: 0.16 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.35 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.35
Output dim: 0, lower bound: -0.0003268, upper bound: 0.0003975
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.35
Output dim: 0, lower bound: -0.0002885, upper bound: 0.0004645
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.35
Output dim: 0, lower bound: -0.0004696, upper bound: 0.0002823
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.35
Output dim: 0, lower bound: -0.0003607, upper bound: 0.0002931
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.35
Output dim: 0, lower bound: -0.0002931, upper bound: 0.0003607
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.35
Output dim: 0, lower bound: -0.0002823, upper bound: 0.0004696
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.35
Output dim: 0, lower bound: -0.0004645, upper bound: 0.0002885
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.35
Output dim: 0, lower bound: -0.0003975, upper bound: 0.0003268

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002842, upper bound: 0.0003937
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003145, upper bound: 0.0003944
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002653, upper bound: 0.0004569
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002860, upper bound: 0.0004607
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004655, upper bound: 0.0002780
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004569, upper bound: 0.0002575
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002575, upper bound: 0.0004569
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002780, upper bound: 0.0004655
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004607, upper bound: 0.0002860
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004569, upper bound: 0.0002653
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003944, upper bound: 0.0003145
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003937, upper bound: 0.0002842
time: 0.16 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.29 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0002842, upper bound: 0.0003937
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0003145, upper bound: 0.0003944
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0002653, upper bound: 0.0004569
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0002860, upper bound: 0.0004607
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0004655, upper bound: 0.0002780
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0004569, upper bound: 0.0002575
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0002575, upper bound: 0.0004569
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0002780, upper bound: 0.0004655
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0004607, upper bound: 0.0002860
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0004569, upper bound: 0.0002653
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0003944, upper bound: 0.0003145
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0003937, upper bound: 0.0002842

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002539, upper bound: 0.0003816
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002722, upper bound: 0.0003553
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002973, upper bound: 0.0003819
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003054, upper bound: 0.0003265
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002578, upper bound: 0.0004000
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002599, upper bound: 0.0003722
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002852, upper bound: 0.0003933
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002629, upper bound: 0.0002546
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002539, upper bound: 0.0002539
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003885, upper bound: 0.0002774
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003623, upper bound: 0.0002539
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004003, upper bound: 0.0002539
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002539, upper bound: 0.0004003
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002539, upper bound: 0.0003623
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002774, upper bound: 0.0003885
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002539, upper bound: 0.0002539
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002546, upper bound: 0.0002629
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003933, upper bound: 0.0002852
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003722, upper bound: 0.0002599
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004000, upper bound: 0.0002578
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003265, upper bound: 0.0003054
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003819, upper bound: 0.0002973
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003553, upper bound: 0.0002722
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003816, upper bound: 0.0002539
time: 0.16 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.33 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0002539, upper bound: 0.0003816
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0002722, upper bound: 0.0003553
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0002973, upper bound: 0.0003819
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0003054, upper bound: 0.0003265
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0002578, upper bound: 0.0004000
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0002599, upper bound: 0.0003722
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0002852, upper bound: 0.0003933
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0002629, upper bound: 0.0002546
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0002539, upper bound: 0.0002539
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0003885, upper bound: 0.0002774
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0003623, upper bound: 0.0002539
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0004003, upper bound: 0.0002539
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0002539, upper bound: 0.0004003
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0002539, upper bound: 0.0003623
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0002774, upper bound: 0.0003885
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0002539, upper bound: 0.0002539
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0002546, upper bound: 0.0002629
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0003933, upper bound: 0.0002852
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0003722, upper bound: 0.0002599
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0004000, upper bound: 0.0002578
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0003265, upper bound: 0.0003054
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0003819, upper bound: 0.0002973
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0003553, upper bound: 0.0002722
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -0.0003816, upper bound: 0.0002539

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002331
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002543
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002252, upper bound: 0.0002615
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002615
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002246, upper bound: 0.0002881
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0003040
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002249, upper bound: 0.0003025
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0003040
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002896, upper bound: 0.0002242
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002881, upper bound: 0.0002242
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002896, upper bound: 0.0002242
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002749, upper bound: 0.0002242
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002749
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002896
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002881
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002896
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003040, upper bound: 0.0002242
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003025, upper bound: 0.0002249
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003040, upper bound: 0.0002242
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002881, upper bound: 0.0002246
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002615, upper bound: 0.0002242
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002615, upper bound: 0.0002252
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 30

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002543, upper bound: 0.0002242
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002331, upper bound: 0.0002242
time: 0.16 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002331
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002543
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002252, upper bound: 0.0002615
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002615
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002246, upper bound: 0.0002881
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0003040
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002249, upper bound: 0.0003025
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0003040
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002896, upper bound: 0.0002242
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002881, upper bound: 0.0002242
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002896, upper bound: 0.0002242
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002749, upper bound: 0.0002242
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002749
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002896
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002881
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002242, upper bound: 0.0002896
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0003040, upper bound: 0.0002242
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0003025, upper bound: 0.0002249
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0003040, upper bound: 0.0002242
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002881, upper bound: 0.0002246
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002615, upper bound: 0.0002242
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002615, upper bound: 0.0002252
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002543, upper bound: 0.0002242
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0002331, upper bound: 0.0002242

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.72 + 47.99 = 49.71 seconds
