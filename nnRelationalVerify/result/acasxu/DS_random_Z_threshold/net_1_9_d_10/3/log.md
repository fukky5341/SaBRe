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
execution time: IAR + RelationalAnalysis = 1.01 + 0.55 = 1.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0004766, upper bound: 0.0004766

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004747, upper bound: 0.0004748
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004748, upper bound: 0.0004747
time: 0.15 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.32 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.32
Output dim: 0, lower bound: -0.0004747, upper bound: 0.0004748
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.32
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

Time for candidate selection: 0.00 seconds

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
time: 0.16 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

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
time: 0.14 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.15 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.15
Output dim: 0, lower bound: -0.0003303, upper bound: 0.0004743
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.15
Output dim: 0, lower bound: -0.0004742, upper bound: 0.0003094
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.15
Output dim: 0, lower bound: -0.0003094, upper bound: 0.0004742
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.15
Output dim: 0, lower bound: -0.0004743, upper bound: 0.0003303

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002672, upper bound: 0.0004353
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002623, upper bound: 0.0004353
time: 0.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004696, upper bound: 0.0002823
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003607, upper bound: 0.0002931
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003075, upper bound: 0.0004738
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003074, upper bound: 0.0004278
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0002623
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0002672
time: 0.14 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.14 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0002672, upper bound: 0.0004353
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0002623, upper bound: 0.0004353
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0004696, upper bound: 0.0002823
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0003607, upper bound: 0.0002931
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0003075, upper bound: 0.0004738
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0003074, upper bound: 0.0004278
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0002623
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0002672

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002574, upper bound: 0.0003996
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002657, upper bound: 0.0002889
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002514, upper bound: 0.0003996
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002605, upper bound: 0.0002893
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004270, upper bound: 0.0002317
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004270, upper bound: 0.0002317
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003069, upper bound: 0.0004171
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002824, upper bound: 0.0003785
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003059, upper bound: 0.0003933
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003022, upper bound: 0.0003769
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003015, upper bound: 0.0002579
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004350, upper bound: 0.0002616
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002889, upper bound: 0.0002657
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003996, upper bound: 0.0002574
time: 0.15 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0002574, upper bound: 0.0003996
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0002657, upper bound: 0.0002889
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0002514, upper bound: 0.0003996
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0002605, upper bound: 0.0002893
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0004270, upper bound: 0.0002317
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0004270, upper bound: 0.0002317
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0003069, upper bound: 0.0004171
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0002824, upper bound: 0.0003785
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0003059, upper bound: 0.0003933
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0003022, upper bound: 0.0003769
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0003015, upper bound: 0.0002579
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0004350, upper bound: 0.0002616
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0002889, upper bound: 0.0002657
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.10
Output dim: 0, lower bound: -0.0003996, upper bound: 0.0002574

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002454, upper bound: 0.0003010
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002454, upper bound: 0.0003164
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002454, upper bound: 0.0003170
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002454, upper bound: 0.0003170
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002664, upper bound: 0.0002301
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003795, upper bound: 0.0002301
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002301, upper bound: 0.0002301
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003744, upper bound: 0.0002301
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002513, upper bound: 0.0003871
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002513, upper bound: 0.0003882
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

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
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002901, upper bound: 0.0003394
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002690, upper bound: 0.0003849
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003589, upper bound: 0.0002486
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003456, upper bound: 0.0002474
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003879, upper bound: 0.0002334
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003421, upper bound: 0.0002427
time: 0.15 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.16 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0002454, upper bound: 0.0003010
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0002454, upper bound: 0.0003164
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0002454, upper bound: 0.0003170
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0002454, upper bound: 0.0003170
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0002664, upper bound: 0.0002301
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0003795, upper bound: 0.0002301
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0002301, upper bound: 0.0002301
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0003744, upper bound: 0.0002301
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0002513, upper bound: 0.0003871
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0002513, upper bound: 0.0003882
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0002901, upper bound: 0.0003394
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0002690, upper bound: 0.0003849
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0003589, upper bound: 0.0002486
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0003456, upper bound: 0.0002474
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0003879, upper bound: 0.0002334
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.16
Output dim: 0, lower bound: -0.0003421, upper bound: 0.0002427

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 19

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002452, upper bound: 0.0002848
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002452, upper bound: 0.0003004
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 44

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002299, upper bound: 0.0003129
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002299, upper bound: 0.0003776
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002498, upper bound: 0.0003796
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002666, upper bound: 0.0003620
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 44
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Candidate
type: DSZ, layer: 5, pos: 19

### Candidate
type: DSZ, layer: 5, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002299, upper bound: 0.0002299
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003876, upper bound: 0.0002330
time: 0.18 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.32 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.32
Output dim: 0, lower bound: -0.0002452, upper bound: 0.0002848
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.32
Output dim: 0, lower bound: -0.0002452, upper bound: 0.0003004
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.32
Output dim: 0, lower bound: -0.0002299, upper bound: 0.0003129
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.32
Output dim: 0, lower bound: -0.0002299, upper bound: 0.0003776
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.32
Output dim: 0, lower bound: -0.0002498, upper bound: 0.0003796
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.32
Output dim: 0, lower bound: -0.0002666, upper bound: 0.0003620
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.32
Output dim: 0, lower bound: -0.0002299, upper bound: 0.0002299
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.32
Output dim: 0, lower bound: -0.0003876, upper bound: 0.0002330

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356
1: -0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730
2: -0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119
3: -0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747
4: -0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 19
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003023, upper bound: 0.0002246
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002847, upper bound: 0.0002244
time: 0.15 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.18 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.18
Output dim: 0, lower bound: -0.0003023, upper bound: 0.0002246
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.18
Output dim: 0, lower bound: -0.0002847, upper bound: 0.0002244

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.56 + 30.97 = 32.53 seconds
