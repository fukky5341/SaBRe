## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.5653432899999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970)
1: (-0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661)
2: (-0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567)
3: (-0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351)
4: (-0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.81 + 0.94 = 1.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5950982, upper bound: 0.5950982

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5855678, upper bound: 0.5881289
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5881289, upper bound: 0.5855678
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.66 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.5855678, upper bound: 0.5881289
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.5881289, upper bound: 0.5855678

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5820580, upper bound: 0.5832535
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5858660
time: 0.25 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5858660, upper bound: 0.5681019
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5820580
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.39 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.39
Output dim: 0, lower bound: -0.5820580, upper bound: 0.5832535
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.39
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5858660
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.39
Output dim: 0, lower bound: -0.5858660, upper bound: 0.5681019
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.39
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5820580

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.67 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.67
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.67
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457
time: 0.28 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.40
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.85 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5535169, upper bound: 0.5509407
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203359, upper bound: 0.5507160
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5654198
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 1

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311
time: 0.29 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5535169, upper bound: 0.5509407
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5203359, upper bound: 0.5507160
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5654198
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134638, upper bound: 0.5138966
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134646, upper bound: 0.5143431
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134597, upper bound: 0.5134995
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134951, upper bound: 0.5134995
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134506, upper bound: 0.5158797
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134506, upper bound: 0.5158797
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134630, upper bound: 0.5165941
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134637, upper bound: 0.5165941
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5149852
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5149852
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5168306
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5168306
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5157945
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134690, upper bound: 0.5157945
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5157945, upper bound: 0.5134692
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5157945, upper bound: 0.5134692
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5158797, upper bound: 0.5134506
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5158797, upper bound: 0.5134506
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134995, upper bound: 0.5134951
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134995, upper bound: 0.5134597
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5143431, upper bound: 0.5134646
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5138966, upper bound: 0.5134638
time: 0.28 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.47 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134638, upper bound: 0.5138966
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134646, upper bound: 0.5143431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134597, upper bound: 0.5134995
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134951, upper bound: 0.5134995
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134506, upper bound: 0.5158797
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134506, upper bound: 0.5158797
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134630, upper bound: 0.5165941
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134637, upper bound: 0.5165941
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5149852
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5149852
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5168306
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5168306
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5157945
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134690, upper bound: 0.5157945
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5157945, upper bound: 0.5134692
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5157945, upper bound: 0.5134692
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5158797, upper bound: 0.5134506
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5158797, upper bound: 0.5134506
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134995, upper bound: 0.5134951
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5134995, upper bound: 0.5134597
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5143431, upper bound: 0.5134646
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.47
Output dim: 0, lower bound: -0.5138966, upper bound: 0.5134638

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.75 + 77.37 = 79.12 seconds
