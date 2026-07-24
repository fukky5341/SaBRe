## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 3.05840151


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395)
1: (-1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596)
2: (-1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697)
3: (-3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084)
4: (-2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.59 + 1.14 = 1.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3.3982239, upper bound: 3.3982239

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3430054, upper bound: 3.3430054
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3430054, upper bound: 3.3430054
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.65 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -3.3430054, upper bound: 3.3430054
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -3.3430054, upper bound: 3.3430054

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.29 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.19 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.19
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.19
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.19
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.19
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.25 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.27 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.18 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.18
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.36 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.36
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.29 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.76 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.76
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.30 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.08 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.08
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 9
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 9
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 9
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 9
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 9
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 9
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 9
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.39 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 9
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.41 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 4

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 9
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.35 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 1.78 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.78
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 1.73 + 418.74 = 420.47 seconds
