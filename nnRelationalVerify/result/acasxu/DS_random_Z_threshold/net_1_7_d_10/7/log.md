## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 81.860902399251


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057)
1: (-55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903)
2: (-82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490)
3: (-47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564)
4: (-75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.95 + 3.53 = 4.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -81.8633583, upper bound: 81.8633583

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8633510, upper bound: 81.8633510
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8633510, upper bound: 81.8633546
time: 0.83 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.66 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.66
Output dim: 0, lower bound: -81.8633510, upper bound: 81.8633510
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.66
Output dim: 0, lower bound: -81.8633510, upper bound: 81.8633546

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631302, upper bound: 81.8631302
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631302, upper bound: 81.8631302
time: 0.83 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632973, upper bound: 81.8633233
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632973, upper bound: 81.8633428
time: 0.89 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.68 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -81.8631302, upper bound: 81.8631302
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -81.8631302, upper bound: 81.8631302
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -81.8632973, upper bound: 81.8633233
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -81.8632973, upper bound: 81.8633428

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629793, upper bound: 81.8629750
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629793, upper bound: 81.8629750
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631302, upper bound: 81.8631302
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631302, upper bound: 81.8631302
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632948, upper bound: 81.8633233
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632948, upper bound: 81.8632949
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620774, upper bound: 81.8620774
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620774, upper bound: 81.8620774
time: 0.85 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.59 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -81.8629793, upper bound: 81.8629750
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -81.8629793, upper bound: 81.8629750
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -81.8631302, upper bound: 81.8631302
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -81.8631302, upper bound: 81.8631302
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -81.8632948, upper bound: 81.8633233
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -81.8632948, upper bound: 81.8632949
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -81.8620774, upper bound: 81.8620774
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -81.8620774, upper bound: 81.8620774

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629489, upper bound: 81.8629428
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629428, upper bound: 81.8629428
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626991, upper bound: 81.8626990
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627061, upper bound: 81.8626990
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618507, upper bound: 81.8618583
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618507, upper bound: 81.8618583
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632939, upper bound: 81.8632960
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632939, upper bound: 81.8633233
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632948, upper bound: 81.8632948
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632948, upper bound: 81.8632949
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620753, upper bound: 81.8620753
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620753, upper bound: 81.8620753
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620758, upper bound: 81.8620758
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620758, upper bound: 81.8620758
time: 0.81 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8629489, upper bound: 81.8629428
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8629428, upper bound: 81.8629428
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8626991, upper bound: 81.8626990
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8627061, upper bound: 81.8626990
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8618507, upper bound: 81.8618583
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8618507, upper bound: 81.8618583
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8632939, upper bound: 81.8632960
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8632939, upper bound: 81.8633233
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8632948, upper bound: 81.8632948
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8632948, upper bound: 81.8632949
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8620753, upper bound: 81.8620753
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8620753, upper bound: 81.8620753
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8620758, upper bound: 81.8620758
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -81.8620758, upper bound: 81.8620758

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620981
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620980
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629428, upper bound: 81.8629428
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629428, upper bound: 81.8629428
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626823, upper bound: 81.8626823
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626848, upper bound: 81.8626822
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626991, upper bound: 81.8626991
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626991, upper bound: 81.8626991
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618445, upper bound: 81.8618462
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618445, upper bound: 81.8618583
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618420, upper bound: 81.8618420
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618483, upper bound: 81.8618566
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632939, upper bound: 81.8632960
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632939, upper bound: 81.8632940
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632944, upper bound: 81.8633206
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8632944, upper bound: 81.8633233
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8607089, upper bound: 81.8607089
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8607089, upper bound: 81.8607089
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627269, upper bound: 81.8627269
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627269, upper bound: 81.8627269
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620113, upper bound: 81.8620113
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620113, upper bound: 81.8620113
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620753, upper bound: 81.8620753
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620753, upper bound: 81.8620753
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620758, upper bound: 81.8620758
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620758, upper bound: 81.8620758
time: 1.28 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620981
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620980
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8629428, upper bound: 81.8629428
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8629428, upper bound: 81.8629428
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8626823, upper bound: 81.8626823
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8626848, upper bound: 81.8626822
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8626991, upper bound: 81.8626991
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8626991, upper bound: 81.8626991
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8618445, upper bound: 81.8618462
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8618445, upper bound: 81.8618583
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8618420, upper bound: 81.8618420
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8618483, upper bound: 81.8618566
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8632939, upper bound: 81.8632960
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8632939, upper bound: 81.8632940
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8632944, upper bound: 81.8633206
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8632944, upper bound: 81.8633233
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8607089, upper bound: 81.8607089
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8607089, upper bound: 81.8607089
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8627269, upper bound: 81.8627269
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8627269, upper bound: 81.8627269
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8620113, upper bound: 81.8620113
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8620113, upper bound: 81.8620113
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8620753, upper bound: 81.8620753
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8620753, upper bound: 81.8620753
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8620758, upper bound: 81.8620758
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -81.8620758, upper bound: 81.8620758

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620981
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620981
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626870, upper bound: 81.8626826
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626756, upper bound: 81.8626755
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626756, upper bound: 81.8626756
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626848, upper bound: 81.8626822
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626823, upper bound: 81.8626823
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626015, upper bound: 81.8626015
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626015, upper bound: 81.8626014
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626948, upper bound: 81.8626948
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626948, upper bound: 81.8626947
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618081, upper bound: 81.8618081
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618081, upper bound: 81.8618089
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618373, upper bound: 81.8618373
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618373, upper bound: 81.8618387
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618420, upper bound: 81.8618420
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618420, upper bound: 81.8618420
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618351, upper bound: 81.8618351
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618351, upper bound: 81.8618377
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8628528, upper bound: 81.8628580
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8628528, upper bound: 81.8628580
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631827, upper bound: 81.8631827
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631827, upper bound: 81.8631827
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631858, upper bound: 81.8632049
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631858, upper bound: 81.8631858
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631858, upper bound: 81.8632047
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631859, upper bound: 81.8632045
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8621488, upper bound: 81.8621488
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8621488, upper bound: 81.8621488
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627269, upper bound: 81.8627268
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627269, upper bound: 81.8627269
time: 5.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619786, upper bound: 81.8619786
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619786, upper bound: 81.8619786
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619786, upper bound: 81.8619786
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619786, upper bound: 81.8619786
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619619, upper bound: 81.8619619
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619619, upper bound: 81.8619619
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619619, upper bound: 81.8619619
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619619, upper bound: 81.8619619
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619296, upper bound: 81.8619296
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619296, upper bound: 81.8619296
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620672, upper bound: 81.8620672
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620672, upper bound: 81.8620672
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620758, upper bound: 81.8620758
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620758, upper bound: 81.8620758
time: 0.75 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.83 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620981
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620981
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626870, upper bound: 81.8626826
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626756, upper bound: 81.8626755
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626756, upper bound: 81.8626756
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626848, upper bound: 81.8626822
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626823, upper bound: 81.8626823
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626015, upper bound: 81.8626015
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626015, upper bound: 81.8626014
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626948, upper bound: 81.8626948
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8626948, upper bound: 81.8626947
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8618081, upper bound: 81.8618081
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8618081, upper bound: 81.8618089
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8618373, upper bound: 81.8618373
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8618373, upper bound: 81.8618387
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8618420, upper bound: 81.8618420
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8618420, upper bound: 81.8618420
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8618351, upper bound: 81.8618351
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8618351, upper bound: 81.8618377
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8628528, upper bound: 81.8628580
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8628528, upper bound: 81.8628580
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8631827, upper bound: 81.8631827
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8631827, upper bound: 81.8631827
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8631858, upper bound: 81.8632049
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8631858, upper bound: 81.8631858
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8631858, upper bound: 81.8632047
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8631859, upper bound: 81.8632045
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8621488, upper bound: 81.8621488
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8621488, upper bound: 81.8621488
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8627269, upper bound: 81.8627268
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8627269, upper bound: 81.8627269
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8619786, upper bound: 81.8619786
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8619786, upper bound: 81.8619786
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8619786, upper bound: 81.8619786
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8619786, upper bound: 81.8619786
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8619619, upper bound: 81.8619619
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8619619, upper bound: 81.8619619
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8619619, upper bound: 81.8619619
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8619619, upper bound: 81.8619619
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8619296, upper bound: 81.8619296
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8619296, upper bound: 81.8619296
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8620672, upper bound: 81.8620672
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8620672, upper bound: 81.8620672
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8620758, upper bound: 81.8620758
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -81.8620758, upper bound: 81.8620758

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620962, upper bound: 81.8620962
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620962, upper bound: 81.8620962
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620981
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620981
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625578, upper bound: 81.8625578
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625578, upper bound: 81.8625578
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626870, upper bound: 81.8626826
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626843, upper bound: 81.8626826
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626663, upper bound: 81.8626647
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626663, upper bound: 81.8626646
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618150, upper bound: 81.8618154
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618150, upper bound: 81.8618154
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626197, upper bound: 81.8626343
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626197, upper bound: 81.8626343
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626638, upper bound: 81.8626637
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626638, upper bound: 81.8626638
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626756, upper bound: 81.8626756
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626756, upper bound: 81.8626755
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625834, upper bound: 81.8625812
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625834, upper bound: 81.8625812
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626631, upper bound: 81.8626630
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626631, upper bound: 81.8626631
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625820, upper bound: 81.8625820
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625820, upper bound: 81.8625820
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625820, upper bound: 81.8625820
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625820, upper bound: 81.8625820
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626829, upper bound: 81.8626829
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626887, upper bound: 81.8626829
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626829, upper bound: 81.8626829
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626829, upper bound: 81.8626829
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618081, upper bound: 81.8618081
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618135, upper bound: 81.8618081
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8617914, upper bound: 81.8617914
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8617914, upper bound: 81.8617918
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618373, upper bound: 81.8618373
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618373, upper bound: 81.8618373
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618373, upper bound: 81.8618387
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618373, upper bound: 81.8618378
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618283, upper bound: 81.8618230
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618230, upper bound: 81.8618230
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618056, upper bound: 81.8618056
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618056, upper bound: 81.8618056
time: 1.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618351, upper bound: 81.8618351
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618351, upper bound: 81.8618351
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618351, upper bound: 81.8618377
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618351, upper bound: 81.8618363
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8622768, upper bound: 81.8622768
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8622768, upper bound: 81.8622768
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8628528, upper bound: 81.8628580
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8628528, upper bound: 81.8628528
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629894, upper bound: 81.8629894
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629894, upper bound: 81.8629894
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629995, upper bound: 81.8629994
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8629995, upper bound: 81.8629995
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631774, upper bound: 81.8631774
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631774, upper bound: 81.8632049
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627864, upper bound: 81.8627864
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627864, upper bound: 81.8627864
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631686, upper bound: 81.8631885
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631686, upper bound: 81.8631823
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631859, upper bound: 81.8631858
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8631858, upper bound: 81.8632045
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8621488, upper bound: 81.8621488
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8621488, upper bound: 81.8621488
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8621245, upper bound: 81.8621245
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8621245, upper bound: 81.8621245
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627269, upper bound: 81.8627268
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8627269, upper bound: 81.8627268
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625568, upper bound: 81.8625568
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625568, upper bound: 81.8625568
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618881, upper bound: 81.8618881
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8618881, upper bound: 81.8618881
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619759, upper bound: 81.8619759
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619759, upper bound: 81.8619759
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619759, upper bound: 81.8619759
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619759, upper bound: 81.8619759
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619786, upper bound: 81.8619786
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619786, upper bound: 81.8619786
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619272, upper bound: 81.8619272
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619272, upper bound: 81.8619272
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619374, upper bound: 81.8619374
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619374, upper bound: 81.8619374
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619272, upper bound: 81.8619272
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619272, upper bound: 81.8619272
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619613, upper bound: 81.8619613
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619613, upper bound: 81.8619613
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619296, upper bound: 81.8619296
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619296, upper bound: 81.8619296
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619170, upper bound: 81.8619170
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619170, upper bound: 81.8619169
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620612, upper bound: 81.8620612
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620612, upper bound: 81.8620612
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620578, upper bound: 81.8620578
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620578, upper bound: 81.8620578
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620456, upper bound: 81.8620456
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620456, upper bound: 81.8620456
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619699, upper bound: 81.8619699
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8619699, upper bound: 81.8619699
time: 0.96 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.09 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620962, upper bound: 81.8620962
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620962, upper bound: 81.8620962
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620981
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620981
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8625578, upper bound: 81.8625578
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8625578, upper bound: 81.8625578
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626057, upper bound: 81.8626057
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626870, upper bound: 81.8626826
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626843, upper bound: 81.8626826
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626663, upper bound: 81.8626647
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626663, upper bound: 81.8626646
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626826, upper bound: 81.8626826
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618150, upper bound: 81.8618154
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618150, upper bound: 81.8618154
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626197, upper bound: 81.8626343
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626197, upper bound: 81.8626343
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626638, upper bound: 81.8626637
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626638, upper bound: 81.8626638
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626756, upper bound: 81.8626756
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626756, upper bound: 81.8626755
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8625834, upper bound: 81.8625812
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8625834, upper bound: 81.8625812
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626631, upper bound: 81.8626630
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626631, upper bound: 81.8626631
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8625820, upper bound: 81.8625820
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8625820, upper bound: 81.8625820
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8625820, upper bound: 81.8625820
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8625820, upper bound: 81.8625820
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626829, upper bound: 81.8626829
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626887, upper bound: 81.8626829
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626829, upper bound: 81.8626829
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8626829, upper bound: 81.8626829
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618081, upper bound: 81.8618081
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618135, upper bound: 81.8618081
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8617914, upper bound: 81.8617914
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8617914, upper bound: 81.8617918
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618373, upper bound: 81.8618373
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618373, upper bound: 81.8618373
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618373, upper bound: 81.8618387
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618373, upper bound: 81.8618378
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618283, upper bound: 81.8618230
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618230, upper bound: 81.8618230
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618056, upper bound: 81.8618056
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618056, upper bound: 81.8618056
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618351, upper bound: 81.8618351
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618351, upper bound: 81.8618351
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618351, upper bound: 81.8618377
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618351, upper bound: 81.8618363
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8622768, upper bound: 81.8622768
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8622768, upper bound: 81.8622768
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8628528, upper bound: 81.8628580
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8628528, upper bound: 81.8628528
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8629894, upper bound: 81.8629894
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8629894, upper bound: 81.8629894
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8629995, upper bound: 81.8629994
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8629995, upper bound: 81.8629995
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8631774, upper bound: 81.8631774
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8631774, upper bound: 81.8632049
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8627864, upper bound: 81.8627864
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8627864, upper bound: 81.8627864
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8631686, upper bound: 81.8631885
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8631686, upper bound: 81.8631823
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8631859, upper bound: 81.8631858
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8631858, upper bound: 81.8632045
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8621488, upper bound: 81.8621488
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8621488, upper bound: 81.8621488
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8621245, upper bound: 81.8621245
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8621245, upper bound: 81.8621245
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8627269, upper bound: 81.8627268
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8627269, upper bound: 81.8627268
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8625568, upper bound: 81.8625568
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8625568, upper bound: 81.8625568
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618881, upper bound: 81.8618881
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8618881, upper bound: 81.8618881
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619759, upper bound: 81.8619759
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619759, upper bound: 81.8619759
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619759, upper bound: 81.8619759
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619759, upper bound: 81.8619759
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619786, upper bound: 81.8619786
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619786, upper bound: 81.8619786
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619272, upper bound: 81.8619272
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619272, upper bound: 81.8619272
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619374, upper bound: 81.8619374
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619374, upper bound: 81.8619374
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619272, upper bound: 81.8619272
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619272, upper bound: 81.8619272
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619613, upper bound: 81.8619613
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619613, upper bound: 81.8619613
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620341, upper bound: 81.8620341
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619296, upper bound: 81.8619296
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619296, upper bound: 81.8619296
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619170, upper bound: 81.8619170
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619170, upper bound: 81.8619169
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620612, upper bound: 81.8620612
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620612, upper bound: 81.8620612
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620578, upper bound: 81.8620578
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620578, upper bound: 81.8620578
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620456, upper bound: 81.8620456
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8620456, upper bound: 81.8620456
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619699, upper bound: 81.8619699
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -81.8619699, upper bound: 81.8619699

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620042, upper bound: 81.8620042
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620042, upper bound: 81.8620042
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620568, upper bound: 81.8620568
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620568, upper bound: 81.8620568
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620591, upper bound: 81.8620591
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620745, upper bound: 81.8620745
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8616836, upper bound: 81.8616836
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8616836, upper bound: 81.8616836
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620981
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620980
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625578, upper bound: 81.8625577
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625578, upper bound: 81.8625578
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625578, upper bound: 81.8625578
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625578, upper bound: 81.8625578
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620980
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8620981, upper bound: 81.8620980
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626012, upper bound: 81.8626012
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626012, upper bound: 81.8626012
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626027, upper bound: 81.8626027
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8626027, upper bound: 81.8626027
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625578, upper bound: 81.8625795
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8625578, upper bound: 81.8625578
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.48 + 415.57 = 420.06 seconds
