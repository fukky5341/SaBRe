## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 554.967677004936


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141)
1: (-208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797)
2: (-176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039)
3: (-185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579)
4: (-158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.66 + 2.33 = 2.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -554.9898766, upper bound: 554.9898766

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784369, upper bound: 554.9784369
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784369, upper bound: 554.9784369
time: 1.39 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.86 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.86
Output dim: 0, lower bound: -554.9784369, upper bound: 554.9784369
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.86
Output dim: 0, lower bound: -554.9784369, upper bound: 554.9784369

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9717649
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9708893
time: 1.08 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9717649
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9717649, upper bound: 554.9708893
time: 1.11 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.82 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9717649
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9708893
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9717649
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 0, lower bound: -554.9717649, upper bound: 554.9708893

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9717649
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9709086
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9708893
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9708893
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9717649
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9709086
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9708893
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9717649, upper bound: 554.9708893
time: 1.11 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.92 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9717649
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9709086
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9708893
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9708893
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9717649
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9709086
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -554.9708893, upper bound: 554.9708893
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -554.9717649, upper bound: 554.9708893

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684934
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684934
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684934, upper bound: 554.9684930
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930
time: 1.10 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.17 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684934
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684934
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9684934, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684998
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684934
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684998
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684934
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684934, upper bound: 554.9684930
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684998, upper bound: 554.9684930
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.92 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.92 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684998
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684934
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684998
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684934
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684934, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684998, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9688188
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684998
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684934
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684934, upper bound: 554.9684930
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684998, upper bound: 554.9684930
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684935, upper bound: 554.9684930
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9688188, upper bound: 554.9684930
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9688188
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684935
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684998
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684934
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684934, upper bound: 554.9684930
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684998, upper bound: 554.9684930
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
time: 1.00 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9688188
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684998
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684934
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684934, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684998, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684935, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9688188, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9694418
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9688188
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684935
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684998
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684934
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684934, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684998, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9694418, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -554.9684930, upper bound: 554.9684930

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
time: 0.73 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.64 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -554.9005504, upper bound: 554.9005504

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.99 + 371.88 = 374.87 seconds
