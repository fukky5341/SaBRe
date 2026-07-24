## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 465.361891711094


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480)
1: (-100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447)
2: (-110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621)
3: (-99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465)
4: (-158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.63 + 2.12 = 3.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -465.3898151, upper bound: 465.3898151

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3636018
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3636018
time: 0.89 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.95 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.95
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3636018
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.95
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3636018

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3635971
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3635971, upper bound: 465.3636018
time: 0.80 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3635971
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3635971, upper bound: 465.3636018
time: 0.71 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.34 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3635971
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -465.3635971, upper bound: 465.3636018
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3635971
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -465.3635971, upper bound: 465.3636018

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3635951, upper bound: 465.3635971
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3635951, upper bound: 465.3635951
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3635951, upper bound: 465.3636018
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3635971, upper bound: 465.3635951
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3635951, upper bound: 465.3635971
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3635951
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3635951, upper bound: 465.3636018
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3635951, upper bound: 465.3635951
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.42 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.42
Output dim: 0, lower bound: -465.3635951, upper bound: 465.3635971
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.42
Output dim: 0, lower bound: -465.3635951, upper bound: 465.3635951
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.42
Output dim: 0, lower bound: -465.3635951, upper bound: 465.3636018
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.42
Output dim: 0, lower bound: -465.3635971, upper bound: 465.3635951
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.42
Output dim: 0, lower bound: -465.3635951, upper bound: 465.3635971
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.42
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3635951
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.42
Output dim: 0, lower bound: -465.3635951, upper bound: 465.3636018
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.42
Output dim: 0, lower bound: -465.3635951, upper bound: 465.3635951

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480
1: -100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447
2: -110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621
3: -99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465
4: -158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 0.99 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.42 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.42
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.76 + 50.41 = 54.17 seconds
