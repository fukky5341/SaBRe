## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 56.43210397135999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973)
1: (-30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767)
2: (-21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816)
3: (-20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810)
4: (-17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.81 + 1.56 = 2.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -56.4772858, upper bound: 56.4772858

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4771448, upper bound: 56.4772858
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4771448, upper bound: 56.4771448
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.95 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.95
Output dim: 4, lower bound: -56.4771448, upper bound: 56.4772858
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.95
Output dim: 4, lower bound: -56.4771448, upper bound: 56.4771448

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4737531, upper bound: 56.4740145
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4737531, upper bound: 56.4740144
time: 0.42 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4749439, upper bound: 56.4749439
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4749439, upper bound: 56.4750942
time: 0.49 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.79 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.79
Output dim: 4, lower bound: -56.4737531, upper bound: 56.4740145
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.79
Output dim: 4, lower bound: -56.4737531, upper bound: 56.4740144
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.79
Output dim: 4, lower bound: -56.4749439, upper bound: 56.4749439
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.79
Output dim: 4, lower bound: -56.4749439, upper bound: 56.4750942

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4692202, upper bound: 56.4695641
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4692202, upper bound: 56.4692303
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4731613, upper bound: 56.4738926
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4731613, upper bound: 56.4737200
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4696733, upper bound: 56.4693705
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4694988, upper bound: 56.4693705
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4702381, upper bound: 56.4706856
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4702329, upper bound: 56.4706343
time: 0.47 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.73 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.73
Output dim: 4, lower bound: -56.4692202, upper bound: 56.4695641
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.73
Output dim: 4, lower bound: -56.4692202, upper bound: 56.4692303
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.73
Output dim: 4, lower bound: -56.4731613, upper bound: 56.4738926
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.73
Output dim: 4, lower bound: -56.4731613, upper bound: 56.4737200
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.73
Output dim: 4, lower bound: -56.4696733, upper bound: 56.4693705
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.73
Output dim: 4, lower bound: -56.4694988, upper bound: 56.4693705
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.73
Output dim: 4, lower bound: -56.4702381, upper bound: 56.4706856
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.73
Output dim: 4, lower bound: -56.4702329, upper bound: 56.4706343

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4691640, upper bound: 56.4694660
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4691645, upper bound: 56.4695066
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4693126, upper bound: 56.4692303
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4692200, upper bound: 56.4692200
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4731403, upper bound: 56.4737257
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4731403, upper bound: 56.4738831
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4690982, upper bound: 56.4695095
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4690982, upper bound: 56.4690996
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4693705, upper bound: 56.4693705
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4694646, upper bound: 56.4693705
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4696490, upper bound: 56.4693705
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4694646, upper bound: 56.4693705
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4701747, upper bound: 56.4706856
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4703965, upper bound: 56.4703014
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4701747, upper bound: 56.4706210
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4702053, upper bound: 56.4703541
time: 0.42 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4691640, upper bound: 56.4694660
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4691645, upper bound: 56.4695066
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4693126, upper bound: 56.4692303
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4692200, upper bound: 56.4692200
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4731403, upper bound: 56.4737257
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4731403, upper bound: 56.4738831
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4690982, upper bound: 56.4695095
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4690982, upper bound: 56.4690996
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4693705, upper bound: 56.4693705
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4694646, upper bound: 56.4693705
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4696490, upper bound: 56.4693705
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4694646, upper bound: 56.4693705
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4701747, upper bound: 56.4706856
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4703965, upper bound: 56.4703014
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4701747, upper bound: 56.4706210
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 4, lower bound: -56.4702053, upper bound: 56.4703541

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4691636, upper bound: 56.4694660
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4691636, upper bound: 56.4691631
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4691620, upper bound: 56.4691620
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4691620, upper bound: 56.4694930
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4692200, upper bound: 56.4692200
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4692934, upper bound: 56.4692303
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4654170, upper bound: 56.4654170
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4655366, upper bound: 56.4654170
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4727468, upper bound: 56.4733285
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4727468, upper bound: 56.4733523
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4022598, upper bound: 56.4022598
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4022598, upper bound: 56.4022598
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4690982, upper bound: 56.4690982
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4690982, upper bound: 56.4693044
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4485464, upper bound: 56.4485483
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4485464, upper bound: 56.4485483
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4694347, upper bound: 56.4693016
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4693091, upper bound: 56.4693016
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4675160, upper bound: 56.4675150
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4675145, upper bound: 56.4675150
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4678526, upper bound: 56.4675145
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4678679, upper bound: 56.4675145
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4652058, upper bound: 56.4652129
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4653315, upper bound: 56.4652205
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4581410
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4584222
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4701790, upper bound: 56.4703014
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4703965, upper bound: 56.4702860
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4645359, upper bound: 56.4649384
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4645359, upper bound: 56.4649775
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4650430, upper bound: 56.4650440
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4650430, upper bound: 56.4650430
time: 0.45 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4691636, upper bound: 56.4694660
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4691636, upper bound: 56.4691631
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4691620, upper bound: 56.4691620
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4691620, upper bound: 56.4694930
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4692200, upper bound: 56.4692200
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4692934, upper bound: 56.4692303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4654170, upper bound: 56.4654170
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4655366, upper bound: 56.4654170
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4727468, upper bound: 56.4733285
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4727468, upper bound: 56.4733523
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4022598, upper bound: 56.4022598
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4022598, upper bound: 56.4022598
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4690982, upper bound: 56.4690982
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4690982, upper bound: 56.4693044
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4485464, upper bound: 56.4485483
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4485464, upper bound: 56.4485483
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4694347, upper bound: 56.4693016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4693091, upper bound: 56.4693016
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4675160, upper bound: 56.4675150
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4675145, upper bound: 56.4675150
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4678526, upper bound: 56.4675145
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4678679, upper bound: 56.4675145
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4652058, upper bound: 56.4652129
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4653315, upper bound: 56.4652205
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4581410
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4584222
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4701790, upper bound: 56.4703014
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4703965, upper bound: 56.4702860
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4645359, upper bound: 56.4649384
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4645359, upper bound: 56.4649775
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4650430, upper bound: 56.4650440
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 4, lower bound: -56.4650430, upper bound: 56.4650430

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4637399, upper bound: 56.4637399
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4637533, upper bound: 56.4637990
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4483192, upper bound: 56.4483192
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4483192, upper bound: 56.4483192
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4000228, upper bound: 56.4000228
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4000228, upper bound: 56.4000228
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4691620, upper bound: 56.4694930
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4691620, upper bound: 56.4691620
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4692200, upper bound: 56.4692200
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4692200, upper bound: 56.4692200
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4637882, upper bound: 56.4637550
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4637864, upper bound: 56.4637664
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478069
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478069
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4653750
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4653750
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4691800
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4687532
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4692043
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4687851
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4687163
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4687163
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4656173
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4653819, upper bound: 56.4653750
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4481601, upper bound: 56.4481536
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4481528, upper bound: 56.4481528
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478069
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478135
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4693016, upper bound: 56.4693016
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4693016, upper bound: 56.4693016
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4693016, upper bound: 56.4693016
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4693052, upper bound: 56.4693016
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4669976, upper bound: 56.4670026
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4669976, upper bound: 56.4670026
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4639343, upper bound: 56.4639343
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4639343, upper bound: 56.4639343
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543407, upper bound: 56.4543401
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4670458, upper bound: 56.4669976
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4670531, upper bound: 56.4669976
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4652058, upper bound: 56.4652129
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4652058, upper bound: 56.4652057
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543295, upper bound: 56.4544452
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4542963, upper bound: 56.4543105
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4542761, upper bound: 56.4547743
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4542761, upper bound: 56.4542761
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4584222
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4574695
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4699698, upper bound: 56.4699698
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4699698, upper bound: 56.4700877
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4575370
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4574153
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4545224
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4547420
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4547333
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
time: 0.44 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.78 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4637399, upper bound: 56.4637399
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4637533, upper bound: 56.4637990
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4483192, upper bound: 56.4483192
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4483192, upper bound: 56.4483192
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4000228, upper bound: 56.4000228
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4000228, upper bound: 56.4000228
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4691620, upper bound: 56.4694930
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4691620, upper bound: 56.4691620
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4692200, upper bound: 56.4692200
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4692200, upper bound: 56.4692200
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4637882, upper bound: 56.4637550
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4637864, upper bound: 56.4637664
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478069
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478069
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4653750
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4653750
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4691800
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4687532
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4692043
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4687851
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4687163
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4687163
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4656173
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4653819, upper bound: 56.4653750
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4481601, upper bound: 56.4481536
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4481528, upper bound: 56.4481528
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478069
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478135
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4693016, upper bound: 56.4693016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4693016, upper bound: 56.4693016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4693016, upper bound: 56.4693016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4693052, upper bound: 56.4693016
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4669976, upper bound: 56.4670026
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4669976, upper bound: 56.4670026
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4639343, upper bound: 56.4639343
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4639343, upper bound: 56.4639343
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4543407, upper bound: 56.4543401
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4670458, upper bound: 56.4669976
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4670531, upper bound: 56.4669976
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4652058, upper bound: 56.4652129
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4652058, upper bound: 56.4652057
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4543295, upper bound: 56.4544452
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4542963, upper bound: 56.4543105
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4542761, upper bound: 56.4547743
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4542761, upper bound: 56.4542761
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4584222
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4574695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4699698, upper bound: 56.4699698
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4699698, upper bound: 56.4700877
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4575370
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4574153
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4545224
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4547420
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4547333
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.78
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4637399, upper bound: 56.4637399
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4637399, upper bound: 56.4637399
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4637533, upper bound: 56.4637990
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4637519, upper bound: 56.4637611
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4483192, upper bound: 56.4483192
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4483192, upper bound: 56.4483192
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4476065, upper bound: 56.4476065
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4476065, upper bound: 56.4476065
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4691620, upper bound: 56.4694930
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4691620, upper bound: 56.4691620
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4636921, upper bound: 56.4636921
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4636921, upper bound: 56.4636921
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687906, upper bound: 56.4687906
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687906, upper bound: 56.4687906
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4671944, upper bound: 56.4671944
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4671944, upper bound: 56.4671944
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4602613, upper bound: 56.4600991
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4633837, upper bound: 56.4633563
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4633396, upper bound: 56.4633396
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4473064, upper bound: 56.4473064
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4473064, upper bound: 56.4473064
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4328725, upper bound: 56.4328725
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4328725, upper bound: 56.4328725
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4653750
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4653750
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4317028, upper bound: 56.4317028
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4317028, upper bound: 56.4317028
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.3939100, upper bound: 56.3939100
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.3939100, upper bound: 56.3939100
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664946, upper bound: 56.4665233
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664946, upper bound: 56.4665234
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4692043
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4691743
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4650724
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4649675
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4584408, upper bound: 56.4584408
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4584408, upper bound: 56.4584408
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4687163
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4687163
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4656170
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4653750
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649747, upper bound: 56.4649679
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4649675
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4481601, upper bound: 56.4481528
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4481528, upper bound: 56.4481536
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4333313, upper bound: 56.4333313
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4333313, upper bound: 56.4333313
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4311298, upper bound: 56.4311298
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4311298, upper bound: 56.4311298
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478069
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478135
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4668138
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4668038
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4307364, upper bound: 56.4307364
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4307364, upper bound: 56.4307364
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4667885
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4667885
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4307364, upper bound: 56.4307364
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4307364, upper bound: 56.4307364
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.3940621, upper bound: 56.3940621
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.3940621, upper bound: 56.3940621
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4358883, upper bound: 56.4358883
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4358883, upper bound: 56.4358883
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4638775, upper bound: 56.4638775
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4638775, upper bound: 56.4638775
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4518837, upper bound: 56.4518837
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4518837, upper bound: 56.4518837
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.3940621, upper bound: 56.3940621
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.3940621, upper bound: 56.3940621
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4667885
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4667885
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4472376, upper bound: 56.4472376
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4472376, upper bound: 56.4472376
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4639343, upper bound: 56.4639343
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4639343, upper bound: 56.4639343
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4542963, upper bound: 56.4544452
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543295, upper bound: 56.4544437
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541397, upper bound: 56.4541568
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541397, upper bound: 56.4541397
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541164, upper bound: 56.4545922
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541164, upper bound: 56.4541164
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4518542, upper bound: 56.4518542
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4518542, upper bound: 56.4518542
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4581854
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4581840
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4572169
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4571708
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4647553, upper bound: 56.4647553
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4647553, upper bound: 56.4647553
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4696887, upper bound: 56.4698263
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4696921, upper bound: 56.4696887
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4575370
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4574153
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4571708
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4571708
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4543157
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4547420
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4547333
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4519454, upper bound: 56.4518865
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4518542, upper bound: 56.4518542
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4596844, upper bound: 56.4596844
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4596844, upper bound: 56.4596844
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
time: 0.54 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.96 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4637399, upper bound: 56.4637399
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4637399, upper bound: 56.4637399
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4637533, upper bound: 56.4637990
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4637519, upper bound: 56.4637611
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4483192, upper bound: 56.4483192
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4483192, upper bound: 56.4483192
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4476065, upper bound: 56.4476065
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4476065, upper bound: 56.4476065
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4691620, upper bound: 56.4694930
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4691620, upper bound: 56.4691620
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4636921, upper bound: 56.4636921
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4636921, upper bound: 56.4636921
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4687906, upper bound: 56.4687906
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4687906, upper bound: 56.4687906
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4671944, upper bound: 56.4671944
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4671944, upper bound: 56.4671944
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4602613, upper bound: 56.4600991
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4633837, upper bound: 56.4633563
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4633396, upper bound: 56.4633396
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4473064, upper bound: 56.4473064
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4473064, upper bound: 56.4473064
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4328725, upper bound: 56.4328725
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4328725, upper bound: 56.4328725
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4653750
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4653750
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4317028, upper bound: 56.4317028
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4317028, upper bound: 56.4317028
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.3939100, upper bound: 56.3939100
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.3939100, upper bound: 56.3939100
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4664946, upper bound: 56.4665233
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4664946, upper bound: 56.4665234
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4692043
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4691743
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4650724
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4649675
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4584408, upper bound: 56.4584408
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4584408, upper bound: 56.4584408
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4687163
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4687163, upper bound: 56.4687163
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4656170
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4653750
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4649747, upper bound: 56.4649679
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4649675
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4481601, upper bound: 56.4481528
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4481528, upper bound: 56.4481536
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4333313, upper bound: 56.4333313
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4333313, upper bound: 56.4333313
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4311298, upper bound: 56.4311298
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4311298, upper bound: 56.4311298
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478069
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478135
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4668138
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4668038
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4307364, upper bound: 56.4307364
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4307364, upper bound: 56.4307364
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4667885
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4667885
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4307364, upper bound: 56.4307364
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4307364, upper bound: 56.4307364
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.3940621, upper bound: 56.3940621
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.3940621, upper bound: 56.3940621
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4358883, upper bound: 56.4358883
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4358883, upper bound: 56.4358883
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4638775, upper bound: 56.4638775
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4638775, upper bound: 56.4638775
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4518837, upper bound: 56.4518837
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4518837, upper bound: 56.4518837
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.3940621, upper bound: 56.3940621
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.3940621, upper bound: 56.3940621
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4667885
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4667885
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4472376, upper bound: 56.4472376
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4472376, upper bound: 56.4472376
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4639343, upper bound: 56.4639343
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4639343, upper bound: 56.4639343
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4542963, upper bound: 56.4544452
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4543295, upper bound: 56.4544437
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4541397, upper bound: 56.4541568
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4541397, upper bound: 56.4541397
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4541164, upper bound: 56.4545922
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4541164, upper bound: 56.4541164
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4518542, upper bound: 56.4518542
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4518542, upper bound: 56.4518542
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4581854
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4581840
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4572169
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4571708
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4647553, upper bound: 56.4647553
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4647553, upper bound: 56.4647553
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4696887, upper bound: 56.4698263
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4696921, upper bound: 56.4696887
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4575370
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4574153, upper bound: 56.4574153
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4571708
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4571708
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4543157
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4547420
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4547333
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4519454, upper bound: 56.4518865
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4518542, upper bound: 56.4518542
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4596844, upper bound: 56.4596844
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4596844, upper bound: 56.4596844
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.96
Output dim: 4, lower bound: -56.4600991, upper bound: 56.4600991

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4633236, upper bound: 56.4633236
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4633236, upper bound: 56.4633236
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4636921, upper bound: 56.4636921
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4636921, upper bound: 56.4636921
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4636921, upper bound: 56.4637108
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4636921, upper bound: 56.4637905
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600789, upper bound: 56.4600678
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600789, upper bound: 56.4602172
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4320340, upper bound: 56.4320340
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4320340, upper bound: 56.4320340
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4000228, upper bound: 56.4000228
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4000228, upper bound: 56.4000228
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4471210, upper bound: 56.4471210
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4471210, upper bound: 56.4471210
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4471210, upper bound: 56.4471210
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4471210, upper bound: 56.4471210
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4656081
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4653750, upper bound: 56.4653750
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4670946, upper bound: 56.4670946
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4670946, upper bound: 56.4670946
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4632754, upper bound: 56.4632754
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4632754, upper bound: 56.4632754
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4599798, upper bound: 56.4599798
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4599798, upper bound: 56.4599798
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4668237, upper bound: 56.4668237
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4668237, upper bound: 56.4668237
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687315, upper bound: 56.4687315
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4687315, upper bound: 56.4687315
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4000228, upper bound: 56.4000228
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4000228, upper bound: 56.4000228
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4668237, upper bound: 56.4668237
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4668237, upper bound: 56.4668237
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4596844, upper bound: 56.4596844
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4596844, upper bound: 56.4596844
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4596844, upper bound: 56.4596844
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4596844, upper bound: 56.4596844
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4561481, upper bound: 56.4561481
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4563619, upper bound: 56.4561481
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4597258, upper bound: 56.4596844
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4596844, upper bound: 56.4596844
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4473064, upper bound: 56.4473064
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4473064, upper bound: 56.4473064
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4328031, upper bound: 56.4328031
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4328031, upper bound: 56.4328031
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4328725, upper bound: 56.4328725
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4328725, upper bound: 56.4328725
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4328031, upper bound: 56.4328031
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4328031, upper bound: 56.4328031
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4635022, upper bound: 56.4635022
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4635022, upper bound: 56.4635022
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4649675
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4649675
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664946, upper bound: 56.4665233
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664946, upper bound: 56.4664988
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4537724, upper bound: 56.4540995
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4537724, upper bound: 56.4540995
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4651499
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4649675
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4651356
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4649675
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.3916782, upper bound: 56.3916782
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.3916782, upper bound: 56.3916782
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4631137, upper bound: 56.4631137
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4631137, upper bound: 56.4631137
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4537724, upper bound: 56.4537724
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4537724, upper bound: 56.4537724
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4584408, upper bound: 56.4584402
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4584408, upper bound: 56.4584408
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664946, upper bound: 56.4664946
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664946, upper bound: 56.4664998
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4649675
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4649675
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4651865
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4649675
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4635022, upper bound: 56.4635044
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4635022, upper bound: 56.4635022
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4649679
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4649675, upper bound: 56.4649675
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4631137, upper bound: 56.4631137
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4631137, upper bound: 56.4631137
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4333313, upper bound: 56.4333313
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4333313, upper bound: 56.4333313
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4475040, upper bound: 56.4475040
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4475040, upper bound: 56.4475044
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4333313, upper bound: 56.4333313
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4333313, upper bound: 56.4333313
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4333311, upper bound: 56.4333311
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4333311, upper bound: 56.4333311
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478069
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4478069, upper bound: 56.4478069
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4311298, upper bound: 56.4311298
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4311298, upper bound: 56.4311298
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664946, upper bound: 56.4665171
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664946, upper bound: 56.4664946
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4635022, upper bound: 56.4635022
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4635022, upper bound: 56.4635022
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4667885
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4667885, upper bound: 56.4667885
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539979, upper bound: 56.4539979
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539979, upper bound: 56.4539979
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4358315, upper bound: 56.4358315
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4358315, upper bound: 56.4358315
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4328725, upper bound: 56.4328725
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4328725, upper bound: 56.4328725
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4635151, upper bound: 56.4635151
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4635151, upper bound: 56.4635151
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4471210, upper bound: 56.4471210
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4471210, upper bound: 56.4471210
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4518837, upper bound: 56.4518837
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4518837, upper bound: 56.4518837
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4517376, upper bound: 56.4517376
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4517376, upper bound: 56.4517376
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4515112, upper bound: 56.4515112
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4515112, upper bound: 56.4515112
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664946, upper bound: 56.4664946
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664946, upper bound: 56.4664946
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664946, upper bound: 56.4664946
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4665189, upper bound: 56.4664946
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4468592, upper bound: 56.4468592
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4468592, upper bound: 56.4468592
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4329540, upper bound: 56.4329540
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4329540, upper bound: 56.4329540
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4638775, upper bound: 56.4638775
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4638775, upper bound: 56.4638775
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4471210, upper bound: 56.4471210
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4471210, upper bound: 56.4471210
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539869, upper bound: 56.4541335
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539869, upper bound: 56.4539869
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539869, upper bound: 56.4541314
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539876, upper bound: 56.4539869
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541164, upper bound: 56.4541362
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541164, upper bound: 56.4541164
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541397, upper bound: 56.4541397
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541397, upper bound: 56.4541397
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4538008, upper bound: 56.4538908
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4538008, upper bound: 56.4543026
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4517082, upper bound: 56.4517082
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4517082, upper bound: 56.4517082
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4514817, upper bound: 56.4514817
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4514817, upper bound: 56.4514817
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4517082, upper bound: 56.4517082
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4517082, upper bound: 56.4517082
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539979, upper bound: 56.4539979
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539979, upper bound: 56.4543455
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539979, upper bound: 56.4539979
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539979, upper bound: 56.4544133
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539649, upper bound: 56.4539649
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539649, upper bound: 56.4541277
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539649, upper bound: 56.4539649
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539649, upper bound: 56.4539649
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4647553, upper bound: 56.4647553
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4647553, upper bound: 56.4647553
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4647553, upper bound: 56.4647553
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4647553, upper bound: 56.4647553
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4696887, upper bound: 56.4698263
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4696887, upper bound: 56.4696887
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4569470, upper bound: 56.4568845
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4568845, upper bound: 56.4568845
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4571708
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4572927
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4571708
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4571708, upper bound: 56.4571708
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539649, upper bound: 56.4539649
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539649, upper bound: 56.4539649
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4568845, upper bound: 56.4568845
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4568845, upper bound: 56.4568845
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4537724, upper bound: 56.4537826
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4537724, upper bound: 56.4539953
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4517082, upper bound: 56.4517082
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4517082, upper bound: 56.4517082
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539979, upper bound: 56.4539979
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539979, upper bound: 56.4539979
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4545204
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539979, upper bound: 56.4539979
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4539979, upper bound: 56.4539979
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4545094
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543106, upper bound: 56.4543106
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4518542, upper bound: 56.4518865
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4518542, upper bound: 56.4518749
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4518542, upper bound: 56.4518542
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4518542, upper bound: 56.4518542
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4596844, upper bound: 56.4596844
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4596844, upper bound: 56.4596844
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 1.16 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.37 + 418.29 = 420.66 seconds
