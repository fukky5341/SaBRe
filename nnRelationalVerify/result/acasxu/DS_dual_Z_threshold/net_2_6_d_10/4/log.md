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
execution time: IAR + RelationalAnalysis = 2.26 + 1.68 = 3.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -56.4772858, upper bound: 56.4772858

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4739328, upper bound: 56.4740208
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4739328, upper bound: 56.4739328
time: 0.53 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.30 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.30
Output dim: 4, lower bound: -56.4739328, upper bound: 56.4740208
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.30
Output dim: 4, lower bound: -56.4739328, upper bound: 56.4739328

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4706840, upper bound: 56.4702381
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4704139, upper bound: 56.4706856
time: 0.58 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4702381, upper bound: 56.4704139
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4702381, upper bound: 56.4706840
time: 0.56 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.45 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 4, lower bound: -56.4706840, upper bound: 56.4702381
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 4, lower bound: -56.4704139, upper bound: 56.4706856
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 4, lower bound: -56.4702381, upper bound: 56.4704139
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 4, lower bound: -56.4702381, upper bound: 56.4706840

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4706840, upper bound: 56.4702285
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4704123, upper bound: 56.4702267
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4702267, upper bound: 56.4704123
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4702285, upper bound: 56.4706856
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4702267, upper bound: 56.4704139
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4704123, upper bound: 56.4703838
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4702267, upper bound: 56.4705370
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4702285, upper bound: 56.4706840
time: 0.55 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.35 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -56.4706840, upper bound: 56.4702285
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -56.4704123, upper bound: 56.4702267
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -56.4702267, upper bound: 56.4704123
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -56.4702285, upper bound: 56.4706856
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -56.4702267, upper bound: 56.4704139
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -56.4704123, upper bound: 56.4703838
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -56.4702267, upper bound: 56.4705370
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -56.4702285, upper bound: 56.4706840

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4581576, upper bound: 56.4574234
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4581576, upper bound: 56.4574234
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4575224, upper bound: 56.4574234
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4576303, upper bound: 56.4577291
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4576303
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4575224
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4581576
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4584248
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4581576, upper bound: 56.4574234
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4581576, upper bound: 56.4574234
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4575224, upper bound: 56.4574234
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4576303, upper bound: 56.4577291
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4576303
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4575224
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4581576
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4584248
time: 0.59 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.42 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4581576, upper bound: 56.4574234
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4581576, upper bound: 56.4574234
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4575224, upper bound: 56.4574234
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4576303, upper bound: 56.4577291
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4576303
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4575224
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4581576
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4584248
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4581576, upper bound: 56.4574234
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4581576, upper bound: 56.4574234
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4575224, upper bound: 56.4574234
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4576303, upper bound: 56.4577291
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4576303
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4575224
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4581576
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 4, lower bound: -56.4574234, upper bound: 56.4584248

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4547339, upper bound: 56.4543401
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4547420, upper bound: 56.4543401
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543563
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4544131, upper bound: 56.4543401
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4544131, upper bound: 56.4544518
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4544131
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4544224
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4544226
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543563, upper bound: 56.4545224
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4547420
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4547339
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4547420, upper bound: 56.4543401
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4545224, upper bound: 56.4543563
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4544226, upper bound: 56.4543401
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4544224, upper bound: 56.4543401
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4544131, upper bound: 56.4544518
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4544131
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4544224
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4544226
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543563, upper bound: 56.4545224
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4547420
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4547339
time: 0.56 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.48 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4547339, upper bound: 56.4543401
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4547420, upper bound: 56.4543401
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543563
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4544131, upper bound: 56.4543401
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4544131, upper bound: 56.4544518
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4544131
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4544224
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4544226
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543563, upper bound: 56.4545224
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4547420
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4547339
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4547420, upper bound: 56.4543401
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4545224, upper bound: 56.4543563
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4544226, upper bound: 56.4543401
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4544224, upper bound: 56.4543401
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4544131, upper bound: 56.4544518
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4544131
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4544224
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4544226
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543563, upper bound: 56.4545224
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4547420
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4543401
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 4, lower bound: -56.4543401, upper bound: 56.4547339

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543167, upper bound: 56.4541246
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543167, upper bound: 56.4541246
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541367
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541420
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4542188, upper bound: 56.4541246
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541261
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542148
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542344
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542104
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4542148, upper bound: 56.4541246
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542185
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542188
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541549
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541420, upper bound: 56.4543159
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541367, upper bound: 56.4541246
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4545204
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4545105
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4543167
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541367
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541420
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541549, upper bound: 56.4541246
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4542188, upper bound: 56.4541246
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541261
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542148
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4542104, upper bound: 56.4542344
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542104
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4542148, upper bound: 56.4541246
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542185
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541261, upper bound: 56.4541246
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542188
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541549
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541420, upper bound: 56.4543159
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4545204
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4545105
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4543167
time: 0.64 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4543167, upper bound: 56.4541246
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4543167, upper bound: 56.4541246
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541367
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541420
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4542188, upper bound: 56.4541246
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541261
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542148
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542344
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542104
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4542148, upper bound: 56.4541246
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542185
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542188
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541549
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541420, upper bound: 56.4543159
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541367, upper bound: 56.4541246
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4545204
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4545105
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4543167
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541367
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541420
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541549, upper bound: 56.4541246
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4542188, upper bound: 56.4541246
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541261
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542148
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4542104, upper bound: 56.4542344
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542104
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4542148, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542185
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541261, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542188
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541549
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541420, upper bound: 56.4543159
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4545204
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4545105
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.15
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4543167

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541206
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541184, upper bound: 56.4541295
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541535, upper bound: 56.4540952
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541535, upper bound: 56.4540952
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540967
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4542185, upper bound: 56.4540952
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541944
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4542309
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4542075
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4542309, upper bound: 56.4541458
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4542185
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4542187
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541535
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4543157
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541184
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4545204
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4545094
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4543146
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4543146, upper bound: 56.4540952
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4545094, upper bound: 56.4540952
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541206
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541295
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4541535, upper bound: 56.4540952
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540967
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541944
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4542309
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4542075, upper bound: 56.4540952
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973
1: -30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767
2: -21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816
3: -20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810
4: -17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4542075
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541458
time: 0.61 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541206
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4541184, upper bound: 56.4541295
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4541535, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4541535, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540967
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4542185, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541944
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4542309
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4542075
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4542309, upper bound: 56.4541458
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4542185
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4542187
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541535
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4543157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541184
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4545204
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4545094
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4543146
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4543146, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4545094, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541206
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541295
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4541535, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540967
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541944
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4540952
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4542309
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4542075, upper bound: 56.4540952
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4542075
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 4, lower bound: -56.4540952, upper bound: 56.4541458
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4542148, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542185
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541261, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4542188
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541549
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541420, upper bound: 56.4543159
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4545204
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4541246
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4545105
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -56.4541246, upper bound: 56.4543167

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.94 + 416.48 = 420.42 seconds
