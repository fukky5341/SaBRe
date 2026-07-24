## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.012884755


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974)
1: (-0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598)
2: (-0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447)
3: (-0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147)
4: (-0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.77 + 0.74 = 1.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0135629, upper bound: 0.0135629

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135450, upper bound: 0.0135599
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135450, upper bound: 0.0135450
time: 0.18 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.45 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0135450, upper bound: 0.0135599
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0135450, upper bound: 0.0135450

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126810, upper bound: 0.0126810
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126810, upper bound: 0.0135452
time: 0.20 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126810, upper bound: 0.0127085
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126810, upper bound: 0.0135348
time: 0.19 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.15 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 1.15
Output dim: 0, lower bound: -0.0126810, upper bound: 0.0126810
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.15
Output dim: 0, lower bound: -0.0126810, upper bound: 0.0135452
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 1.15
Output dim: 0, lower bound: -0.0126810, upper bound: 0.0127085
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.15
Output dim: 0, lower bound: -0.0126810, upper bound: 0.0135348

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124726, upper bound: 0.0134595
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124726, upper bound: 0.0134595
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124726, upper bound: 0.0134472
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124726, upper bound: 0.0134474
time: 0.19 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.17 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.0124726, upper bound: 0.0134595
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.0124726, upper bound: 0.0134595
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.0124726, upper bound: 0.0134472
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.0124726, upper bound: 0.0134474

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124461, upper bound: 0.0134493
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124527, upper bound: 0.0120876
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124461, upper bound: 0.0134493
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124461, upper bound: 0.0124848
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124461, upper bound: 0.0134359
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124483, upper bound: 0.0131248
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124461, upper bound: 0.0134359
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126145, upper bound: 0.0130331
time: 0.20 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.19 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0124461, upper bound: 0.0134493
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0124527, upper bound: 0.0120876
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0124461, upper bound: 0.0134493
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0124461, upper bound: 0.0124848
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0124461, upper bound: 0.0134359
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0124483, upper bound: 0.0131248
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0124461, upper bound: 0.0134359
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0126145, upper bound: 0.0130331

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120855, upper bound: 0.0124624
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124495, upper bound: 0.0134126
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126479, upper bound: 0.0134489
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124586, upper bound: 0.0133735
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124008, upper bound: 0.0134353
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124471, upper bound: 0.0134086
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124730, upper bound: 0.0131246
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124618, upper bound: 0.0131199
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124384, upper bound: 0.0134353
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124447, upper bound: 0.0120855
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126136, upper bound: 0.0130331
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124624, upper bound: 0.0120855
time: 0.21 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.24 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.24
Output dim: 0, lower bound: -0.0120855, upper bound: 0.0124624
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.24
Output dim: 0, lower bound: -0.0124495, upper bound: 0.0134126
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.24
Output dim: 0, lower bound: -0.0126479, upper bound: 0.0134489
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.24
Output dim: 0, lower bound: -0.0124586, upper bound: 0.0133735
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.24
Output dim: 0, lower bound: -0.0124008, upper bound: 0.0134353
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.24
Output dim: 0, lower bound: -0.0124471, upper bound: 0.0134086
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.24
Output dim: 0, lower bound: -0.0124730, upper bound: 0.0131246
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.24
Output dim: 0, lower bound: -0.0124618, upper bound: 0.0131199
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.24
Output dim: 0, lower bound: -0.0124384, upper bound: 0.0134353
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.24
Output dim: 0, lower bound: -0.0124447, upper bound: 0.0120855
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.24
Output dim: 0, lower bound: -0.0126136, upper bound: 0.0130331
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.24
Output dim: 0, lower bound: -0.0124624, upper bound: 0.0120855

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120971, upper bound: 0.0127496
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121434, upper bound: 0.0127264
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0117555, upper bound: 0.0127734
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122808, upper bound: 0.0127734
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0117555, upper bound: 0.0126164
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121580, upper bound: 0.0127132
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120076, upper bound: 0.0127626
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120890, upper bound: 0.0125965
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121140, upper bound: 0.0127460
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121395, upper bound: 0.0120214
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121708, upper bound: 0.0123862
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121672, upper bound: 0.0117555
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121457, upper bound: 0.0123645
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121529, upper bound: 0.0117555
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0117555, upper bound: 0.0127730
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121323, upper bound: 0.0127730
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121912, upper bound: 0.0124096
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123304, upper bound: 0.0124096
time: 0.20 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.22 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0120971, upper bound: 0.0127496
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0121434, upper bound: 0.0127264
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0117555, upper bound: 0.0127734
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0122808, upper bound: 0.0127734
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0117555, upper bound: 0.0126164
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0121580, upper bound: 0.0127132
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0120076, upper bound: 0.0127626
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0120890, upper bound: 0.0125965
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0121140, upper bound: 0.0127460
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0121395, upper bound: 0.0120214
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0121708, upper bound: 0.0123862
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0121672, upper bound: 0.0117555
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0121457, upper bound: 0.0123645
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0121529, upper bound: 0.0117555
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0117555, upper bound: 0.0127730
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0121323, upper bound: 0.0127730
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0121912, upper bound: 0.0124096
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.22
Output dim: 0, lower bound: -0.0123304, upper bound: 0.0124096

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.51 + 27.97 = 29.48 seconds
