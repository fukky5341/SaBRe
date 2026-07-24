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
execution time: IAR + RelationalAnalysis = 1.02 + 0.78 = 1.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0135629, upper bound: 0.0135629

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131842, upper bound: 0.0135384
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131842, upper bound: 0.0131842
time: 0.21 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.45 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0131842, upper bound: 0.0135384
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0131842, upper bound: 0.0131842

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131589, upper bound: 0.0135382
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131589, upper bound: 0.0135178
time: 0.20 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126649, upper bound: 0.0126649
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126649, upper bound: 0.0131705
time: 0.22 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.36 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.36
Output dim: 0, lower bound: -0.0131589, upper bound: 0.0135382
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.36
Output dim: 0, lower bound: -0.0131589, upper bound: 0.0135178
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 1.36
Output dim: 0, lower bound: -0.0126649, upper bound: 0.0126649
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.36
Output dim: 0, lower bound: -0.0126649, upper bound: 0.0131705

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120878, upper bound: 0.0127919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126251, upper bound: 0.0127919
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124568, upper bound: 0.0134325
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124568, upper bound: 0.0134122
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124593, upper bound: 0.0131691
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124593, upper bound: 0.0131297
time: 0.21 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.31 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0120878, upper bound: 0.0127919
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0126251, upper bound: 0.0127919
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0124568, upper bound: 0.0134325
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0124568, upper bound: 0.0134122
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0124593, upper bound: 0.0131691
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0124593, upper bound: 0.0131297

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130648, upper bound: 0.0134264
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125499, upper bound: 0.0134152
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131511, upper bound: 0.0134051
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124904, upper bound: 0.0121352
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124731, upper bound: 0.0131415
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124028, upper bound: 0.0130331
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124621, upper bound: 0.0120855
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124384, upper bound: 0.0131296
time: 0.24 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.74 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 0, lower bound: -0.0130648, upper bound: 0.0134264
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 0, lower bound: -0.0125499, upper bound: 0.0134152
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 0, lower bound: -0.0131511, upper bound: 0.0134051
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.74
Output dim: 0, lower bound: -0.0124904, upper bound: 0.0121352
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 0, lower bound: -0.0124731, upper bound: 0.0131415
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 0, lower bound: -0.0124028, upper bound: 0.0130331
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.74
Output dim: 0, lower bound: -0.0124621, upper bound: 0.0120855
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.74
Output dim: 0, lower bound: -0.0124384, upper bound: 0.0131296

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130331, upper bound: 0.0126136
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124495, upper bound: 0.0134126
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124841, upper bound: 0.0126601
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124471, upper bound: 0.0134086
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131246, upper bound: 0.0124730
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124586, upper bound: 0.0133735
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120855, upper bound: 0.0120855
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124730, upper bound: 0.0131246
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122056, upper bound: 0.0126473
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123786, upper bound: 0.0126292
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124618, upper bound: 0.0131199
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124624, upper bound: 0.0120855
time: 0.24 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.38 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.38
Output dim: 0, lower bound: -0.0130331, upper bound: 0.0126136
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.38
Output dim: 0, lower bound: -0.0124495, upper bound: 0.0134126
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.38
Output dim: 0, lower bound: -0.0124841, upper bound: 0.0126601
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.38
Output dim: 0, lower bound: -0.0124471, upper bound: 0.0134086
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.38
Output dim: 0, lower bound: -0.0131246, upper bound: 0.0124730
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.38
Output dim: 0, lower bound: -0.0124586, upper bound: 0.0133735
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.38
Output dim: 0, lower bound: -0.0120855, upper bound: 0.0120855
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.38
Output dim: 0, lower bound: -0.0124730, upper bound: 0.0131246
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.38
Output dim: 0, lower bound: -0.0122056, upper bound: 0.0126473
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.38
Output dim: 0, lower bound: -0.0123786, upper bound: 0.0126292
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.38
Output dim: 0, lower bound: -0.0124618, upper bound: 0.0131199
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.38
Output dim: 0, lower bound: -0.0124624, upper bound: 0.0120855

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124096, upper bound: 0.0123304
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124096, upper bound: 0.0121912
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120971, upper bound: 0.0127496
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121434, upper bound: 0.0127264
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121140, upper bound: 0.0127460
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121395, upper bound: 0.0120214
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0117555, upper bound: 0.0121672
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123862, upper bound: 0.0121708
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0117555, upper bound: 0.0126164
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121580, upper bound: 0.0127132
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121708, upper bound: 0.0123862
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121672, upper bound: 0.0117555
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974
1: -0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598
2: -0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447
3: -0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147
4: -0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121457, upper bound: 0.0123645
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121529, upper bound: 0.0117555
time: 0.24 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.52 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0124096, upper bound: 0.0123304
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0124096, upper bound: 0.0121912
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0120971, upper bound: 0.0127496
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0121434, upper bound: 0.0127264
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0121140, upper bound: 0.0127460
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0121395, upper bound: 0.0120214
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0117555, upper bound: 0.0121672
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0123862, upper bound: 0.0121708
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0117555, upper bound: 0.0126164
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0121580, upper bound: 0.0127132
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0121708, upper bound: 0.0123862
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0121672, upper bound: 0.0117555
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0121457, upper bound: 0.0123645
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.52
Output dim: 0, lower bound: -0.0121529, upper bound: 0.0117555

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.80 + 31.96 = 33.76 seconds
