## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.012999239999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621)
1: (-0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673)
2: (0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0190535, 0.0190535)
3: (-0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474)
4: (0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0052041)
5: (-0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063)
6: (-0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656)
7: (-0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616)
8: (-0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768)
9: (0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.73 + 2.55 = 3.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0144436, upper bound: 0.0144436

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143557, upper bound: 0.0142131
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142131, upper bound: 0.0143557
time: 1.24 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.01 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.01
Output dim: 9, lower bound: -0.0143557, upper bound: 0.0142131
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.01
Output dim: 9, lower bound: -0.0142131, upper bound: 0.0143557

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0188742, 0.0189428
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0052041
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143112, upper bound: 0.0141729
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141729, upper bound: 0.0141686
time: 1.33 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0189428, 0.0188742
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0052041
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132224, upper bound: 0.0133638
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132224, upper bound: 0.0133638
time: 1.34 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.40 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 9, lower bound: -0.0143112, upper bound: 0.0141729
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 9, lower bound: -0.0141729, upper bound: 0.0141686
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 9, lower bound: -0.0132224, upper bound: 0.0133638
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 9, lower bound: -0.0132224, upper bound: 0.0133638

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0188697, 0.0188809
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0052041
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143112, upper bound: 0.0141090
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142567, upper bound: 0.0141729
time: 2.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0188123, 0.0189439
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0052041
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141075, upper bound: 0.0137799
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138911, upper bound: 0.0139560
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0189292, 0.0188598
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0052041
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131311, upper bound: 0.0133376
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131965, upper bound: 0.0132805
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0189428, 0.0188606
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0052041
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131311, upper bound: 0.0133376
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131310, upper bound: 0.0132805
time: 1.56 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.84 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 9, lower bound: -0.0143112, upper bound: 0.0141090
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 9, lower bound: -0.0142567, upper bound: 0.0141729
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 9, lower bound: -0.0141075, upper bound: 0.0137799
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 9, lower bound: -0.0138911, upper bound: 0.0139560
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 9, lower bound: -0.0131311, upper bound: 0.0133376
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 9, lower bound: -0.0131965, upper bound: 0.0132805
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 9, lower bound: -0.0131311, upper bound: 0.0133376
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 9, lower bound: -0.0131310, upper bound: 0.0132805

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0188596, 0.0188737
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0052041
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138286, upper bound: 0.0137862
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139780, upper bound: 0.0136644
time: 2.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0188626, 0.0188709
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0052041
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139086, upper bound: 0.0139184
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140267, upper bound: 0.0138351
time: 2.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0184296, 0.0186465
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0050976
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120553, upper bound: 0.0119181
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120553, upper bound: 0.0119181
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0185222, 0.0185612
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0051506, 0.0051670
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129360, upper bound: 0.0129977
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130014, upper bound: 0.0129977
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0183234, 0.0182192
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0050114, 0.0050761
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129018, upper bound: 0.0129503
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127591, upper bound: 0.0131092
time: 2.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0182886, 0.0182575
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0050351, 0.0050552
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128057, upper bound: 0.0130492
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130020, upper bound: 0.0130045
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0183378, 0.0182200
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0050117, 0.0050842
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0122765, upper bound: 0.0124137
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0122765, upper bound: 0.0124137
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0183030, 0.0182575
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0050352, 0.0050633
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129695, upper bound: 0.0129003
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127591, upper bound: 0.0130503
time: 1.54 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.91 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0138286, upper bound: 0.0137862
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0139780, upper bound: 0.0136644
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0139086, upper bound: 0.0139184
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0140267, upper bound: 0.0138351
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0120553, upper bound: 0.0119181
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0120553, upper bound: 0.0119181
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0129360, upper bound: 0.0129977
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0130014, upper bound: 0.0129977
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0129018, upper bound: 0.0129503
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0127591, upper bound: 0.0131092
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0128057, upper bound: 0.0130492
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0130020, upper bound: 0.0130045
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0122765, upper bound: 0.0124137
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0122765, upper bound: 0.0124137
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0129695, upper bound: 0.0129003
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 9, lower bound: -0.0127591, upper bound: 0.0130503

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0187505, 0.0187074
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0052041
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0117055, upper bound: 0.0116540
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0117055, upper bound: 0.0116540
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0186932, 0.0188737
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0051672
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138380, upper bound: 0.0135982
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139263, upper bound: 0.0134617
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0182735, 0.0181417
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0050330, 0.0050866
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133871, upper bound: 0.0136162
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136192, upper bound: 0.0134467
time: 1.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0181334, 0.0183046
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0051290, 0.0049990
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134191, upper bound: 0.0132674
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132956, upper bound: 0.0132674
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0184567, 0.0185612
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0051506, 0.0051446
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0123035, upper bound: 0.0122513
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0123042, upper bound: 0.0122498
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0180267, 0.0178291
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068060, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0047733, 0.0049112
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0119190, upper bound: 0.0121790
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0119190, upper bound: 0.0121790
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0170177, 0.0168530
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0045276, 0.0045724
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120753, upper bound: 0.0121735
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120753, upper bound: 0.0121735
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0168841, 0.0170444
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0045678, 0.0045478
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129699, upper bound: 0.0129679
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129700, upper bound: 0.0129610
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0180096, 0.0178674
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068244, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0047971, 0.0048968
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0119189, upper bound: 0.0121317
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0119718, upper bound: 0.0121317
time: 1.41 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0117055, upper bound: 0.0116540
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0117055, upper bound: 0.0116540
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0138380, upper bound: 0.0135982
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0139263, upper bound: 0.0134617
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0133871, upper bound: 0.0136162
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0136192, upper bound: 0.0134467
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0134191, upper bound: 0.0132674
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0132956, upper bound: 0.0132674
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0123035, upper bound: 0.0122513
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0123042, upper bound: 0.0122498
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0119190, upper bound: 0.0121790
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0119190, upper bound: 0.0121790
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0120753, upper bound: 0.0121735
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0120753, upper bound: 0.0121735
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0129699, upper bound: 0.0129679
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0129700, upper bound: 0.0129610
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0119189, upper bound: 0.0121317
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.13
Output dim: 9, lower bound: -0.0119718, upper bound: 0.0121317

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0180765, 0.0182284
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068235
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0050483, 0.0048857
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0125738, upper bound: 0.0124753
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0125738, upper bound: 0.0124753
time: 2.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0180427, 0.0182632
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068050
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0050692, 0.0048617
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126213, upper bound: 0.0124190
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0125078, upper bound: 0.0124190
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0168158, 0.0165905
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0044741, 0.0045304
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0108139, upper bound: 0.0109045
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0108139, upper bound: 0.0109045
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0167223, 0.0167249
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0044992, 0.0045277
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128315, upper bound: 0.0127186
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128315, upper bound: 0.0127186
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0181091, 0.0181120
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068473
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0050648, 0.0049815
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130413, upper bound: 0.0129946
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131702, upper bound: 0.0128586
time: 2.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0179409, 0.0183046
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0051290, 0.0049347
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131418, upper bound: 0.0132171
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133779, upper bound: 0.0131141
time: 1.26 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 9, lower bound: -0.0125738, upper bound: 0.0124753
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 9, lower bound: -0.0125738, upper bound: 0.0124753
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 9, lower bound: -0.0126213, upper bound: 0.0124190
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 9, lower bound: -0.0125078, upper bound: 0.0124190
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 9, lower bound: -0.0108139, upper bound: 0.0109045
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 9, lower bound: -0.0108139, upper bound: 0.0109045
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 9, lower bound: -0.0128315, upper bound: 0.0127186
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 9, lower bound: -0.0128315, upper bound: 0.0127186
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 9, lower bound: -0.0130413, upper bound: 0.0129946
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 9, lower bound: -0.0131702, upper bound: 0.0128586
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 9, lower bound: -0.0131418, upper bound: 0.0132171
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 9, lower bound: -0.0133779, upper bound: 0.0131141

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0166881, 0.0165644
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068469
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0045060, 0.0044418
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127013, upper bound: 0.0129390
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129775, upper bound: 0.0127975
time: 2.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0165615, 0.0166631
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0045054, 0.0044228
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0125713, upper bound: 0.0123494
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126463, upper bound: 0.0123069
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0173000, 0.0176283
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068405, 0.0067919
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0048060, 0.0046353
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130619, upper bound: 0.0128240
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127840, upper bound: 0.0129735
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0172662, 0.0176580
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0067734
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0048224, 0.0046112
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120158, upper bound: 0.0118745
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0120158, upper bound: 0.0118745
time: 1.02 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.67 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0127013, upper bound: 0.0129390
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0129775, upper bound: 0.0127975
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0125713, upper bound: 0.0123494
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0126463, upper bound: 0.0123069
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0130619, upper bound: 0.0128240
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0127840, upper bound: 0.0129735
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0120158, upper bound: 0.0118745
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0120158, upper bound: 0.0118745

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0168884, 0.0173019
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0067434, 0.0066629
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0046186, 0.0043852
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126673, upper bound: 0.0125315
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124540, upper bound: 0.0124176
time: 2.32 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.65 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.65
Output dim: 9, lower bound: -0.0126673, upper bound: 0.0125315
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.65
Output dim: 9, lower bound: -0.0124540, upper bound: 0.0124176

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.29 + 155.00 = 158.29 seconds
