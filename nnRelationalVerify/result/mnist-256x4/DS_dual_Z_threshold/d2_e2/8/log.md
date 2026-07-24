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
execution time: IAR + RelationalAnalysis = 1.81 + 2.73 = 4.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0144436, upper bound: 0.0144436

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142980, upper bound: 0.0143818
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143818, upper bound: 0.0142980
time: 1.36 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.31 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.31
Output dim: 9, lower bound: -0.0142980, upper bound: 0.0143818
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.31
Output dim: 9, lower bound: -0.0143818, upper bound: 0.0142980

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0184711, 0.0184337
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0051004, 0.0051239
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133151, upper bound: 0.0133685
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133151, upper bound: 0.0133685
time: 1.23 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0184337, 0.0184711
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0051239, 0.0051004
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133685, upper bound: 0.0133151
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133685, upper bound: 0.0133151
time: 1.45 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.54 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 9, lower bound: -0.0133151, upper bound: 0.0133685
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 9, lower bound: -0.0133151, upper bound: 0.0133685
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 9, lower bound: -0.0133685, upper bound: 0.0133151
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 9, lower bound: -0.0133685, upper bound: 0.0133151

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0183750, 0.0183566
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0050538, 0.0050935
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129463, upper bound: 0.0130862
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130013, upper bound: 0.0130036
time: 1.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0183940, 0.0184337
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0051004, 0.0050773
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129463, upper bound: 0.0130862
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130013, upper bound: 0.0130036
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0183532, 0.0183940
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0050773, 0.0050817
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130036, upper bound: 0.0130013
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130862, upper bound: 0.0129463
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0183566, 0.0184711
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0051239, 0.0050538
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130036, upper bound: 0.0130013
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130862, upper bound: 0.0129463
time: 1.21 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.04 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 9, lower bound: -0.0129463, upper bound: 0.0130862
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 9, lower bound: -0.0130013, upper bound: 0.0130036
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 9, lower bound: -0.0129463, upper bound: 0.0130862
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 9, lower bound: -0.0130013, upper bound: 0.0130036
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 9, lower bound: -0.0130036, upper bound: 0.0130013
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 9, lower bound: -0.0130862, upper bound: 0.0129463
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 9, lower bound: -0.0130036, upper bound: 0.0130013
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 9, lower bound: -0.0130862, upper bound: 0.0129463

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0172401, 0.0169996
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0045767, 0.0046640
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128412, upper bound: 0.0128586
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126374, upper bound: 0.0129809
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0170180, 0.0171367
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0046001, 0.0046164
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128980, upper bound: 0.0127435
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127256, upper bound: 0.0128970
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0172314, 0.0170743
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0046218, 0.0046426
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128412, upper bound: 0.0128586
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126374, upper bound: 0.0129809
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0170370, 0.0172114
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0046452, 0.0046002
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128412, upper bound: 0.0127435
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127256, upper bound: 0.0128970
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0171575, 0.0170370
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0046002, 0.0046257
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128970, upper bound: 0.0127256
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127435, upper bound: 0.0128980
time: 1.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0169962, 0.0172314
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0046426, 0.0046046
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129809, upper bound: 0.0126374
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128586, upper bound: 0.0128412
time: 2.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0171367, 0.0171117
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0046453, 0.0046001
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128970, upper bound: 0.0127256
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127435, upper bound: 0.0128980
time: 1.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621
1: -0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673
2: 0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0169996, 0.0173061
3: -0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474
4: 0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0046877, 0.0045767
5: -0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063
6: -0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656
7: -0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616
8: -0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768
9: 0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129809, upper bound: 0.0126374
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128586, upper bound: 0.0128411
time: 2.70 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.96 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0128412, upper bound: 0.0128586
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0126374, upper bound: 0.0129809
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0128980, upper bound: 0.0127435
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0127256, upper bound: 0.0128970
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0128412, upper bound: 0.0128586
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0126374, upper bound: 0.0129809
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0128412, upper bound: 0.0127435
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0127256, upper bound: 0.0128970
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0128970, upper bound: 0.0127256
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0127435, upper bound: 0.0128980
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0129809, upper bound: 0.0126374
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0128586, upper bound: 0.0128412
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0128970, upper bound: 0.0127256
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0127435, upper bound: 0.0128980
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0129809, upper bound: 0.0126374
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.96
Output dim: 9, lower bound: -0.0128586, upper bound: 0.0128411

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.54 + 74.65 = 79.19 seconds
