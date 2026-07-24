## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0016185600000000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013854, 0.0013854)
1: (-0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035157, 0.0035157)
2: (0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021812, 0.0021812)
3: (0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040728, 0.0040728)
4: (-0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035761, 0.0035761)
5: (0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013545, 0.0013545)
6: (0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051689, 0.0051689)
7: (0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036170, 0.0036170)
8: (-0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038780, 0.0038780)
9: (-0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025616, 0.0025616)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.94 + 2.88 = 4.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0020232, upper bound: 0.0020232

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019127, upper bound: 0.0019127
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019127, upper bound: 0.0019127
time: 1.51 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.32 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.32
Output dim: 7, lower bound: -0.0019127, upper bound: 0.0019127
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.32
Output dim: 7, lower bound: -0.0019127, upper bound: 0.0019127

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013796, 0.0013851
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035010, 0.0035150
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021721, 0.0021807
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040720, 0.0040558
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035612, 0.0035754
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013489, 0.0013542
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051679, 0.0051473
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036162, 0.0036019
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038772, 0.0038618
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025509, 0.0025611

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019116, upper bound: 0.0019116
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019116, upper bound: 0.0019116
time: 1.72 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013854, 0.0013796
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035157, 0.0035010
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021812, 0.0021721
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040558, 0.0040728
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035761, 0.0035612
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013545, 0.0013489
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051473, 0.0051689
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036019, 0.0036170
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038618, 0.0038780
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025616, 0.0025509

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019116, upper bound: 0.0019116
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019115, upper bound: 0.0019116
time: 1.56 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.35 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.35
Output dim: 7, lower bound: -0.0019116, upper bound: 0.0019116
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.35
Output dim: 7, lower bound: -0.0019116, upper bound: 0.0019116
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.35
Output dim: 7, lower bound: -0.0019116, upper bound: 0.0019116
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.35
Output dim: 7, lower bound: -0.0019115, upper bound: 0.0019116

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013800, 0.0013857
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035020, 0.0035164
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021727, 0.0021816
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040736, 0.0040569
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035621, 0.0035768
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013492, 0.0013548
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051699, 0.0051487
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036177, 0.0036028
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038787, 0.0038628
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025516, 0.0025621

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018652, upper bound: 0.0017939
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018651
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013802, 0.0013855
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035025, 0.0035160
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021730, 0.0021813
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040731, 0.0040575
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035626, 0.0035763
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013494, 0.0013546
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051693, 0.0051494
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036172, 0.0036033
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038782, 0.0038633
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025520, 0.0025618

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018651, upper bound: 0.0017939
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018652
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013857, 0.0013802
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035165, 0.0035025
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021816, 0.0021730
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040575, 0.0040737
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035768, 0.0035626
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013548, 0.0013494
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051494, 0.0051700
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036033, 0.0036177
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038633, 0.0038788
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025621, 0.0025520

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018652, upper bound: 0.0017939
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018651
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013859, 0.0013800
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035170, 0.0035020
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021819, 0.0021727
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040569, 0.0040742
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035773, 0.0035621
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013550, 0.0013492
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051487, 0.0051707
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036028, 0.0036182
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038628, 0.0038793
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025625, 0.0025516

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018651, upper bound: 0.0017939
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018652
time: 1.24 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.90 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.0018652, upper bound: 0.0017939
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018651
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.0018651, upper bound: 0.0017939
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018652
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.0018652, upper bound: 0.0017939
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018651
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.0018651, upper bound: 0.0017939
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018652

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013154, 0.0013361
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033379, 0.0033906
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020709, 0.0021035
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039278, 0.0038668
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033952, 0.0034488
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012860, 0.0013063
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049849, 0.0049075
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034882, 0.0034340
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037399, 0.0036818
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024320, 0.0024704

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013311, 0.0013210
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033780, 0.0033523
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020957, 0.0020798
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038835, 0.0039132
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034360, 0.0034099
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013015, 0.0012916
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049287, 0.0049664
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034488, 0.0034752
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036977, 0.0037260
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024612, 0.0024425

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013156, 0.0013360
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033384, 0.0033902
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020712, 0.0021033
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039274, 0.0038674
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033957, 0.0034484
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012862, 0.0013062
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049844, 0.0049082
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034878, 0.0034345
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037395, 0.0036824
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024324, 0.0024701

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013314, 0.0013209
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033785, 0.0033519
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020960, 0.0020795
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038830, 0.0039138
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034365, 0.0034094
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013017, 0.0012914
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049280, 0.0049672
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034484, 0.0034758
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036972, 0.0037266
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024616, 0.0024422

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013211, 0.0013314
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033526, 0.0033785
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020799, 0.0020960
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039138, 0.0038838
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034101, 0.0034365
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012917, 0.0013017
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049672, 0.0049290
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034758, 0.0034491
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037266, 0.0036980
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024427, 0.0024616

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013369, 0.0013156
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033926, 0.0033384
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021048, 0.0020712
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038674, 0.0039302
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034509, 0.0033957
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013071, 0.0012862
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049082, 0.0049879
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034345, 0.0034903
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036824, 0.0037421
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024719, 0.0024324

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013213, 0.0013311
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033530, 0.0033780
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020802, 0.0020957
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039132, 0.0038843
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034106, 0.0034360
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012918, 0.0013015
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049664, 0.0049297
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034752, 0.0034496
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037260, 0.0036985
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024431, 0.0024612

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013371, 0.0013154
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033931, 0.0033379
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021051, 0.0020709
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038668, 0.0039308
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034514, 0.0033952
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013073, 0.0012860
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049075, 0.0049887
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034340, 0.0034908
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036818, 0.0037427
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024723, 0.0024320

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
time: 1.54 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.60 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.60
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012857, 0.0013073
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032625, 0.0033175
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020241, 0.0020582
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038431, 0.0037795
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033186, 0.0033744
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012570, 0.0012781
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048774, 0.0047967
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034130, 0.0033565
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036593, 0.0035987
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023771, 0.0024172

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012866, 0.0013071
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032648, 0.0033169
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020255, 0.0020578
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038425, 0.0037821
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033209, 0.0033739
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012579, 0.0012779
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048766, 0.0048000
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034124, 0.0033588
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036587, 0.0036012
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023788, 0.0024167

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013019, 0.0012922
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033037, 0.0032792
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020496, 0.0020344
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037988, 0.0038271
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033604, 0.0033355
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012728, 0.0012634
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048212, 0.0048571
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033736, 0.0033988
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036171, 0.0036440
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024071, 0.0023893

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
time: 1.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013023, 0.0012917
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033049, 0.0032778
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020504, 0.0020336
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037972, 0.0038285
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033616, 0.0033341
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012733, 0.0012629
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048191, 0.0048589
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033722, 0.0034000
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036155, 0.0036454
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024080, 0.0023882

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012858, 0.0013072
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032629, 0.0033171
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020243, 0.0020579
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038427, 0.0037799
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033189, 0.0033740
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012571, 0.0012780
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048769, 0.0047971
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034126, 0.0033568
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036589, 0.0035990
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023774, 0.0024169

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016430
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012867, 0.0013069
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032653, 0.0033165
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020258, 0.0020576
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038420, 0.0037827
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033214, 0.0033735
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012580, 0.0012778
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048760, 0.0048007
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034120, 0.0033593
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036582, 0.0036017
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023791, 0.0024165

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016430
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013020, 0.0012921
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033040, 0.0032788
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020498, 0.0020342
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037983, 0.0038275
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033607, 0.0033351
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012730, 0.0012632
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048205, 0.0048576
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033732, 0.0033991
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036166, 0.0036444
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024073, 0.0023890

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017125
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013025, 0.0012916
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033054, 0.0032775
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020507, 0.0020334
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037968, 0.0038291
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033621, 0.0033338
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012735, 0.0012627
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048187, 0.0048597
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033719, 0.0034006
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036152, 0.0036459
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024084, 0.0023880

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017126
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012914, 0.0013025
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032772, 0.0033054
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020332, 0.0020507
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038291, 0.0037965
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033334, 0.0033621
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012626, 0.0012735
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048597, 0.0048182
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034006, 0.0033715
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036459, 0.0036148
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023878, 0.0024084

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012923, 0.0013020
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032795, 0.0033040
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020346, 0.0020498
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038275, 0.0037991
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033358, 0.0033607
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012635, 0.0012730
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048576, 0.0048215
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033991, 0.0033739
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036444, 0.0036173
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023895, 0.0024073

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013076, 0.0012867
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033183, 0.0032653
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020587, 0.0020258
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037827, 0.0038441
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033753, 0.0033214
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012785, 0.0012580
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048007, 0.0048787
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033593, 0.0034138
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036017, 0.0036602
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024178, 0.0023791

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
time: 1.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013081, 0.0012858
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033195, 0.0032629
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020594, 0.0020243
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037799, 0.0038455
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033765, 0.0033189
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012789, 0.0012571
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047971, 0.0048804
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033568, 0.0034151
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035990, 0.0036615
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024186, 0.0023774

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012915, 0.0013023
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032775, 0.0033049
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020334, 0.0020504
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038285, 0.0037968
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033338, 0.0033616
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012627, 0.0012733
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048589, 0.0048186
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034000, 0.0033719
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036454, 0.0036152
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023880, 0.0024080

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016430
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012925, 0.0013019
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032799, 0.0033037
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020349, 0.0020496
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038271, 0.0037997
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033363, 0.0033604
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012637, 0.0012728
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048571, 0.0048223
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033988, 0.0033744
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036440, 0.0036179
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023898, 0.0024071

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016431
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013078, 0.0012866
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033186, 0.0032648
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020589, 0.0020255
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037821, 0.0038445
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033756, 0.0033209
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012786, 0.0012579
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048000, 0.0048792
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033588, 0.0034142
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036012, 0.0036606
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024180, 0.0023788

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017125
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013083, 0.0012857
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033200, 0.0032625
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020598, 0.0020241
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037795, 0.0038461
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033770, 0.0033186
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012791, 0.0012570
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047967, 0.0048812
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033565, 0.0034156
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035987, 0.0036621
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024190, 0.0023771

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017126
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
time: 1.74 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.62 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016430
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016430
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017125
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017126
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016430
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017125
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017126
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.62
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012062, 0.0012360
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030608, 0.0031365
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018989, 0.0019459
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036334, 0.0035458
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031133, 0.0031903
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011792, 0.0012084
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046113, 0.0045001
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032268, 0.0031489
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034596, 0.0033761
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022301, 0.0022853

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012162, 0.0012278
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030862, 0.0031157
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019147, 0.0019330
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036094, 0.0035752
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031392, 0.0031692
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011890, 0.0012004
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045808, 0.0045374
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032054, 0.0031750
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034367, 0.0034041
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022486, 0.0022702

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012071, 0.0012356
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030631, 0.0031354
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019003, 0.0019452
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036322, 0.0035484
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031157, 0.0031892
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011801, 0.0012080
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046098, 0.0045034
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032257, 0.0031513
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034584, 0.0033787
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022318, 0.0022845

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012178, 0.0012276
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030903, 0.0031152
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019172, 0.0019327
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036088, 0.0035800
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031434, 0.0031687
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011906, 0.0012002
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045800, 0.0045435
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032049, 0.0031793
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034361, 0.0034087
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022516, 0.0022698

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012224, 0.0012221
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031019, 0.0031013
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019244, 0.0019240
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035927, 0.0035934
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031552, 0.0031545
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011951, 0.0011948
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045595, 0.0045605
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031905, 0.0031912
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034208, 0.0034215
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022601, 0.0022596

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012317, 0.0012127
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031256, 0.0030775
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019391, 0.0019093
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035651, 0.0036209
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031793, 0.0031303
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012042, 0.0011857
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045246, 0.0045954
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031661, 0.0032156
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033945, 0.0034477
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022774, 0.0022423

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012228, 0.0012209
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031031, 0.0030983
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019252, 0.0019222
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035892, 0.0035948
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031564, 0.0031515
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011956, 0.0011937
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045552, 0.0045623
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031875, 0.0031925
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034175, 0.0034228
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022610, 0.0022574

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012326, 0.0012122
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031279, 0.0030761
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019406, 0.0019084
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035635, 0.0036235
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031816, 0.0031289
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012051, 0.0011851
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045225, 0.0045987
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031646, 0.0032180
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033930, 0.0034502
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022790, 0.0022413

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012063, 0.0012358
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030611, 0.0031361
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018991, 0.0019456
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036330, 0.0035461
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031137, 0.0031899
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011794, 0.0012082
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046107, 0.0045005
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032263, 0.0031492
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034592, 0.0033765
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022304, 0.0022850

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012164, 0.0012277
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030868, 0.0031154
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019150, 0.0019328
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036090, 0.0035759
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031398, 0.0031688
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011893, 0.0012003
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045803, 0.0045383
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032051, 0.0031756
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034363, 0.0034048
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022491, 0.0022699

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012072, 0.0012353
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030636, 0.0031348
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019006, 0.0019448
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036315, 0.0035490
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031162, 0.0031886
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011803, 0.0012078
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046089, 0.0045041
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032251, 0.0031518
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034578, 0.0033792
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022321, 0.0022841

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012180, 0.0012274
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030909, 0.0031148
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019176, 0.0019324
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036083, 0.0035807
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031440, 0.0031683
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011909, 0.0012000
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045794, 0.0045444
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032045, 0.0031799
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034357, 0.0034094
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022521, 0.0022695

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012225, 0.0012219
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031023, 0.0031008
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019247, 0.0019237
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035921, 0.0035938
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031555, 0.0031540
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011952, 0.0011947
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045588, 0.0045610
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031901, 0.0031916
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034202, 0.0034219
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022603, 0.0022593

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012320, 0.0012126
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031263, 0.0030770
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019396, 0.0019090
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035646, 0.0036216
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031799, 0.0031299
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012045, 0.0011855
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045239, 0.0045963
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031656, 0.0032163
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033941, 0.0034484
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022778, 0.0022420

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012230, 0.0012207
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031037, 0.0030977
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019255, 0.0019218
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035886, 0.0035954
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031569, 0.0031509
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011958, 0.0011935
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045543, 0.0045631
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031869, 0.0031930
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034169, 0.0034234
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022614, 0.0022570

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012328, 0.0012121
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031285, 0.0030758
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019409, 0.0019082
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035631, 0.0036242
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031822, 0.0031286
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012053, 0.0011850
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045220, 0.0045996
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031643, 0.0032186
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033926, 0.0034508
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022795, 0.0022410

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939
time: 1.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012118, 0.0012328
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030752, 0.0031285
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019079, 0.0019409
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036242, 0.0035625
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031280, 0.0031822
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011848, 0.0012053
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045996, 0.0045213
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032186, 0.0031638
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034508, 0.0033921
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022407, 0.0022795

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012218, 0.0012230
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031006, 0.0031037
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019236, 0.0019255
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035954, 0.0035919
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031538, 0.0031569
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011946, 0.0011958
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045631, 0.0045586
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031930, 0.0031899
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034234, 0.0034201
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022591, 0.0022614

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012127, 0.0012320
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030775, 0.0031263
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019093, 0.0019396
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036216, 0.0035652
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031304, 0.0031799
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011857, 0.0012045
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045963, 0.0045246
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032163, 0.0031661
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034484, 0.0033946
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022423, 0.0022778

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012235, 0.0012225
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031048, 0.0031023
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019262, 0.0019247
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035938, 0.0035967
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031581, 0.0031555
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011962, 0.0011952
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045610, 0.0045647
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031916, 0.0031942
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034219, 0.0034246
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022622, 0.0022603

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012281, 0.0012180
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031164, 0.0030909
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019334, 0.0019176
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035807, 0.0036102
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031699, 0.0031440
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012007, 0.0011909
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045444, 0.0045818
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031799, 0.0032061
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034094, 0.0034374
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022706, 0.0022521

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012374, 0.0012072
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031401, 0.0030636
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019481, 0.0019006
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035490, 0.0036376
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031940, 0.0031162
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012098, 0.0011803
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045041, 0.0046166
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031518, 0.0032305
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033792, 0.0034636
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022879, 0.0022321

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012285, 0.0012164
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031176, 0.0030868
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019342, 0.0019150
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035759, 0.0036116
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031711, 0.0031398
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012011, 0.0011893
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045383, 0.0045835
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031756, 0.0032073
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034048, 0.0034388
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022715, 0.0022491

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012383, 0.0012063
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031423, 0.0030611
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019495, 0.0018991
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035461, 0.0036403
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031963, 0.0031137
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012107, 0.0011794
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045005, 0.0046200
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031492, 0.0032328
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033765, 0.0034661
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022895, 0.0022304

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
time: 1.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012120, 0.0012326
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030756, 0.0031279
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019081, 0.0019406
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036235, 0.0035629
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031284, 0.0031816
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011849, 0.0012051
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045987, 0.0045218
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032180, 0.0031641
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034502, 0.0033924
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022409, 0.0022790

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012221, 0.0012228
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031012, 0.0031031
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019240, 0.0019252
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035948, 0.0035926
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031545, 0.0031564
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011948, 0.0011956
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045623, 0.0045595
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031925, 0.0031905
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034228, 0.0034207
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022596, 0.0022610

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012129, 0.0012317
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030780, 0.0031256
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019096, 0.0019391
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036209, 0.0035657
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031309, 0.0031793
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011859, 0.0012042
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045954, 0.0045254
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032156, 0.0031666
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034477, 0.0033951
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022427, 0.0022774

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012237, 0.0012224
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031054, 0.0031019
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019266, 0.0019244
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035934, 0.0035975
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031587, 0.0031552
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011964, 0.0011951
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045605, 0.0045656
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031912, 0.0031948
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034215, 0.0034253
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022626, 0.0022601

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012282, 0.0012178
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031167, 0.0030903
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019336, 0.0019172
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035800, 0.0036106
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031702, 0.0031434
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012008, 0.0011906
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045435, 0.0045823
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031793, 0.0032064
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034087, 0.0034378
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022709, 0.0022516

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012377, 0.0012071
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031407, 0.0030631
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019485, 0.0019003
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035484, 0.0036384
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031946, 0.0031157
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012100, 0.0011801
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045034, 0.0046176
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031513, 0.0032312
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033787, 0.0034643
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022884, 0.0022318

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012287, 0.0012162
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031181, 0.0030862
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019345, 0.0019147
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035752, 0.0036122
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031716, 0.0031392
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012013, 0.0011890
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045374, 0.0045843
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031750, 0.0032079
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034041, 0.0034394
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022719, 0.0022486

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012385, 0.0012062
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031430, 0.0030608
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019499, 0.0018989
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035458, 0.0036410
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031969, 0.0031133
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012109, 0.0011792
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045001, 0.0046209
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031489, 0.0032335
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033761, 0.0034668
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022900, 0.0022301

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939
time: 1.49 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 12.60 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.60
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011709, 0.0011981
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029713, 0.0030402
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018434, 0.0018862
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035220, 0.0034421
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030223, 0.0030924
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011448, 0.0011713
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044698, 0.0043685
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031278, 0.0030569
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033535, 0.0032774
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021649, 0.0022152

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 178

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015928, upper bound: 0.0015039
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015928, upper bound: 0.0015039
time: 1.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011682, 0.0012013
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029646, 0.0030484
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018392, 0.0018912
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035314, 0.0034343
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030155, 0.0031007
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011422, 0.0011745
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044818, 0.0043586
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031362, 0.0030499
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033625, 0.0032700
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021600, 0.0022211

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 178

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015931, upper bound: 0.0015025
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015931, upper bound: 0.0015025
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011816, 0.0011899
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029984, 0.0030195
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018602, 0.0018733
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034980, 0.0034735
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030499, 0.0030714
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011552, 0.0011633
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044394, 0.0044084
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031065, 0.0030848
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033306, 0.0033074
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021847, 0.0022001

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 178

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015625, upper bound: 0.0015328
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015625, upper bound: 0.0015328
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011782, 0.0011929
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029900, 0.0030272
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018550, 0.0018781
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035068, 0.0034637
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030413, 0.0030792
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011520, 0.0011663
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044506, 0.0043959
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031143, 0.0030761
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033391, 0.0032980
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021785, 0.0022056

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 178

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015636, upper bound: 0.0015326
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015636, upper bound: 0.0015326
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011722, 0.0011977
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029745, 0.0030392
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018454, 0.0018855
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035208, 0.0034459
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030256, 0.0030914
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011460, 0.0011709
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044683, 0.0043732
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031267, 0.0030602
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033523, 0.0032810
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021673, 0.0022144

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 178

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015928, upper bound: 0.0015039
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015928, upper bound: 0.0015039
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011691, 0.0012010
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029669, 0.0030477
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018407, 0.0018908
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035306, 0.0034370
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030178, 0.0031000
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011431, 0.0011742
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044808, 0.0043620
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031354, 0.0030523
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033617, 0.0032725
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021617, 0.0022206

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 178

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015931, upper bound: 0.0015025
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015931, upper bound: 0.0015025
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011833, 0.0011897
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030029, 0.0030190
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018630, 0.0018730
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034973, 0.0034787
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030544, 0.0030708
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011569, 0.0011631
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044386, 0.0044149
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031059, 0.0030893
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033300, 0.0033122
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021879, 0.0021997

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 178

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015625, upper bound: 0.0015328
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015625, upper bound: 0.0015328
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011799, 0.0011926
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029941, 0.0030264
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018576, 0.0018776
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035059, 0.0034685
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030455, 0.0030784
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011536, 0.0011660
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044495, 0.0044020
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031135, 0.0030803
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033382, 0.0033026
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021815, 0.0022051

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 178

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015636, upper bound: 0.0015326
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015636, upper bound: 0.0015326
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011872, 0.0011842
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030127, 0.0030050
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018691, 0.0018643
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034812, 0.0034900
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030644, 0.0030566
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011607, 0.0011578
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044181, 0.0044293
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030916, 0.0030994
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033147, 0.0033231
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021951, 0.0021895

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 178

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015339, upper bound: 0.0015637
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015339, upper bound: 0.0015637
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011845, 0.0011875
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030057, 0.0030133
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018648, 0.0018695
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034908, 0.0034820
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030573, 0.0030651
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011580, 0.0011610
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044303, 0.0044191
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031001, 0.0030923
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033238, 0.0033154
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021900, 0.0021956

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 178

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015340, upper bound: 0.0015625
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015340, upper bound: 0.0015625
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011969, 0.0011748
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030373, 0.0029813
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018843, 0.0018496
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034537, 0.0035185
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030894, 0.0030325
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011702, 0.0011486
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043831, 0.0044655
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030671, 0.0031247
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032884, 0.0033502
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022130, 0.0021722

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 178

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015025, upper bound: 0.0015921
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015025, upper bound: 0.0015921
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011938, 0.0011778
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030294, 0.0029890
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018795, 0.0018544
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034626, 0.0035094
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030814, 0.0030403
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011672, 0.0011516
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043944, 0.0044539
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030750, 0.0031166
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032969, 0.0033415
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022073, 0.0021778

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 178

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015039, upper bound: 0.0015915
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015039, upper bound: 0.0015915
time: 1.64 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.29 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015928, upper bound: 0.0015039
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015928, upper bound: 0.0015039
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015931, upper bound: 0.0015025
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015931, upper bound: 0.0015025
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015625, upper bound: 0.0015328
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015625, upper bound: 0.0015328
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015636, upper bound: 0.0015326
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015636, upper bound: 0.0015326
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015928, upper bound: 0.0015039
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015928, upper bound: 0.0015039
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015931, upper bound: 0.0015025
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015931, upper bound: 0.0015025
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015625, upper bound: 0.0015328
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015625, upper bound: 0.0015328
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015636, upper bound: 0.0015326
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015636, upper bound: 0.0015326
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015339, upper bound: 0.0015637
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015339, upper bound: 0.0015637
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015340, upper bound: 0.0015625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015340, upper bound: 0.0015625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015025, upper bound: 0.0015921
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015025, upper bound: 0.0015921
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015039, upper bound: 0.0015915
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.29
Output dim: 7, lower bound: -0.0015039, upper bound: 0.0015915
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.82 + 596.69 = 601.51 seconds
