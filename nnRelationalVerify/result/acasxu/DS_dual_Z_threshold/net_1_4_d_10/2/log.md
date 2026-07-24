## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.055158916499999995


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479)
1: (-0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910)
2: (-0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850)
3: (-0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681)
4: (-0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.55 + 0.75 = 1.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0559989, upper bound: 0.0559989

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0559573, upper bound: 0.0559608
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0559573, upper bound: 0.0559573
time: 0.19 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.43 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.43
Output dim: 0, lower bound: -0.0559573, upper bound: 0.0559608
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.43
Output dim: 0, lower bound: -0.0559573, upper bound: 0.0559573

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0555729
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557567, upper bound: 0.0559600
time: 0.17 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0559600, upper bound: 0.0557567
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0559570
time: 0.17 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 0.94 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 0.94
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0555729
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 0.94
Output dim: 0, lower bound: -0.0557567, upper bound: 0.0559600
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 0.94
Output dim: 0, lower bound: -0.0559600, upper bound: 0.0557567
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 0.94
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0559570

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558773, upper bound: 0.0553177
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0554600
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555336, upper bound: 0.0549383
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555854, upper bound: 0.0558783
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558783, upper bound: 0.0555854
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0555336
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554600, upper bound: 0.0557573
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0558773
time: 0.18 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 0.96 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 0.96
Output dim: 0, lower bound: -0.0558773, upper bound: 0.0553177
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 0.96
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0554600
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 0.96
Output dim: 0, lower bound: -0.0555336, upper bound: 0.0549383
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 0.96
Output dim: 0, lower bound: -0.0555854, upper bound: 0.0558783
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 0.96
Output dim: 0, lower bound: -0.0558783, upper bound: 0.0555854
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 0.96
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0555336
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 0.96
Output dim: 0, lower bound: -0.0554600, upper bound: 0.0557573
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 0.96
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0558773

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554043, upper bound: 0.0548637
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554484, upper bound: 0.0548890
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0550239
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552851, upper bound: 0.0550239
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550647, upper bound: 0.0545182
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550409, upper bound: 0.0544534
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551489, upper bound: 0.0554709
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549652, upper bound: 0.0550508
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550508, upper bound: 0.0549652
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554709, upper bound: 0.0551489
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0550409
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545182, upper bound: 0.0550647
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0552851
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0553714
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0554484
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548637, upper bound: 0.0554043
time: 0.20 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 0.98 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0554043, upper bound: 0.0548637
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0554484, upper bound: 0.0548890
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0550239
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0552851, upper bound: 0.0550239
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0550647, upper bound: 0.0545182
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0550409, upper bound: 0.0544534
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0551489, upper bound: 0.0554709
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0549652, upper bound: 0.0550508
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0550508, upper bound: 0.0549652
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0554709, upper bound: 0.0551489
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0550409
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0545182, upper bound: 0.0550647
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0552851
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0553714
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0554484
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 0.98
Output dim: 0, lower bound: -0.0548637, upper bound: 0.0554043

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553897, upper bound: 0.0548340
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552708, upper bound: 0.0542730
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554328, upper bound: 0.0548631
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552708, upper bound: 0.0548474
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552409, upper bound: 0.0549288
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552412, upper bound: 0.0549990
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551418, upper bound: 0.0551544
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551334, upper bound: 0.0554542
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554542, upper bound: 0.0551334
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551544, upper bound: 0.0551418
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549990, upper bound: 0.0552412
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0552409
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549996, upper bound: 0.0552656
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549837, upper bound: 0.0553509
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548474, upper bound: 0.0552708
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548631, upper bound: 0.0554328
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0542730, upper bound: 0.0552708
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548340, upper bound: 0.0553897
time: 0.18 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 0.98 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0553897, upper bound: 0.0548340
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0552708, upper bound: 0.0542730
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0554328, upper bound: 0.0548631
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0552708, upper bound: 0.0548474
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0552409, upper bound: 0.0549288
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0552412, upper bound: 0.0549990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0551418, upper bound: 0.0551544
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0551334, upper bound: 0.0554542
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0554542, upper bound: 0.0551334
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0551544, upper bound: 0.0551418
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0549990, upper bound: 0.0552412
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0552409
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0549996, upper bound: 0.0552656
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0549837, upper bound: 0.0553509
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0548474, upper bound: 0.0552708
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0548631, upper bound: 0.0554328
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0542730, upper bound: 0.0552708
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 0.98
Output dim: 0, lower bound: -0.0548340, upper bound: 0.0553897

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548793, upper bound: 0.0548333
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553429, upper bound: 0.0548286
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551987, upper bound: 0.0542719
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544741, upper bound: 0.0548461
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554151, upper bound: 0.0548619
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542189, upper bound: 0.0546111
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552177, upper bound: 0.0548448
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549288
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551723, upper bound: 0.0549096
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549847
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551696, upper bound: 0.0549483
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548820, upper bound: 0.0554362
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550939, upper bound: 0.0550777
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550777, upper bound: 0.0550939
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554362, upper bound: 0.0548820
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549483, upper bound: 0.0551696
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549847, upper bound: 0.0542153
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549096, upper bound: 0.0551723
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0542153
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549590, upper bound: 0.0551949
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549954, upper bound: 0.0542153
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549181, upper bound: 0.0553016
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549836, upper bound: 0.0542961
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548448, upper bound: 0.0552177
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546111, upper bound: 0.0542189
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548619, upper bound: 0.0554151
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548461, upper bound: 0.0544741
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0542719, upper bound: 0.0551987
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548286, upper bound: 0.0553429
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548333, upper bound: 0.0548793
time: 0.19 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.01 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0548793, upper bound: 0.0548333
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0553429, upper bound: 0.0548286
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0551987, upper bound: 0.0542719
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0544741, upper bound: 0.0548461
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0554151, upper bound: 0.0548619
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0542189, upper bound: 0.0546111
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0552177, upper bound: 0.0548448
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549288
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0551723, upper bound: 0.0549096
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549847
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0551696, upper bound: 0.0549483
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0548820, upper bound: 0.0554362
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0550939, upper bound: 0.0550777
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0550777, upper bound: 0.0550939
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0554362, upper bound: 0.0548820
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0549483, upper bound: 0.0551696
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0549847, upper bound: 0.0542153
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0549096, upper bound: 0.0551723
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0542153
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0549590, upper bound: 0.0551949
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0549954, upper bound: 0.0542153
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0549181, upper bound: 0.0553016
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0549836, upper bound: 0.0542961
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0548448, upper bound: 0.0552177
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0546111, upper bound: 0.0542189
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0548619, upper bound: 0.0554151
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0548461, upper bound: 0.0544741
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0542719, upper bound: 0.0551987
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0548286, upper bound: 0.0553429
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.01
Output dim: 0, lower bound: -0.0548333, upper bound: 0.0548793

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552459, upper bound: 0.0547601
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551380, upper bound: 0.0542334
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553143, upper bound: 0.0547911
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545645, upper bound: 0.0544377
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551596, upper bound: 0.0547800
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0548355
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551202, upper bound: 0.0548338
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545126, upper bound: 0.0548673
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551166, upper bound: 0.0548326
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548095, upper bound: 0.0553438
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547575, upper bound: 0.0544061
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544061, upper bound: 0.0547575
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553438, upper bound: 0.0548095
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0551166
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548673, upper bound: 0.0545126
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0551202
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548355, upper bound: 0.0541766
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0551337
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548763, upper bound: 0.0543205
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0552247
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548423, upper bound: 0.0542680
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547800, upper bound: 0.0551596
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544377, upper bound: 0.0545645
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547911, upper bound: 0.0553143
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542334, upper bound: 0.0551380
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547601, upper bound: 0.0552459
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.21 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.05 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0552459, upper bound: 0.0547601
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0551380, upper bound: 0.0542334
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0553143, upper bound: 0.0547911
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0545645, upper bound: 0.0544377
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0551596, upper bound: 0.0547800
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0548355
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0551202, upper bound: 0.0548338
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0545126, upper bound: 0.0548673
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0551166, upper bound: 0.0548326
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0548095, upper bound: 0.0553438
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0547575, upper bound: 0.0544061
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0544061, upper bound: 0.0547575
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0553438, upper bound: 0.0548095
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0551166
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0548673, upper bound: 0.0545126
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0551202
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0548355, upper bound: 0.0541766
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0551337
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0548763, upper bound: 0.0543205
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0552247
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0548423, upper bound: 0.0542680
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0547800, upper bound: 0.0551596
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0544377, upper bound: 0.0545645
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0547911, upper bound: 0.0553143
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0542334, upper bound: 0.0551380
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0547601, upper bound: 0.0552459
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.05
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551278, upper bound: 0.0541406
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552158, upper bound: 0.0547237
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552847, upper bound: 0.0547539
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552784, upper bound: 0.0547548
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551295, upper bound: 0.0547485
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0543165
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0552914
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553133
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553133, upper bound: 0.0547746
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552914, upper bound: 0.0547648
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0551918
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551102
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551295
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0552784
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0552847
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552158
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278
time: 0.20 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.06 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0551278, upper bound: 0.0541406
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0552158, upper bound: 0.0547237
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0552847, upper bound: 0.0547539
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0552784, upper bound: 0.0547548
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0551295, upper bound: 0.0547485
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0543165
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0552914
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553133
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0553133, upper bound: 0.0547746
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0552914, upper bound: 0.0547648
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0551918
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551102
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551295
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0552784
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0552847
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552158
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.06
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548031, upper bound: 0.0546462
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551925, upper bound: 0.0546336
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547348, upper bound: 0.0546752
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552754, upper bound: 0.0546745
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547421, upper bound: 0.0546742
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552719, upper bound: 0.0546742
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546906, upper bound: 0.0552913
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547029, upper bound: 0.0547366
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546961, upper bound: 0.0553076
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547191, upper bound: 0.0547256
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547256, upper bound: 0.0547191
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553076, upper bound: 0.0546961
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547366, upper bound: 0.0547029
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552913, upper bound: 0.0546906
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547001, upper bound: 0.0551635
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547388, upper bound: 0.0548765
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0552719
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0547421
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546745, upper bound: 0.0552754
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546752, upper bound: 0.0547348
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546336, upper bound: 0.0551925
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546462, upper bound: 0.0548031
time: 0.20 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 1.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0548031, upper bound: 0.0546462
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0551925, upper bound: 0.0546336
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0547348, upper bound: 0.0546752
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0552754, upper bound: 0.0546745
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0547421, upper bound: 0.0546742
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0552719, upper bound: 0.0546742
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0546906, upper bound: 0.0552913
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0547029, upper bound: 0.0547366
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0546961, upper bound: 0.0553076
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0547191, upper bound: 0.0547256
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0547256, upper bound: 0.0547191
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0553076, upper bound: 0.0546961
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0547366, upper bound: 0.0547029
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0552913, upper bound: 0.0546906
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0547001, upper bound: 0.0551635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0547388, upper bound: 0.0548765
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0552719
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0547421
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0546745, upper bound: 0.0552754
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0546752, upper bound: 0.0547348
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0546336, upper bound: 0.0551925
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.15
Output dim: 0, lower bound: -0.0546462, upper bound: 0.0548031

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551568, upper bound: 0.0545933
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551563, upper bound: 0.0546168
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551771, upper bound: 0.0546520
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552416, upper bound: 0.0546680
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551809, upper bound: 0.0546525
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552342, upper bound: 0.0546675
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546812, upper bound: 0.0552522
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546719, upper bound: 0.0551706
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546866, upper bound: 0.0552771
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546852, upper bound: 0.0551756
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0546852
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552771, upper bound: 0.0546866
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551706, upper bound: 0.0546719
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552522, upper bound: 0.0546812
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546827, upper bound: 0.0551277
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546882, upper bound: 0.0551322
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546675, upper bound: 0.0552342
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546525, upper bound: 0.0551809
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546680, upper bound: 0.0552416
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546520, upper bound: 0.0551771
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546168, upper bound: 0.0551563
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545933, upper bound: 0.0551568
time: 0.20 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 1.35 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0551568, upper bound: 0.0545933
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0551563, upper bound: 0.0546168
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0551771, upper bound: 0.0546520
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0552416, upper bound: 0.0546680
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0551809, upper bound: 0.0546525
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0552342, upper bound: 0.0546675
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0546812, upper bound: 0.0552522
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0546719, upper bound: 0.0551706
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0546866, upper bound: 0.0552771
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0546852, upper bound: 0.0551756
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0546852
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0552771, upper bound: 0.0546866
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0551706, upper bound: 0.0546719
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0552522, upper bound: 0.0546812
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0546827, upper bound: 0.0551277
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0546882, upper bound: 0.0551322
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0546675, upper bound: 0.0552342
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0546525, upper bound: 0.0551809
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0546680, upper bound: 0.0552416
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0546520, upper bound: 0.0551771
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0546168, upper bound: 0.0551563
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.35
Output dim: 0, lower bound: -0.0545933, upper bound: 0.0551568

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551675, upper bound: 0.0546348
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552314, upper bound: 0.0546536
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551716, upper bound: 0.0546353
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552235, upper bound: 0.0546531
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546658, upper bound: 0.0548326
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546574, upper bound: 0.0552465
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546556, upper bound: 0.0540389
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546475, upper bound: 0.0551575
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546715, upper bound: 0.0550260
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546591, upper bound: 0.0552768
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546704, upper bound: 0.0546886
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546476, upper bound: 0.0551638
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551638, upper bound: 0.0546476
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546886, upper bound: 0.0546704
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552768, upper bound: 0.0546591
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550260, upper bound: 0.0546715
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551575, upper bound: 0.0546475
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0546556
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552465, upper bound: 0.0546574
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0546658
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546531, upper bound: 0.0552235
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546353, upper bound: 0.0551716
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546536, upper bound: 0.0552314
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546348, upper bound: 0.0551675
time: 0.22 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 1.19 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0551675, upper bound: 0.0546348
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0552314, upper bound: 0.0546536
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0551716, upper bound: 0.0546353
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0552235, upper bound: 0.0546531
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0546658, upper bound: 0.0548326
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0546574, upper bound: 0.0552465
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0546556, upper bound: 0.0540389
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0546475, upper bound: 0.0551575
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0546715, upper bound: 0.0550260
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0546591, upper bound: 0.0552768
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0546704, upper bound: 0.0546886
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0546476, upper bound: 0.0551638
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0551638, upper bound: 0.0546476
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0546886, upper bound: 0.0546704
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0552768, upper bound: 0.0546591
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0550260, upper bound: 0.0546715
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0551575, upper bound: 0.0546475
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0546556
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0552465, upper bound: 0.0546574
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0546658
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0546531, upper bound: 0.0552235
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0546353, upper bound: 0.0551716
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0546536, upper bound: 0.0552314
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 11, time: 1.19
Output dim: 0, lower bound: -0.0546348, upper bound: 0.0551675

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549097, upper bound: 0.0537445
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549446, upper bound: 0.0542670
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549755, upper bound: 0.0542626
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549726, upper bound: 0.0543092
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549347, upper bound: 0.0537445
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549487, upper bound: 0.0542674
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549643, upper bound: 0.0542614
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549637, upper bound: 0.0543084
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543084, upper bound: 0.0549538
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542863, upper bound: 0.0549618
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543184, upper bound: 0.0549948
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542928, upper bound: 0.0550198
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542808, upper bound: 0.0549406
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0548219
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548219, upper bound: 0.0537445
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549406, upper bound: 0.0542808
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550198, upper bound: 0.0542928
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549948, upper bound: 0.0543184
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549618, upper bound: 0.0542863
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549538, upper bound: 0.0543084
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543084, upper bound: 0.0549637
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542614, upper bound: 0.0549643
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542674, upper bound: 0.0549487
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0549347
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543092, upper bound: 0.0549726
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542626, upper bound: 0.0549755
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542670, upper bound: 0.0549446
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0549097
time: 0.23 seconds

## Summary of splitting (split count: 11)
- Time for DS candidates: 1.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0549097, upper bound: 0.0537445
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0549446, upper bound: 0.0542670
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0549755, upper bound: 0.0542626
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0549726, upper bound: 0.0543092
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0549347, upper bound: 0.0537445
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0549487, upper bound: 0.0542674
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0549643, upper bound: 0.0542614
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0549637, upper bound: 0.0543084
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0543084, upper bound: 0.0549538
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0542863, upper bound: 0.0549618
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0543184, upper bound: 0.0549948
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0542928, upper bound: 0.0550198
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0542808, upper bound: 0.0549406
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0548219
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0548219, upper bound: 0.0537445
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0549406, upper bound: 0.0542808
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0550198, upper bound: 0.0542928
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0549948, upper bound: 0.0543184
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0549618, upper bound: 0.0542863
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0549538, upper bound: 0.0543084
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0543084, upper bound: 0.0549637
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0542614, upper bound: 0.0549643
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0542674, upper bound: 0.0549487
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0549347
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0543092, upper bound: 0.0549726
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0542626, upper bound: 0.0549755
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0542670, upper bound: 0.0549446
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.72
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0549097

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.29 + 133.94 = 135.23 seconds
